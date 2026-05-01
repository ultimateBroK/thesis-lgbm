"""Stage 2: symmetric upper/lower barrier direction labeling.

Uses a single ``atr_multiplier`` for all hours. No DST detection,
no session definitions, no dead-hour filtering.

Classes:
    +1  Long  (upper barrier hit first)
     0  Hold  (neither barrier hit within horizon)
    -1  Short (lower barrier hit first)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from thesis.config import Config

logger = logging.getLogger("thesis.labels")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_labels(config: Config) -> None:
    """
    Generate triple-barrier labels and write them to the configured labels path.

    Loads features and OHLCV parquet files, joins them on `timestamp`, validates the presence of the ATR feature named `atr_{atr_period}`, computes symmetric upper/lower barrier direction labels using `config.labels` parameters (`atr_multiplier`, `horizon_bars`, `min_atr`), merges label columns (`label`, `upper_barrier`, `lower_barrier`, `touched_bar`) into the dataset, logs the label distribution, and persists the result to `config.paths.labels`.

    Args:
        config (Config): Application configuration containing:
            - paths.features: path to features parquet
            - paths.ohlcv: path to OHLCV parquet
            - paths.labels: output path for labels parquet
            - features.atr_period: integer ATR period (used to form `atr_{period}` column)
            - labels.atr_multiplier: ATR multiplier for TP/SL
            - labels.horizon_bars: forward horizon in bars
            - labels.min_atr: minimum ATR value to use

    Raises:
        FileNotFoundError: If the features or OHLCV input paths do not exist.
        ValueError: If the required ATR column (`atr_{atr_period}`) is missing from the features.
    """
    features_path = Path(config.paths.features)
    ohlcv_path = Path(config.paths.ohlcv)
    _validate_paths(features_path, ohlcv_path)

    logger.info("Loading features: %s", features_path)
    df_feat = pl.read_parquet(features_path)

    logger.info("Loading OHLCV: %s", ohlcv_path)
    df_ohlcv = pl.read_parquet(ohlcv_path).select(
        ["timestamp", "open", "high", "low", "close"]
    )

    df = df_feat.join(df_ohlcv, on="timestamp", how="inner")
    logger.info("Joined rows: %d", len(df))

    atr_col = f"atr_{config.features.atr_period}"
    if atr_col not in df.columns:
        raise ValueError(f"{atr_col} not in features. Run feature engineering first.")

    _log_atr_stats(df, atr_col, config.labels.min_atr)

    labels_arr, upper_arr, lower_arr, touched_bars_arr = _compute_labels(
        close=df["close"].to_numpy(),
        high=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        atr=df[atr_col].to_numpy(),
        mult=config.labels.atr_multiplier,
        horizon=config.labels.horizon_bars,
        min_atr=config.labels.min_atr,
    )

    logger.info(
        "Triple-barrier params: mult=%.2f, horizon=%d, min_atr=%.6f",
        config.labels.atr_multiplier,
        config.labels.horizon_bars,
        config.labels.min_atr,
    )

    df = _merge_label_columns(df, labels_arr, upper_arr, lower_arr, touched_bars_arr)
    df = _filter_censored(df)
    _log_distribution(df)

    out_path = Path(config.paths.labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Labels saved: %s (%d rows)", out_path, len(df))


# ---------------------------------------------------------------------------
# Core labeling logic
# ---------------------------------------------------------------------------


@njit
def _compute_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    mult: float,
    horizon: int,
    min_atr: float,
) -> tuple:
    """
    Compute symmetric direction-barrier outcomes for each bar.

    For each index i this sets upper = close[i] + mult * max(atr[i], min_atr) and lower = close[i] - mult * max(atr[i], min_atr), then inspects bars i+1 .. i+horizon (bounded by series end) to determine which barrier is touched first. If neither barrier is touched within the horizon the label remains 0. If both barriers are touched on the same OHLC bar, the sample is treated as ambiguous and labeled Hold (0). Rows within `horizon` bars of the series end are marked -2 (censored) and are dropped from training.

    Returns:
        dict: A dictionary with the following keys:
            - "labels" (np.ndarray[int32]): per-bar labels where 1 = upper barrier hit, -1 = lower barrier hit, 0 = Hold, -2 = censored (right-censored, insufficient forward bars).
            - "upper_barriers" (np.ndarray[float64]): upper barrier price set at each bar.
            - "lower_barriers" (np.ndarray[float64]): lower barrier price set at each bar.
            - "touched_bars" (np.ndarray[int32]): number of bars forward until the barrier was touched; -1 if not touched, -2 if censored.
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)
    upper_barriers = np.zeros(n, dtype=np.float64)
    lower_barriers = np.zeros(n, dtype=np.float64)
    touched_bars = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        a = max(atr[i], min_atr)
        upper = close[i] + mult * a
        lower = close[i] - mult * a
        upper_barriers[i] = upper
        lower_barriers[i] = lower

        # Right-censored: not enough forward bars to evaluate horizon
        if i + horizon >= n:
            labels[i] = -2  # Special marker: censored (excluded from training)
            touched_bars[i] = -2
            continue

        label = 0  # Hold by default
        for j in range(i + 1, min(i + 1 + horizon, n)):
            upper_hit = high[j] >= upper
            lower_hit = low[j] <= lower
            if upper_hit and lower_hit:
                # OHLC bars do not reveal intra-bar path; keep ambiguous samples neutral.
                break
            if upper_hit:
                label = 1  # Long
                touched_bars[i] = j - i
                break
            if lower_hit:
                label = -1  # Short
                touched_bars[i] = j - i
                break
        labels[i] = label

    return labels, upper_barriers, lower_barriers, touched_bars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_paths(features_path: Path, ohlcv_path: Path) -> None:
    """Validate that required input paths exist."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")


def _merge_label_columns(
    df: pl.DataFrame,
    labels_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    touched_bars_arr: np.ndarray,
) -> pl.DataFrame:
    """Build and join label columns into the main dataframe.

    ``tp_price`` and ``sl_price`` are retained as backward-compatible aliases
    for the upper/lower direction barriers.
    """
    ts_dtype = df["timestamp"].dtype
    labels_df = pl.DataFrame(
        {
            "timestamp": pl.Series(df["timestamp"].to_list(), dtype=ts_dtype),
            "label": labels_arr,
            "upper_barrier": upper_arr,
            "lower_barrier": lower_arr,
            "tp_price": upper_arr,
            "sl_price": lower_arr,
            "touched_bar": touched_bars_arr,
        }
    )
    return df.join(labels_df, on="timestamp", how="left")


def _log_atr_stats(df: pl.DataFrame, atr_col: str, min_atr: float) -> None:
    """Log ATR distribution and how often the configured floor applies."""
    stats = df.select(
        [
            pl.col(atr_col).min().alias("min"),
            pl.col(atr_col).median().alias("median"),
            pl.col(atr_col).quantile(0.05).alias("p5"),
            pl.col(atr_col).quantile(0.95).alias("p95"),
            (pl.col(atr_col) < min_atr).mean().alias("floor_rate"),
        ]
    ).row(0, named=True)
    logger.info(
        "ATR stats (%s): min=%.6f, median=%.6f, p5=%.6f, p95=%.6f, "
        "below_min_atr=%.2f%%",
        atr_col,
        stats["min"] or 0.0,
        stats["median"] or 0.0,
        stats["p5"] or 0.0,
        stats["p95"] or 0.0,
        (stats["floor_rate"] or 0.0) * 100,
    )


def _filter_censored(df: pl.DataFrame) -> pl.DataFrame:
    """Remove censored rows (label == -2) where forward horizon is insufficient.

    Censored rows lack enough future data to evaluate the triple-barrier outcome.
    Keeping them as Hold would inject label noise, so they are dropped entirely.
    """
    n_censored = int((df["label"] == -2).sum())
    if n_censored <= 0:
        return df
    logger.info("Dropping %d censored rows (insufficient forward horizon)", n_censored)
    return df.filter(pl.col("label") != -2)


def _log_distribution(df: pl.DataFrame) -> None:
    """
    Log counts and percentages for each value in the dataframe's `label` column.

    If the `label` column is not present the function returns without logging. Each logged line reports the label value, its absolute count, and its percentage of the dataframe rows.

    Args:
        df (pl.DataFrame): DataFrame expected to contain a `label` column.
    """
    if "label" not in df.columns:
        return
    counts = df["label"].value_counts().sort("label")
    total = len(df)
    for row in counts.iter_rows():
        label, count = row
        logger.info("  Class %s: %d (%.1f%%)", label, count, count / total * 100)
