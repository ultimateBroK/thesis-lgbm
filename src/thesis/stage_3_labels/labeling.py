"""Stage 3: asymmetric direction-barrier labeling.

Encodes labels as ``+1`` (long), ``0`` (hold), ``-1`` (short), with ``-2`` as
censored (insufficient forward horizon).
"""

from __future__ import annotations

import logging
from pathlib import Path

from numba import njit
import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import (
    ATR_HIGH_QUANTILE,
    ATR_LOW_QUANTILE,
    CENSORED_LABEL,
    LABEL_PROFITABILITY_WARN_PCT,
    ROUNDTRIP_MULT,
    SAMPLE_WEIGHT_MIN,
)
from thesis.shared.schemas import FeaturesSchema, LabelsSchema
from thesis.shared.ui import console

logger = logging.getLogger("thesis.labels")


def generate_labels(config: Config) -> None:
    """Compute direction-barrier labels and write parquet output."""
    df, atr_col = _load_inputs(config)
    logger.info("Rows for labeling: %d", len(df))
    _log_atr_stats(df, atr_col, config.labels.min_atr)

    labels, upper, lower, touched_bars, ambiguous_count = _compute_labels(
        close=df["close"].to_numpy(),
        high=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        atr=df[atr_col].to_numpy(),
        tp_mult=config.labels.atr_tp_multiplier,
        sl_mult=config.labels.atr_sl_multiplier,
        horizon=config.labels.horizon_bars,
        min_atr=config.labels.min_atr,
    )

    logger.info(
        "Direction-barrier params: tp_mult=%.2f, sl_mult=%.2f,"
        " horizon=%d, min_atr=%.6f",
        config.labels.atr_tp_multiplier,
        config.labels.atr_sl_multiplier,
        config.labels.horizon_bars,
        config.labels.min_atr,
    )
    logger.info(
        "Ambiguous same-bar both-hit labels: %d (treated as Hold)",
        ambiguous_count,
    )

    event_end = compute_event_end(touched_bars, config.labels.horizon_bars)
    sample_weight = compute_average_uniqueness(event_end)

    df = _merge_label_columns(
        df, labels, upper, lower, touched_bars, event_end, sample_weight
    )
    _log_label_profitability(df, config)
    df = _filter_censored(df)
    _log_distribution(df)
    _log_weight_stats(df)

    out_path = Path(config.paths.labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    right_cols = [c for c in df.columns if c.endswith("_right")]
    assert not right_cols, f"labels.parquet contains join artifacts: {right_cols}"

    LabelsSchema.validate(df, config=config)
    df.write_parquet(out_path)
    logger.info("Labels saved: %s (%d rows)", out_path, len(df))


@njit
def _compute_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    min_atr: float,
) -> tuple:
    """Compute direction-barrier outcomes and touched offsets."""
    n = len(close)
    labels = np.zeros(n, dtype=np.int32)
    upper_barriers = np.zeros(n, dtype=np.float64)
    lower_barriers = np.zeros(n, dtype=np.float64)
    touched_bars = np.full(n, -1, dtype=np.int32)
    ambiguous_count = 0

    for i in range(n):
        a = max(atr[i], min_atr)
        upper = close[i] + tp_mult * a
        lower = close[i] - sl_mult * a
        upper_barriers[i] = upper
        lower_barriers[i] = lower

        if i + horizon >= n:
            labels[i] = CENSORED_LABEL
            touched_bars[i] = CENSORED_LABEL
            continue

        label = 0
        for j in range(i + 1, min(i + 1 + horizon, n)):
            upper_hit = high[j] >= upper
            lower_hit = low[j] <= lower
            if upper_hit and lower_hit:
                ambiguous_count += 1
                touched_bars[i] = j - i
                break
            if upper_hit:
                label = 1
                touched_bars[i] = j - i
                break
            if lower_hit:
                label = -1
                touched_bars[i] = j - i
                break
        labels[i] = label

    return labels, upper_barriers, lower_barriers, touched_bars, ambiguous_count


@njit
def compute_event_end(touched_bars: np.ndarray, horizon: int) -> np.ndarray:
    """Convert touched offsets into absolute end indices."""
    n = len(touched_bars)
    event_end = np.empty(n, dtype=np.int32)
    for i in range(n):
        offset = touched_bars[i]
        if offset < 0:
            offset = horizon
        event_end[i] = i + offset
    return event_end


@njit
def compute_average_uniqueness(event_end: np.ndarray) -> np.ndarray:
    """Compute López de Prado average-uniqueness sample weights."""
    n = len(event_end)
    diff = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        end = event_end[i]
        if end < i:
            end = i
        if end >= n:
            end = n - 1
        diff[i] += 1.0
        diff[end + 1] -= 1.0

    concurrency = np.empty(n, dtype=np.float64)
    running = 0.0
    for i in range(n):
        running += diff[i]
        concurrency[i] = max(running, 1.0)

    inv_prefix = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        inv_prefix[i + 1] = inv_prefix[i] + 1.0 / concurrency[i]

    weights = np.empty(n, dtype=np.float64)
    total = 0.0
    for i in range(n):
        end = event_end[i]
        if end < i:
            end = i
        if end >= n:
            end = n - 1
        span = end - i + 1
        weight = (inv_prefix[end + 1] - inv_prefix[i]) / span
        weight = max(weight, SAMPLE_WEIGHT_MIN)
        weights[i] = weight
        total += weight

    mean = total / n if n > 0 else 1.0
    if mean <= 0.0:
        mean = 1.0
    for i in range(n):
        weights[i] /= mean
    return weights


def _validate_paths(features_path: Path, ohlcv_path: Path) -> None:
    """Raise if features or OHLCV path is missing."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")


def _validate_unique_timestamps(df: pl.DataFrame, name: str) -> None:
    """Raise ValueError on duplicate `timestamp` values."""
    if "timestamp" not in df.columns:
        return
    duplicate_count = len(df) - df["timestamp"].n_unique()
    if duplicate_count > 0:
        raise ValueError(
            f"{name} data contains {duplicate_count} duplicate timestamps; "
            "deduplicate before label generation."
        )


def _load_inputs(config: Config) -> tuple[pl.DataFrame, str]:
    """Load and join features with OHLCV; return (df, atr_col)."""
    features_path = Path(config.paths.features)
    ohlcv_path = Path(config.paths.ohlcv)
    _validate_paths(features_path, ohlcv_path)

    logger.info("Loading features: %s", features_path)
    with console.status(f"[cyan]Loading features[/] {features_path}"):
        df_features = pl.read_parquet(features_path)
    _validate_unique_timestamps(df_features, "features")
    FeaturesSchema.validate(df_features, config=config)

    ohlc_required = {"open", "high", "low", "close"}
    if ohlc_required.issubset(set(df_features.columns)):
        logger.info("Features already contain OHLC columns — skipping OHLCV join")
        return df_features, f"atr_{config.features.atr_period}"

    logger.info("Loading OHLCV for missing OHLC columns: %s", ohlcv_path)
    with console.status(f"[cyan]Loading OHLCV[/] {ohlcv_path}"):
        df_ohlcv = pl.read_parquet(ohlcv_path).select(
            ["timestamp", "open", "high", "low", "close"]
        )
    _validate_unique_timestamps(df_ohlcv, "OHLCV")
    df = df_features.join(df_ohlcv, on="timestamp", how="inner")
    right_cols = [c for c in df.columns if c.endswith("_right")]
    if right_cols:
        logger.warning(
            "Dropping %d join-artifact columns: %s", len(right_cols), right_cols
        )
        df = df.drop(right_cols)
    _validate_unique_timestamps(df, "joined feature/OHLCV")

    atr_col = f"atr_{config.features.atr_period}"
    if atr_col not in df.columns:
        raise ValueError(f"{atr_col} not in features. Run feature engineering first.")
    return df, atr_col


def _merge_label_columns(
    df: pl.DataFrame,
    labels_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    touched_bars_arr: np.ndarray,
    event_end_arr: np.ndarray,
    sample_weight_arr: np.ndarray,
) -> pl.DataFrame:
    """Attach label-related arrays as columns to `df`."""
    return df.with_columns(
        [
            pl.Series("label", labels_arr),
            pl.Series("upper_barrier", upper_arr),
            pl.Series("lower_barrier", lower_arr),
            pl.Series("touched_bar", touched_bars_arr),
            pl.Series("event_end", event_end_arr),
            pl.Series("sample_weight", sample_weight_arr),
        ]
    )


def _log_atr_stats(df: pl.DataFrame, atr_col: str, min_atr: float) -> None:
    """Log a compact ATR distribution snapshot."""
    s = df.select(
        pl.col(atr_col).min().alias("min"),
        pl.col(atr_col).median().alias("median"),
        pl.col(atr_col).quantile(ATR_LOW_QUANTILE).alias("p5"),
        pl.col(atr_col).quantile(ATR_HIGH_QUANTILE).alias("p95"),
        (pl.col(atr_col) < min_atr).mean().alias("floor_rate"),
    ).row(0, named=True)
    logger.info(
        "ATR stats (%s): min=%.6f, median=%.6f, p5=%.6f,"
        " p95=%.6f, below_min_atr=%.2f%%",
        atr_col,
        s["min"] or 0.0,
        s["median"] or 0.0,
        s["p5"] or 0.0,
        s["p95"] or 0.0,
        (s["floor_rate"] or 0.0) * 100.0,
    )


def _filter_censored(df: pl.DataFrame) -> pl.DataFrame:
    """Drop censored rows and regression NaNs."""
    n_before = len(df)
    n_censored = int((df["label"] == CENSORED_LABEL).sum())
    if n_censored > 0:
        df = df.filter(pl.col("label") != CENSORED_LABEL)

    n_nan = 0
    if "regression_target" in df.columns:
        n_nan = int(df["regression_target"].is_nan().sum())
        if n_nan > 0:
            df = df.filter(pl.col("regression_target").is_not_nan())

    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Dropped %d censored rows (label=%d, regression_nan=%d)"
            " — insufficient forward horizon",
            n_dropped,
            n_censored,
            n_nan,
        )
    return df


def _log_distribution(df: pl.DataFrame) -> None:
    """Log counts and percentages for the `label` column."""
    if "label" not in df.columns:
        return
    total = len(df)
    for label, count in df["label"].value_counts().sort("label").iter_rows():
        logger.info("  Class %s: %d (%.1f%%)", label, count, count / total * 100)


def _log_weight_stats(df: pl.DataFrame) -> None:
    """Log sample-weight diagnostics."""
    if "sample_weight" not in df.columns:
        return
    s = df.select(
        pl.col("sample_weight").min().alias("min"),
        pl.col("sample_weight").median().alias("median"),
        pl.col("sample_weight").max().alias("max"),
        pl.col("sample_weight").mean().alias("mean"),
    ).row(0, named=True)
    logger.info(
        "Average-uniqueness sample weights: min=%.4f median=%.4f max=%.4f mean=%.4f",
        s["min"] or 0.0,
        s["median"] or 0.0,
        s["max"] or 0.0,
        s["mean"] or 0.0,
    )


def _log_label_profitability(df: pl.DataFrame, config: Config) -> None:
    """Log label profitability diagnostics after trading costs."""
    if not {"close", "label", "timestamp"}.issubset(df.columns):
        return

    h = config.labels.horizon_bars
    cost = (
        (config.backtest.spread_ticks + config.backtest.slippage_ticks)
        * config.data.tick_size
        + config.backtest.commission_per_lot
        * ROUNDTRIP_MULT
        / config.data.contract_size
    )
    lev = config.backtest.leverage

    result = (
        df.sort("timestamp")
        .with_columns(
            (
                (pl.col("close").shift(-h) - pl.col("close").shift(-1))
                / pl.col("close")
                * lev
                - cost / pl.col("close")
            ).alias("_net_return")
        )
        .filter(
            (pl.col("label") != CENSORED_LABEL) & pl.col("_net_return").is_not_null()
        )
    )

    if result.is_empty():
        logger.warning("Label profitability: no valid samples after filtering.")
        return

    pct: dict[int, float] = {1: 0.0, -1: 0.0}
    for label_val, label_name, profit_expr in (
        (1, "Long", pl.col("_net_return") > 0),
        (-1, "Short", pl.col("_net_return") < 0),
    ):
        class_df = result.filter(pl.col("label") == label_val)
        total = class_df.height
        if total == 0:
            logger.info("  Class %d (%s): no samples", label_val, label_name)
            continue
        profitable = class_df.filter(profit_expr).height
        pct[label_val] = profitable / total * 100.0
        logger.info(
            "%% of %s labels that are profitable after costs: %.1f%% (%d/%d)",
            label_name,
            pct[label_val],
            profitable,
            total,
        )

    hold_total = result.filter(pl.col("label") == 0).height
    if hold_total:
        logger.info("  Class 0 (Hold): %d samples", hold_total)

    if pct[1] < LABEL_PROFITABILITY_WARN_PCT and pct[-1] < LABEL_PROFITABILITY_WARN_PCT:
        logger.warning(
            "LABEL PROFITABILITY LOW: Long %.1f%%, Short %.1f%% -- "
            "labels may not be economically useful after trading costs",
            pct[1],
            pct[-1],
        )
