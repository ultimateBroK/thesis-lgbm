"""Triple-barrier labeling for directional trading signals.

Assigns +1 (long) / 0 (hold) / -1 (short) / -2 (censored) labels
using ATR-scaled profit-taking and stop-loss barriers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from thesis.dataset._label_numba import (
    compute_average_uniqueness,
    compute_event_end,
)
from thesis.shared.config import Config
from thesis.shared.constants import (
    ATR_HIGH_QUANTILE,
    ATR_LOW_QUANTILE,
    CENSORED_LABEL,
)
from thesis.shared.utils import SimpleConsole

logger = logging.getLogger("thesis.labels")
_console = SimpleConsole()


def _check_unique_timestamps(df: pl.DataFrame, name: str) -> None:
    if "timestamp" not in df.columns:
        return
    dup_count = len(df) - df["timestamp"].n_unique()
    if dup_count > 0:
        raise ValueError(
            f"{name} has {dup_count} duplicate timestamps — deduplicate first."
        )


def _drop_join_artifacts(df: pl.DataFrame) -> pl.DataFrame:
    """Polars inner join can leave _right suffix columns — remove them."""
    right_cols = [c for c in df.columns if c.endswith("_right")]
    if right_cols:
        logger.warning(
            "Dropping %d join-artifact columns: %s",
            len(right_cols),
            right_cols,
        )
        df = df.drop(right_cols)
    _check_unique_timestamps(df, "joined feature/OHLCV")
    return df


def _load_features_and_ohlcv(config: Config) -> tuple[pl.DataFrame, str]:
    """Features parquet → join OHLCV only when OHLC columns absent."""
    features_path = Path(config.paths.features)
    ohlcv_path = Path(config.paths.ohlcv)

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading features: %s", features_path)
    with _console.status(f"Loading features {features_path}"):
        df_features = pl.read_parquet(features_path)
    _check_unique_timestamps(df_features, "features")

    ohlc_cols = {"open", "high", "low", "close"}
    if ohlc_cols.issubset(set(df_features.columns)):
        logger.info("Features already contain OHLC — skipping OHLCV join")
        atr_col = f"atr_{config.features.atr_period}"
        if atr_col not in df_features.columns:
            raise ValueError(f"{atr_col} missing. Run feature engineering first.")
        return df_features, atr_col

    logger.info("Loading OHLCV: %s", ohlcv_path)
    with _console.status(f"Loading OHLCV {ohlcv_path}"):
        df_ohlcv = pl.read_parquet(ohlcv_path).select(
            ["timestamp", "open", "high", "low", "close"]
        )
    _check_unique_timestamps(df_ohlcv, "OHLCV")

    df = df_features.join(df_ohlcv, on="timestamp", how="inner")
    df = _drop_join_artifacts(df)

    atr_col = f"atr_{config.features.atr_period}"
    if atr_col not in df.columns:
        raise ValueError(f"{atr_col} missing. Run feature engineering first.")
    return df, atr_col


def _compute_triple_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    min_atr: float,
) -> tuple:
    from thesis.dataset._label_numba import compute_labels as _numba_labels

    return _numba_labels(close, high, low, atr, tp_mult, sl_mult, horizon, min_atr)


def _attach_label_columns(
    df: pl.DataFrame,
    labels: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    touched: np.ndarray,
    event_end: np.ndarray,
    weights: np.ndarray,
) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.Series("label", labels),
            pl.Series("upper_barrier", upper),
            pl.Series("lower_barrier", lower),
            pl.Series("touched_bar", touched),
            pl.Series("event_end", event_end),
            pl.Series("sample_weight", weights),
        ]
    )


def _drop_censored_and_nan(df: pl.DataFrame) -> pl.DataFrame:
    """Censored labels (simultaneous barrier hits) are untradeable — drop them."""
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
            "Dropped %d rows (censored=%d regression_nan=%d) — insufficient horizon",
            n_dropped,
            n_censored,
            n_nan,
        )
    return df


def _validate_no_join_artifacts(df: pl.DataFrame) -> None:
    right_cols = [c for c in df.columns if c.endswith("_right")]
    if right_cols:
        raise ValueError(f"labels.parquet contains join artifacts: {right_cols}")


def _log_atr_stats(df: pl.DataFrame, atr_col: str, min_atr: float) -> None:
    s = df.select(
        pl.col(atr_col).min().alias("min"),
        pl.col(atr_col).median().alias("median"),
        pl.col(atr_col).quantile(ATR_LOW_QUANTILE).alias("p5"),
        pl.col(atr_col).quantile(ATR_HIGH_QUANTILE).alias("p95"),
        (pl.col(atr_col) < min_atr).mean().alias("floor_rate"),
    ).row(0, named=True)
    logger.info(
        "ATR (%s): min=%.6f median=%.6f p5=%.6f p95=%.6f below_min=%.2f%%",
        atr_col,
        s["min"] or 0.0,
        s["median"] or 0.0,
        s["p5"] or 0.0,
        s["p95"] or 0.0,
        (s["floor_rate"] or 0.0) * 100.0,
    )


def _log_distribution(df: pl.DataFrame) -> None:
    if "label" not in df.columns:
        return
    total = len(df)
    for label, count in df["label"].value_counts().sort("label").iter_rows():
        logger.info("  Class %s: %d (%.1f%%)", label, count, count / total * 100)


def _log_weight_stats(df: pl.DataFrame) -> None:
    if "sample_weight" not in df.columns:
        return
    s = df.select(
        pl.col("sample_weight").min().alias("min"),
        pl.col("sample_weight").median().alias("median"),
        pl.col("sample_weight").max().alias("max"),
        pl.col("sample_weight").mean().alias("mean"),
    ).row(0, named=True)
    logger.info(
        "Sample weights: min=%.4f median=%.4f max=%.4f mean=%.4f",
        s["min"] or 0.0,
        s["median"] or 0.0,
        s["max"] or 0.0,
        s["mean"] or 0.0,
    )


def build_labels(config: Config) -> None:
    """Features → triple-barrier labels → uniqueness weights → parquet."""
    df, atr_col = _load_features_and_ohlcv(config)
    _log_atr_stats(df, atr_col, config.labels.min_atr)

    labels, upper, lower, touched, _ambiguous = _compute_triple_barrier(
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
        "tp_mult=%.2f sl_mult=%.2f horizon=%d min_atr=%.6f",
        config.labels.atr_tp_multiplier,
        config.labels.atr_sl_multiplier,
        config.labels.horizon_bars,
        config.labels.min_atr,
    )

    event_end = compute_event_end(touched, config.labels.horizon_bars)
    weights = compute_average_uniqueness(event_end)

    df = _attach_label_columns(df, labels, upper, lower, touched, event_end, weights)
    df = _drop_censored_and_nan(df)
    _log_distribution(df)
    _log_weight_stats(df)

    _validate_no_join_artifacts(df)

    out_path = Path(config.paths.labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("Labels saved: %s (%d rows)", out_path, len(df))
