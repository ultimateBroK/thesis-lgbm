"""Feature engineering — production pipeline for price-action features.

The production feature pipeline intentionally stays small and interpretable for
student projects:

- prioritize price structure and trend distance over stacked indicators
- avoid strongly redundant transforms of the same signal
- keep runtime low and behavior stable across runs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import EXCLUDE_COLS as _EXCLUDE_COLS
from thesis.shared.constants import timeframe_to_ms as _timeframe_to_ms
from thesis.shared.ui import console
from thesis.stage_2_features.indicators.core import (
    _add_atr,
    _add_context_features,
    _add_macd,
    _add_rsi,
)
from thesis.stage_2_features.indicators.trend import (
    _add_adx,
    _add_ema_crossover,
    _add_ema_slope,
    _add_high_low_range,
    _add_log_returns,
    _add_ohlcv_norm,
    _add_price_action_features,
    _add_regime,
    _add_volume_zscore,
)

logger = logging.getLogger("thesis.stage_2_features")


# Public API


def generate_features(config: Config) -> None:
    """Generate and persist feature-enriched OHLCV bars.

    Loads OHLCV data from ``config.paths.ohlcv``, computes technical
    indicators and normalized/session features, drops warm-up rows with
    incomplete or non-finite model-facing features, writes the enriched bars
    to ``config.paths.features``, and saves a sidecar JSON file listing the
    produced feature column names.

    Args:
        config: Application configuration containing input/output paths
            and feature parameters.

    Raises:
        FileNotFoundError: If the OHLCV parquet file does not exist.
    """
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading OHLCV: %s", ohlcv_path)
    with console.status(f"[cyan]Loading OHLCV[/] {ohlcv_path}"):
        df = pl.read_parquet(ohlcv_path)
    logger.info("Input bars: %d", len(df))
    _validate_ohlcv_input(df, config)

    # --- Core price-volatility anchor ---
    df = _add_atr(df, config)

    # --- Price-action + session context ---
    df = _add_context_features(df, config)

    # --- Price-action structure ---
    df = _add_price_action_features(df, config)
    df = _add_ema_crossover(df, config)

    df = _add_log_returns(df, config)
    df = _add_high_low_range(df, config)

    # --- Trend quality ---
    df = _add_adx(df, config)
    df = _add_ema_slope(df, config)

    # --- Regime composite ---
    df = _add_regime(df)

    # --- Minimal indicators ---
    df = _add_rsi(df, config)
    df = _add_macd(df, config)
    df = _add_volume_zscore(df, config)

    # --- Normalized raw prices for GRU sequence input ---
    df = _add_ohlcv_norm(df)

    # Backward compatibility: GRU pipeline may request `log_returns`.
    if "return_1h" in df.columns and "log_returns" not in df.columns:
        df = df.with_columns(pl.col("return_1h").alias("log_returns"))

    # Keep compact model-facing features plus raw ATR needed by label barriers.
    keep_features = sorted(
        {
            *config.features.static_feature_cols,
            *config.gru.feature_cols,
        }
    )
    label_helper_cols = [f"atr_{config.features.atr_period}"]
    keep_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        *label_helper_cols,
        *keep_features,
    ]
    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.select(existing_keep_cols)
    df = _drop_warmup_rows(df, keep_features)

    # Persist
    out_path = Path(config.paths.features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    feature_cols = sorted(c for c in df.columns if c not in _EXCLUDE_COLS)
    _save_feature_list(out_path, feature_cols)

    logger.info(
        "Features saved: %s (%d columns, %d rows)", out_path, len(df.columns), len(df)
    )
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)


# Validation & warmup helpers


def _validate_ohlcv_input(df: pl.DataFrame, config: Config) -> None:
    """Log timestamp continuity checks before rolling feature generation.

    Args:
        df: OHLCV DataFrame to validate.
        config: Application configuration.

    Raises:
        ValueError: If required columns are missing, DataFrame is empty,
            timestamps are unsorted, or timestamps are not unique.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"OHLCV missing required columns: {missing}")
    if df.is_empty():
        raise ValueError("OHLCV is empty; cannot generate features")

    unsorted = int(
        df.select((pl.col("timestamp").diff().dt.total_milliseconds() < 0).sum()).item()
        or 0
    )
    duplicate_count = len(df) - df.get_column("timestamp").n_unique()
    if unsorted > 0:
        raise ValueError(f"OHLCV timestamps are not sorted ({unsorted} reversals)")
    if duplicate_count > 0:
        raise ValueError(
            f"OHLCV timestamps are not unique ({duplicate_count} duplicates)"
        )

    if len(df) < 2:
        return

    expected_ms = _timeframe_to_ms(config.data.timeframe)
    deltas = (
        df.select(
            (pl.col("timestamp").diff().dt.total_milliseconds()).alias("delta_ms")
        )
        .drop_nulls()
        .get_column("delta_ms")
    )
    gaps = deltas.filter(deltas > expected_ms)
    largest = int(deltas.max() or 0)
    logger.info(
        "Feature input gap check: expected_delta=%d ms, gap_count=%d, "
        "largest_gap=%.2f bars",
        expected_ms,
        len(gaps),
        largest / expected_ms if expected_ms else 0.0,
    )


def _drop_warmup_rows(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Drop rows whose model-facing features are incomplete or non-finite.

    Args:
        df: Feature DataFrame.
        feature_cols: Column names that must be finite and non-null.

    Returns:
        DataFrame with warm-up rows removed.

    Raises:
        ValueError: If no rows remain after warm-up removal.
    """
    existing_features = [c for c in feature_cols if c in df.columns]
    n_before = len(df)
    df = df.fill_nan(None).drop_nulls(subset=existing_features)
    if existing_features:
        finite_expr = pl.all_horizontal(
            [pl.col(c).is_finite() for c in existing_features]
        )
        df = df.filter(finite_expr)
    dropped = n_before - len(df)
    if dropped > 0:
        logger.info(
            "Dropped %d warm-up rows with incomplete model-facing features",
            dropped,
        )
    if df.is_empty():
        raise ValueError("No feature rows remain after dropping warm-up rows")
    return df


def _save_feature_list(features_path: Path, feature_cols: list[str]) -> None:
    """Write a JSON sidecar listing feature column names.

    Args:
        features_path: Path to the features parquet file (sidecar is written
            alongside with ``.feature_list.json`` suffix).
        feature_cols: List of feature column names to save.
    """
    list_path = features_path.with_suffix(".feature_list.json")
    with open(list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
