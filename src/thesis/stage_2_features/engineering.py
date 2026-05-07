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

import pandera.polars as pa
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import timeframe_to_ms as _timeframe_to_ms
from thesis.shared.feature_registry import (
    build_feature_output_cols,
    get_gru_feature_cols,
    get_static_feature_cols,
)
from thesis.shared.schemas import FeaturesSchema, OhlcvSchema
from thesis.shared.ui import console
from thesis.stage_2_features.indicators import (
    _add_adx,
    _add_atr,
    _add_context_features,
    _add_ema_crossover,
    _add_ema_slope,
    _add_high_low_range,
    _add_log_returns,
    _add_macd,
    _add_ohlcv_norm,
    _add_price_action_features,
    _add_regime,
    _add_rsi,
    _add_volume_zscore,
)

logger = logging.getLogger("thesis.stage_2_features")


# Public API


def generate_features(config: Config) -> None:
    """Generate and persist feature-enriched OHLCV bars."""
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")

    logger.info("Loading OHLCV: %s", ohlcv_path)
    with console.status(f"[cyan]Loading OHLCV[/] {ohlcv_path}"):
        df = pl.read_parquet(ohlcv_path)
    logger.info("Input bars: %d", len(df))
    OhlcvSchema.validate(df)
    _validate_ohlcv_input(df, config)

    df = _add_atr(df, config)
    df = _add_context_features(df, config)
    df = _add_price_action_features(df, config)
    df = _add_ema_crossover(df, config)
    df = _add_log_returns(df, config)
    df = _add_high_low_range(df, config)
    df = _add_adx(df, config)
    df = _add_ema_slope(df, config)
    df = _add_regime(df)
    df = _add_rsi(df, config)
    df = _add_macd(df, config)
    df = _add_volume_zscore(df, config)
    df = _add_ohlcv_norm(df)

    # GRU pipeline may reference `log_returns` as a legacy column name.
    if "return_1h" in df.columns and "log_returns" not in df.columns:
        df = df.with_columns(pl.col("return_1h").alias("log_returns"))

    desired_cols = build_feature_output_cols(config)
    existing_cols = [c for c in desired_cols if c in df.columns]
    df = df.select(existing_cols)
    keep_features = sorted(
        set(get_static_feature_cols(config)) | set(get_gru_feature_cols(config))
    )
    df = _drop_warmup_rows(df, keep_features)
    _validate_feature_quality(df, config)

    FeaturesSchema.validate(df, config)
    out_path = Path(config.paths.features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    all_model_cols = set(get_static_feature_cols(config)) | set(
        get_gru_feature_cols(config)
    )
    feature_cols = sorted(c for c in df.columns if c in all_model_cols)
    _save_feature_list(out_path, feature_cols)

    logger.info(
        "Features saved: %s (%d columns, %d rows)", out_path, len(df.columns), len(df)
    )
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)


# Validation & warmup helpers


def _validate_ohlcv_input(df: pl.DataFrame, config: Config) -> None:
    """Raise on empty, unsorted, or duplicate timestamps; log gap stats."""
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
    """Drop rows with null/non-finite model-facing features."""
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


def _validate_feature_quality(df: pl.DataFrame, config: Config) -> None:
    """Pandera + timestamp/uniqueness/null checks on the feature dataset."""
    p = config.features.rsi_period
    checks: dict[str, pa.Column] = {
        "timestamp": pa.Column(nullable=False),
    }
    if f"rsi_{p}" in df.columns:
        checks[f"rsi_{p}"] = pa.Column(
            pl.Float64,
            checks=[pa.Check.ge(0), pa.Check.le(100)],
            nullable=True,
            coerce=True,
        )
    schema = pa.DataFrameSchema(checks, strict=False)
    schema.validate(df, lazy=True)

    ts = df.get_column("timestamp")
    if ts.n_unique() != len(ts):
        raise ValueError("Features validation failed: timestamp must be unique")
    deltas = ts.diff().drop_nulls().dt.total_milliseconds()
    if int((deltas <= 0).sum()) > 0:
        raise ValueError(
            "Features validation failed: timestamp must be strictly increasing"
        )
    if int(df.null_count().sum_horizontal().sum()) > 0:
        raise ValueError("Features validation failed: null values remain after warm-up")


def _save_feature_list(features_path: Path, feature_cols: list[str]) -> None:
    """Write a JSON sidecar listing feature column names."""
    list_path = features_path.with_suffix(".feature_list.json")
    with open(list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
