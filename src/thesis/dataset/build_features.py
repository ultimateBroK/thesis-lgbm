"""Feature engineering: OHLCV bars → enriched feature matrix."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.dataset.feature_blocks import create_model_features
from thesis.shared.config import Config
from thesis.shared.constants import (
    build_feature_output_cols,
    get_static_feature_cols,
    timeframe_to_ms,
)
from thesis.shared.utils import console

logger = logging.getLogger("thesis.dataset.build_features")


def _read_ohlcv_bars(config: Config) -> pl.DataFrame:
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"OHLCV not found: {ohlcv_path}")
    logger.info("Loading OHLCV: %s", ohlcv_path)
    with console.status(f"Loading OHLCV {ohlcv_path}"):
        return pl.read_parquet(ohlcv_path)


def _require_model_features(df: pl.DataFrame, config: Config) -> None:
    """Fail fast if configured features are missing — config/data mismatch."""
    required = set(get_static_feature_cols(config))
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing configured model features: {missing}")


def _select_feature_output(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    if "return_1h" in df.columns and "log_returns" not in df.columns:
        df = df.with_columns(pl.col("return_1h").alias("log_returns"))
    desired = build_feature_output_cols(config)
    existing = [c for c in desired if c in df.columns]
    return df.select(existing)


def _save_feature_list(features_path: Path, feature_cols: list[str]) -> None:
    list_path = features_path.with_suffix(".feature_list.json")
    with open(list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)


def _write_feature_artifacts(
    df: pl.DataFrame, config: Config, model_cols: list[str]
) -> None:
    out_path = Path(config.paths.features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    _save_feature_list(out_path, model_cols)
    logger.info(
        "Features saved: %s (%d columns, %d rows)", out_path, len(df.columns), len(df)
    )
    logger.info("Feature columns (%d): %s", len(model_cols), model_cols)


def _validate_ohlcv_input(df: pl.DataFrame, config: Config) -> None:
    if df.is_empty():
        raise ValueError("OHLCV is empty; cannot generate features")

    n_unsorted = int(
        df.select((pl.col("timestamp").diff().dt.total_milliseconds() < 0).sum()).item()
        or 0
    )
    n_dupes = len(df) - df["timestamp"].n_unique()
    if n_unsorted > 0:
        raise ValueError(f"OHLCV timestamps not sorted ({n_unsorted} reversals)")
    if n_dupes > 0:
        raise ValueError(f"OHLCV timestamps not unique ({n_dupes} duplicates)")

    if len(df) < 2:
        return

    # Weekends/holidays cause expected gaps — log but don't fail
    expected_ms = timeframe_to_ms(config.data.timeframe)
    deltas = (
        df.select(pl.col("timestamp").diff().dt.total_milliseconds().alias("delta_ms"))
        .drop_nulls()
        .get_column("delta_ms")
    )
    gaps = deltas.filter(deltas > expected_ms)
    largest = int(deltas.max() or 0)
    logger.info(
        "Gap check: expected=%d ms, gap_count=%d, largest=%.2f bars",
        expected_ms,
        len(gaps),
        largest / expected_ms if expected_ms else 0.0,
    )


def _drop_warmup_rows(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Drop rows with null/non-finite features from rolling warm-up period."""
    existing = [c for c in feature_cols if c in df.columns]
    n_before = len(df)
    df = df.fill_nan(None).drop_nulls(subset=existing)
    if existing:
        # Div-by-zero in rolling computations can produce inf/-inf
        df = df.filter(pl.all_horizontal(pl.col(c).is_finite() for c in existing))
    dropped = n_before - len(df)
    if dropped > 0:
        logger.info("Dropped %d warm-up rows", dropped)
    if df.is_empty():
        raise ValueError("No feature rows remain after warm-up drop")
    return df


def _validate_feature_quality(df: pl.DataFrame, config: Config) -> None:
    if "label" in df.columns:
        # Label column should not leak into features
        raise ValueError("Features contain 'label' column — data leakage risk")

    p = config.features.rsi_period
    rsi_col = f"rsi_{p}"
    if rsi_col in df.columns:
        rsi = df[rsi_col].drop_nulls()
        if len(rsi) > 0:
            oob = rsi.filter((rsi < 0) | (rsi > 100))
            if len(oob) > 0:
                raise ValueError(f"{rsi_col}: values outside [0, 100]")

    ts = df["timestamp"]
    if ts.n_unique() != len(ts):
        raise ValueError("Features validation failed: timestamp must be unique")
    deltas = ts.diff().drop_nulls().dt.total_milliseconds()
    if int((deltas <= 0).sum()) > 0:
        raise ValueError(
            "Features validation failed: timestamp must be strictly increasing"
        )
    if int(df.null_count().sum_horizontal().sum()) > 0:
        raise ValueError("Features validation failed: null values remain after warm-up")


def build_features(config: Config) -> None:
    """OHLCV → indicator features → validated feature matrix → parquet."""
    df = _read_ohlcv_bars(config)
    logger.info("Input bars: %d", len(df))
    _validate_ohlcv_input(df, config)

    df = create_model_features(df, config)
    _require_model_features(df, config)
    df = _select_feature_output(df, config)
    model_cols = sorted(
        c for c in df.columns if c in set(get_static_feature_cols(config))
    )
    df = _drop_warmup_rows(df, model_cols)
    _validate_feature_quality(df, config)

    _write_feature_artifacts(df, config, model_cols)
