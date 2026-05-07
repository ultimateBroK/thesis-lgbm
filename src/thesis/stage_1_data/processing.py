"""Data preparation — aggregate raw tick data to OHLCV bars.

Reads monthly tick parquet files from data/raw/XAUUSD/, computes quote
microprice OHLCV bars at the configured timeframe, and saves to
data/processed/ohlcv.parquet.

Memory-efficient: aggregates each monthly file independently, then concats
only the small OHLCV results (~56K rows for 8 years of 1H bars).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import timeframe_to_ms as _timeframe_to_ms
from thesis.shared.data_quality import (
    check_candle_quality,
    check_gap_report,
    classify_calendar_gaps,
)

logger = logging.getLogger("thesis.prepare")


def _parse_datetime_bound(value: str, name: str, dtype: pl.DataType) -> pl.Expr:
    """Parse an inclusive datetime bound from config into a Polars expression."""
    if not value:
        raise ValueError(f"config.data.{name} must not be empty")
    return pl.lit(value).str.to_datetime().cast(dtype)


def _aggregate_file(file_path: Path, group_every: str) -> pl.DataFrame:
    """Aggregate one monthly tick parquet file into OHLCV bars."""
    ticks = pl.read_parquet(
        file_path,
        columns=["timestamp", "bid", "ask", "ask_volume", "bid_volume"],
    )

    n_before = len(ticks)
    ticks = ticks.filter(
        (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
        & (pl.col("ask") >= pl.col("bid"))
        & (pl.col("ask_volume") >= 0)
        & (pl.col("bid_volume") >= 0)
    )
    dropped_quotes = n_before - len(ticks)
    if dropped_quotes > 0:
        logger.warning(
            "%s: dropped %d invalid quote ticks (bid/ask/spread/volume)",
            file_path.name,
            dropped_quotes,
        )

    # Microprice: volume-weighted by opposing-side quote sizes (not midpoint).
    ticks = ticks.with_columns(
        (
            (
                pl.col("ask") * pl.col("bid_volume")
                + pl.col("bid") * pl.col("ask_volume")
            )
            / (pl.col("ask_volume") + pl.col("bid_volume") + 1e-10)
        ).alias("microprice"),
        (pl.col("ask_volume") + pl.col("bid_volume")).alias("volume"),
    )

    ticks = ticks.sort("timestamp")

    return ticks.group_by_dynamic(
        "timestamp",
        every=group_every,
        period=group_every,
        closed="left",
        label="left",
        start_by="window",
    ).agg(
        pl.col("microprice").first().alias("open"),
        pl.col("microprice").max().alias("high"),
        pl.col("microprice").min().alias("low"),
        pl.col("microprice").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("microprice").count().alias("tick_count"),
        (pl.col("ask") - pl.col("bid")).mean().alias("avg_spread"),
    )


def _deduplicate_and_filter(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, int, int]:
    """Deduplicate OHLCV bars and filter corrupted timestamps."""
    duplicate_count = len(ohlcv) - ohlcv.get_column("timestamp").n_unique()
    if duplicate_count > 0:
        logger.warning(
            "Found %d duplicate OHLCV bar timestamps before dedup; keeping first. "
            "Likely overlapping raw files; downstream stages now validate uniqueness.",
            duplicate_count,
        )
    ohlcv = ohlcv.unique(subset=["timestamp"], keep="first").sort("timestamp")
    n_before = len(ohlcv)
    ohlcv = ohlcv.filter(
        (pl.col("timestamp").dt.year() >= 2000)
        & (pl.col("timestamp").dt.year() <= 2100)
    )
    dropped = n_before - len(ohlcv)
    if dropped > 0:
        logger.warning("Dropped %d bars with corrupted timestamps", dropped)
    return ohlcv, dropped, duplicate_count


def _filter_date_range(ohlcv: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Apply the configured inclusive date bounds to OHLCV bars."""
    n_before = len(ohlcv)
    ts_dtype = ohlcv["timestamp"].dtype
    start = _parse_datetime_bound(config.data.start_date, "start_date", ts_dtype)
    end = _parse_datetime_bound(config.data.end_date, "end_date", ts_dtype)
    ohlcv = ohlcv.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    dropped = n_before - len(ohlcv)
    if dropped > 0:
        logger.info(
            "Dropped %d bars outside configured range [%s, %s]",
            dropped,
            config.data.start_date,
            config.data.end_date,
        )
    if ohlcv.is_empty():
        raise ValueError(
            "No OHLCV bars remain after applying configured data date range "
            f"[{config.data.start_date}, {config.data.end_date}]"
        )
    return ohlcv


def _log_gap_report(ohlcv: pl.DataFrame, group_ms: int) -> None:
    """Log timestamp continuity diagnostics via shared data-quality checks."""
    if len(ohlcv) < 2:
        logger.warning("OHLCV gap report skipped: fewer than 2 bars")
        return

    result = check_gap_report(ohlcv, group_ms)

    diffs = (
        ohlcv.select(
            (pl.col("timestamp").diff().dt.total_milliseconds()).alias("delta_ms")
        )
        .drop_nulls()
        .get_column("delta_ms")
    )
    non_increasing = int((diffs <= 0).sum())

    logger.info(
        "OHLCV calendar gap report: expected_delta=%d ms, calendar_gap_count=%d, "
        "estimated_missing_bars=%d, largest_gap=%.2f bars, non_increasing_deltas=%d",
        group_ms,
        result["gap_count"],
        result["estimated_missing_bars"],
        result["largest_gap_bars"],
        non_increasing,
    )


def _log_candle_quality_report(ohlcv: pl.DataFrame) -> None:
    """Log OHLCV candle integrity and outlier diagnostics."""
    if ohlcv.is_empty():
        return

    result = check_candle_quality(ohlcv)
    if result["invalid_count"] > 0:
        logger.warning(
            "OHLCV quality: %d invalid candles detected",
            result["invalid_count"],
        )

    stats = ohlcv.select(
        (pl.col("high") - pl.col("low")).median().alias("median_range"),
        (pl.col("high") - pl.col("low")).quantile(0.99).alias("p99_range"),
        pl.col("avg_spread").median().alias("median_spread"),
        pl.col("avg_spread").quantile(0.99).alias("p99_spread"),
        pl.col("tick_count").quantile(0.01).alias("p01_tick_count"),
    ).row(0, named=True)
    logger.info(
        "OHLCV quality: median_range=%.6f, p99_range=%.6f, "
        "median_spread=%.6f, p99_spread=%.6f, p01_tick_count=%.1f",
        stats["median_range"] or 0.0,
        stats["p99_range"] or 0.0,
        stats["median_spread"] or 0.0,
        stats["p99_spread"] or 0.0,
        stats["p01_tick_count"] or 0.0,
    )


def _compute_data_quality_stats(
    ohlcv: pl.DataFrame,
    group_ms: int,
    deduped_timestamps: int,
) -> dict:
    """Compute data-quality summary stats for OHLCV output."""
    total_bars = len(ohlcv)
    start_date = str(ohlcv["timestamp"].min())
    end_date = str(ohlcv["timestamp"].max())

    base = {
        "total_bars": total_bars,
        "deduped_timestamps": deduped_timestamps,
        "start_date": start_date,
        "end_date": end_date,
        "calendar_gaps": 0,
        "weekend_gaps": 0,
        "real_gaps": 0,
        "estimated_missing_bars": 0,
        "largest_gap_bars": 0,
    }

    if total_bars < 2:
        return base

    gap_summary = classify_calendar_gaps(ohlcv, group_ms)
    base.update(
        calendar_gaps=gap_summary.calendar_gap_count + gap_summary.real_gap_count,
        weekend_gaps=gap_summary.calendar_gap_count,
        real_gaps=gap_summary.real_gap_count,
        estimated_missing_bars=gap_summary.estimated_missing_bars,
        largest_gap_bars=gap_summary.largest_gap_bars,
        calendar_warnings=gap_summary.warnings,
    )
    return base


def _save_data_quality_json(stats: dict, config: Config) -> None:
    """Write data-quality stats to the JSON sidecar path."""
    dq_path = Path(config.paths.data_quality_json)
    dq_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dq_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(
        "Data quality JSON saved: %s (total_bars=%d)",
        dq_path,
        stats["total_bars"],
    )


def generate_data(config: Config) -> None:
    """Build OHLCV bars from raw monthly tick files and persist parquet + JSON stats."""
    raw_dir = Path(config.paths.data_raw)
    ohlcv_path = Path(config.paths.ohlcv)

    if not raw_dir.exists():
        if ohlcv_path.exists():
            logger.warning(
                "Raw data dir missing (%s) but OHLCV exists (%s) — skipping prepare.",
                raw_dir,
                ohlcv_path,
            )
            return
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        if ohlcv_path.exists():
            logger.warning(
                "No parquet files found (%s) but OHLCV exists (%s) — skipping prepare.",
                raw_dir,
                ohlcv_path,
            )
            return
        raise FileNotFoundError(f"No parquet files in {raw_dir}")

    logger.info("Found %d tick files in %s", len(parquet_files), raw_dir)

    group_ms = _timeframe_to_ms(config.data.timeframe)
    group_every = config.data.timeframe.lower()

    monthly_bars: list[pl.DataFrame] = []
    total_files = len(parquet_files)
    for idx, file_path in enumerate(parquet_files, start=1):
        logger.info(
            "Aggregating monthly file %d/%d: %s",
            idx,
            total_files,
            file_path.name,
        )
        monthly_bars.append(_aggregate_file(file_path, group_every))

    ohlcv = pl.concat(monthly_bars, how="vertical").sort("timestamp")

    ohlcv, _, deduped_count = _deduplicate_and_filter(ohlcv)
    ohlcv = _filter_date_range(ohlcv, config)
    _log_gap_report(ohlcv, group_ms)
    _log_candle_quality_report(ohlcv)

    dq_stats = _compute_data_quality_stats(ohlcv, group_ms, deduped_count)
    _save_data_quality_json(dq_stats, config)

    logger.info("OHLCV bars: %d (timeframe=%s)", len(ohlcv), config.data.timeframe)
    logger.info(
        "Date range: %s to %s",
        ohlcv["timestamp"].min(),
        ohlcv["timestamp"].max(),
    )

    ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(ohlcv_path)
    logger.info("Saved OHLCV: %s", ohlcv_path)


prepare_data = generate_data
