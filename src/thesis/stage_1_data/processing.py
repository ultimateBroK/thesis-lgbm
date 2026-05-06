"""Data preparation — aggregate raw tick data to OHLCV bars.

Reads monthly tick parquet files from data/raw/XAUUSD/, computes quote
microprice OHLCV bars at the configured timeframe, and saves to
data/processed/ohlcv.parquet.

Memory-efficient: aggregates each monthly file independently, then concats
only the small OHLCV results (~56K rows for 8 years of 1H bars).
"""

from __future__ import annotations

from datetime import timedelta
import json
import logging
from pathlib import Path

import polars as pl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from thesis.shared.config import Config
from thesis.shared.constants import timeframe_to_ms as _timeframe_to_ms
from thesis.shared.ui import console

logger = logging.getLogger("thesis.prepare")


def _parse_datetime_bound(value: str, name: str, dtype: pl.DataType) -> pl.Expr:
    """Parse an inclusive datetime bound from config into a Polars expression."""
    if not value:
        raise ValueError(f"config.data.{name} must not be empty")
    return pl.lit(value).str.to_datetime().cast(dtype)


def _aggregate_file(file_path: Path, group_ms: int) -> pl.DataFrame:
    """Aggregate one monthly tick parquet file into OHLCV bars.

    Args:
        file_path: Path to a monthly tick parquet file with `timestamp`, `bid`,
            `ask`, `ask_volume`, and `bid_volume` columns.
        group_ms: Bar size in milliseconds used to align ticks to bar boundaries.

    Returns:
        A Polars DataFrame with `timestamp`, `open`, `high`, `low`, `close`,
        `volume`, `tick_count`, and `avg_spread` columns.
    """
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

    # Compute quote microprice and total volume. This is volume-weighted by
    # opposing-side quote sizes, not the simple midpoint (bid + ask) / 2.
    ticks = ticks.with_columns(
        [
            (
                (
                    pl.col("ask") * pl.col("bid_volume")
                    + pl.col("bid") * pl.col("ask_volume")
                )
                / (pl.col("ask_volume") + pl.col("bid_volume") + 1e-10)
            ).alias("microprice"),
            (pl.col("ask_volume") + pl.col("bid_volume")).alias("volume"),
        ]
    )

    # Filter out corrupted timestamps (year must be 2000-2100)
    ticks = ticks.filter(
        (pl.col("timestamp").dt.year() >= 2000)
        & (pl.col("timestamp").dt.year() <= 2100)
    )

    # Floor timestamps to bar boundaries
    ts_ms = ticks["timestamp"].dt.timestamp("ms")
    bar_group = (ts_ms // group_ms) * group_ms

    ticks = ticks.with_columns(
        [
            bar_group.cast(pl.Datetime("ms")).alias("bar_time"),
        ]
    )

    # Sort ticks by bar_time then timestamp before aggregation
    # so first()/last() within each bar give deterministic open/close
    ticks = ticks.sort(["bar_time", "timestamp"])

    # Aggregate to OHLCV
    ohlcv = (
        ticks.group_by("bar_time", maintain_order=True)
        .agg(
            [
                pl.col("microprice").first().alias("open"),
                pl.col("microprice").max().alias("high"),
                pl.col("microprice").min().alias("low"),
                pl.col("microprice").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("microprice").count().alias("tick_count"),
                ((pl.col("ask") - pl.col("bid")).mean()).alias("avg_spread"),
            ]
        )
        .rename({"bar_time": "timestamp"})
    )

    return ohlcv


def _aggregate_monthly_files(
    parquet_files: list[Path],
    group_ms: int,
) -> list[pl.DataFrame]:
    """Aggregate monthly tick files into OHLCV bars.

    Args:
        parquet_files: List of monthly parquet file paths.
        group_ms: Bar size in milliseconds for grouping.

    Returns:
        List of OHLCV DataFrames, one per input file.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Aggregating {len(parquet_files)} monthly files",
            total=len(parquet_files),
        )
        monthly_bars: list[pl.DataFrame] = []
        for f in parquet_files:
            progress.update(task, description=f"[cyan]{f.name}")
            bars = _aggregate_file(f, group_ms)
            monthly_bars.append(bars)
            progress.advance(task)
        return monthly_bars


def _deduplicate_and_filter(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, int, int]:
    """Concat, deduplicate, and filter OHLCV bars.

    Args:
        ohlcv: Concatenated OHLCV DataFrame.

    Returns:
        Tuple of (filtered DataFrame, number of dropped bars,
        number of duplicate timestamps removed).
    """
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
    n_after = len(ohlcv)
    dropped = n_before - n_after
    if dropped > 0:
        logger.warning("Dropped %d bars with corrupted timestamps", dropped)
    return ohlcv, dropped, duplicate_count


def _filter_date_range(ohlcv: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Apply inclusive configured data date range to OHLCV bars.

    Args:
        ohlcv: OHLCV DataFrame with a ``timestamp`` column.
        config: Application configuration.

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If no bars remain after filtering.
    """
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
    """Log timestamp continuity diagnostics without filling missing bars.

    Args:
        ohlcv: OHLCV DataFrame with a ``timestamp`` column.
        group_ms: Expected bar interval in milliseconds.
    """
    if len(ohlcv) < 2:
        logger.warning("OHLCV gap report skipped: fewer than 2 bars")
        return

    diffs = (
        ohlcv.select(
            (pl.col("timestamp").diff().dt.total_milliseconds()).alias("delta_ms")
        )
        .drop_nulls()
        .get_column("delta_ms")
    )
    missing_gaps = diffs.filter(diffs > group_ms)
    duplicate_or_reversed = diffs.filter(diffs <= 0)
    missing_bars = int(((missing_gaps / group_ms).floor() - 1).sum() or 0)
    largest_gap_ms = int(diffs.max() or 0)

    logger.info(
        "OHLCV calendar gap report: expected_delta=%d ms, calendar_gap_count=%d, "
        "estimated_missing_bars=%d, largest_gap=%.2f bars, non_increasing_deltas=%d",
        group_ms,
        len(missing_gaps),
        missing_bars,
        largest_gap_ms / group_ms if group_ms else 0.0,
        len(duplicate_or_reversed),
    )


def _log_candle_quality_report(ohlcv: pl.DataFrame) -> None:
    """Log OHLCV candle integrity and likely outlier diagnostics.

    Args:
        ohlcv: OHLCV DataFrame.
    """
    if ohlcv.is_empty():
        return

    invalid = ohlcv.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("volume") < 0)
        | (pl.col("tick_count") <= 0)
        | (pl.col("avg_spread") < 0)
    )
    if len(invalid) > 0:
        logger.warning("OHLCV quality: %d invalid candles detected", len(invalid))

    stats = ohlcv.select(
        [
            (pl.col("high") - pl.col("low")).median().alias("median_range"),
            (pl.col("high") - pl.col("low")).quantile(0.99).alias("p99_range"),
            pl.col("avg_spread").median().alias("median_spread"),
            pl.col("avg_spread").quantile(0.99).alias("p99_spread"),
            pl.col("tick_count").quantile(0.01).alias("p01_tick_count"),
        ]
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


def _spans_weekend(start_dt: object, end_dt: object) -> bool:
    """Check if a time interval spans any weekend hours (Saturday or Sunday).

    Only returns ``True`` for gaps of at least 6 hours; shorter gaps are
    never weekend-related for financial intra-week data.

    Args:
        start_dt: Start datetime.
        end_dt: End datetime.

    Returns:
        ``True`` if any Saturday or Sunday falls within the interval.
    """
    delta = (end_dt - start_dt).total_seconds()
    if delta < 6 * 3600:  # shorter than 6 hours: cannot be a weekend gap
        return False

    current = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        days=1
    )
    while current < end_day:
        if current.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return True
        current += timedelta(days=1)
    return False


def _compute_data_quality_stats(
    ohlcv: pl.DataFrame,
    group_ms: int,
    deduped_timestamps: int,
) -> dict:
    """Compute data quality statistics and return as a dictionary.

    Analyses gap distribution (weekend vs real), duplicate removal counts,
    estimated missing bars, and data coverage dates.

    Args:
        ohlcv: OHLCV DataFrame after deduplication and date-range filtering.
        group_ms: Expected bar interval in milliseconds.
        deduped_timestamps: Number of duplicate timestamps removed during
            deduplication.

    Returns:
        Dictionary with keys ``total_bars``, ``deduped_timestamps``,
        ``start_date``, ``end_date``, ``calendar_gaps``, ``weekend_gaps``,
        ``real_gaps``, ``estimated_missing_bars``, ``largest_gap_bars``.
    """
    total_bars = len(ohlcv)
    start_date = str(ohlcv["timestamp"].min())
    end_date = str(ohlcv["timestamp"].max())

    if total_bars < 2:
        return {
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

    diffs = ohlcv.select(
        [
            pl.col("timestamp").diff().dt.total_milliseconds().alias("delta_ms"),
            pl.col("timestamp"),
        ]
    ).drop_nulls()

    timestamps = diffs["timestamp"].to_list()
    deltas = diffs["delta_ms"].to_list()

    weekend_gaps = 0
    real_gaps = 0
    estimated_missing_bars = 0
    largest_gap_bars = 0

    prev_ts_list = ohlcv["timestamp"].to_list()
    for i, (delta, ts) in enumerate(zip(deltas, timestamps)):
        if delta <= 0:
            continue
        if delta == group_ms:
            continue  # expected bar interval — no gap

        gap_bars = int(delta / group_ms)
        if gap_bars > largest_gap_bars:
            largest_gap_bars = gap_bars

        prev_ts = prev_ts_list[i]  # previous timestamp before this gap
        if _spans_weekend(prev_ts, ts):
            weekend_gaps += 1
        else:
            real_gaps += 1

        missing = max(0, gap_bars - 1)
        estimated_missing_bars += missing

    calendar_gaps = weekend_gaps + real_gaps

    return {
        "total_bars": total_bars,
        "deduped_timestamps": deduped_timestamps,
        "start_date": start_date,
        "end_date": end_date,
        "calendar_gaps": calendar_gaps,
        "weekend_gaps": weekend_gaps,
        "real_gaps": real_gaps,
        "estimated_missing_bars": estimated_missing_bars,
        "largest_gap_bars": largest_gap_bars,
    }


def _save_data_quality_json(stats: dict, config: Config) -> None:
    """Save data quality statistics as a JSON sidecar file.

    Args:
        stats: Data quality statistics dictionary.
        config: Application configuration for resolving output path.
    """
    dq_path = Path(config.paths.data_quality_json)
    dq_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dq_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(
        "Data quality JSON saved: %s (total_bars=%d)", dq_path, stats["total_bars"]
    )


def prepare_data(config: Config) -> None:
    """Prepare OHLCV bars from raw tick parquet files.

    Reads monthly tick files from the configured raw directory, aggregates them
    into OHLCV bars at ``config.data.timeframe``, removes duplicates, filters
    corrupted timestamps, and writes the result to ``config.paths.ohlcv``.

    Args:
        config: Application configuration.

    Raises:
        FileNotFoundError: If raw parquet files are unavailable and no cached
            OHLCV output exists.
        ValueError: If the configured timeframe is unsupported.
    """
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

    # Aggregate each monthly file separately — memory-efficient
    monthly_bars = _aggregate_monthly_files(parquet_files, group_ms)

    # Concat small OHLCV DataFrames (tiny compared to ticks)
    ohlcv = pl.concat(monthly_bars, how="vertical").sort("timestamp")

    # Remove duplicate bar timestamps and filter corrupted years
    ohlcv, _, deduped_count = _deduplicate_and_filter(ohlcv)
    ohlcv = _filter_date_range(ohlcv, config)
    _log_gap_report(ohlcv, group_ms)
    _log_candle_quality_report(ohlcv)

    # Compute and save data quality statistics
    dq_stats = _compute_data_quality_stats(ohlcv, group_ms, deduped_count)
    _save_data_quality_json(dq_stats, config)

    logger.info("OHLCV bars: %d (timeframe=%s)", len(ohlcv), config.data.timeframe)
    logger.info(
        "Date range: %s to %s",
        ohlcv["timestamp"].min(),
        ohlcv["timestamp"].max(),
    )

    # Save
    ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(ohlcv_path)
    logger.info("Saved OHLCV: %s", ohlcv_path)
