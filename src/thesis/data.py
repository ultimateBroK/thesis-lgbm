"""Data preparation — aggregate raw tick data to OHLCV bars.

Reads monthly tick parquet files from data/raw/XAUUSD/, computes mid-price
OHLCV bars at the configured timeframe, and saves to data/processed/ohlcv.parquet.

Memory-efficient: aggregates each monthly file independently, then concats
only the small OHLCV results (~56K rows for 8 years of 1H bars).
"""

from __future__ import annotations

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

from thesis.config import Config

logger = logging.getLogger("thesis.prepare")


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

    # Compute mid-price and total volume
    ticks = ticks.with_columns(
        [
            (
                (
                    pl.col("ask") * pl.col("bid_volume")
                    + pl.col("bid") * pl.col("ask_volume")
                )
                / (pl.col("ask_volume") + pl.col("bid_volume") + 1e-10)
            ).alias("mid"),
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
                pl.col("mid").first().alias("open"),
                pl.col("mid").max().alias("high"),
                pl.col("mid").min().alias("low"),
                pl.col("mid").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("mid").count().alias("tick_count"),
                ((pl.col("ask") - pl.col("bid")).mean()).alias("avg_spread"),
            ]
        )
        .rename({"bar_time": "timestamp"})
    )

    return ohlcv


def _parse_timeframe_to_ms(timeframe: str) -> int:
    """Parse config timeframe string to milliseconds.

    Args:
        timeframe: Timeframe string like "1H", "4H", "5MIN", "1D".

    Returns:
        Timeframe in milliseconds.

    Raises:
        ValueError: If timeframe format is unsupported or invalid.
    """
    tf = timeframe.upper()
    if tf.endswith("H"):
        hours = int(tf[:-1])
        if hours <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': hours must be > 0")
        return hours * 3_600_000
    elif tf.endswith("MIN") or tf.endswith("M"):
        minutes = int(tf.replace("MIN", "").replace("M", ""))
        if minutes <= 0:
            raise ValueError(f"Invalid timeframe '{tf}': minutes must be > 0")
        return minutes * 60_000
    elif tf in ("D", "1D"):
        return 86_400_000
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")


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
        console=None,
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


def _deduplicate_and_filter(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Concat, deduplicate, and filter OHLCV bars.

    Args:
        ohlcv: Concatenated OHLCV DataFrame.

    Returns:
        Tuple of (filtered DataFrame, number of dropped bars).
    """
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
    return ohlcv, dropped


def prepare_data(config: Config) -> None:
    """Prepare OHLCV bars from raw tick parquet files.

    Reads monthly tick files from the configured raw directory, aggregates them
    into OHLCV bars at `config.data.timeframe`, removes duplicates, filters
    corrupted timestamps, and writes the result to `config.paths.ohlcv`.

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

    group_ms = _parse_timeframe_to_ms(config.data.timeframe)

    # Aggregate each monthly file separately — memory-efficient
    monthly_bars = _aggregate_monthly_files(parquet_files, group_ms)

    # Concat small OHLCV DataFrames (tiny compared to ticks)
    ohlcv = pl.concat(monthly_bars, how="vertical").sort("timestamp")

    # Remove duplicate bar timestamps and filter corrupted years
    ohlcv, _ = _deduplicate_and_filter(ohlcv)

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
