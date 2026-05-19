"""Raw tick data → validated OHLCV bars with quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import pandas_market_calendars as mcal
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import FEATURE_EPS
from thesis.shared.constants import timeframe_to_ms as _timeframe_to_ms

logger = logging.getLogger("thesis.prepare")


def validate_ohlcv(df: pl.DataFrame) -> dict[str, Any]:
    """OHLCV integrity: OHLC relationships, non-negative prices/volumes."""
    if df.is_empty():
        return dict(
            total_rows=0,
            invalid_count=0,
            ohlc_violations=0,
            price_negative_count=0,
            volume_negative_count=0,
            is_valid=True,
        )

    total = len(df)
    conditions: list[pl.Expr] = []
    price_negative = 0
    volume_negative = 0

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            price_negative += int((df[col] <= 0).sum())

    if "volume" in df.columns:
        volume_negative = int((df["volume"] < 0).sum())
        conditions.append(pl.col("volume") >= 0)

    if all(c in df.columns for c in ("open", "high", "low", "close")):
        conditions.extend(
            [
                pl.col("high") >= pl.col("low"),
                pl.col("high") >= pl.col("open"),
                pl.col("high") >= pl.col("close"),
                pl.col("low") <= pl.col("open"),
                pl.col("low") <= pl.col("close"),
            ]
        )

    invalid_count = (
        total - len(df.filter(pl.all_horizontal(conditions))) if conditions else 0
    )

    return dict(
        total_rows=total,
        invalid_count=invalid_count,
        ohlc_violations=max(invalid_count - volume_negative, 0),
        price_negative_count=price_negative,
        volume_negative_count=volume_negative,
        is_valid=invalid_count == 0 and price_negative == 0,
    )


def check_gap_report(df: pl.DataFrame, timeframe_ms: int) -> dict[str, Any]:
    """Timestamp continuity: gaps > timeframe_ms and duplicate count."""
    if "timestamp" not in df.columns or len(df) < 2:
        return dict(
            gap_count=0,
            estimated_missing_bars=0,
            largest_gap_bars=0,
            duplicate_count=0,
        )

    diffs = (
        df.select(
            (pl.col("timestamp").diff().dt.total_milliseconds()).alias("delta_ms")
        )
        .drop_nulls()
        .get_column("delta_ms")
    )

    missing_gaps = diffs.filter(diffs > timeframe_ms)
    estimated_missing = int(((missing_gaps / timeframe_ms).floor() - 1).sum() or 0)
    largest_gap_bars = int(diffs.max() / timeframe_ms) if diffs.max() else 0
    duplicate_count = len(df) - df["timestamp"].n_unique()

    return dict(
        gap_count=len(missing_gaps),
        estimated_missing_bars=estimated_missing,
        largest_gap_bars=largest_gap_bars,
        duplicate_count=int(duplicate_count),
    )


DEFAULT_GOLD_CALENDARS: tuple[str, ...] = (
    "CME Globex Gold and Silver Futures",
    "CME Globex Commodities",
    "CME_FX",
)


@dataclass(frozen=True)
class GapClassification:
    """Timestamp gap classification summary."""

    calendar_gap_count: int
    real_gap_count: int
    estimated_missing_bars: int
    largest_gap_bars: int
    warnings: list[str]


def _resolve_market_calendar(name: str | None = None):
    """Try CME gold calendars in priority order — first success wins."""
    candidates = [name] if name else list(DEFAULT_GOLD_CALENDARS)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return mcal.get_calendar(candidate)
        except Exception:
            continue
    raise RuntimeError(
        "Could not resolve market calendar. "
        f"Tried: {candidates or list(DEFAULT_GOLD_CALENDARS)}"
    )


def _classify_gaps_with_calendar(
    df: pl.DataFrame, timeframe_ms: int, calendar_name: str | None
) -> GapClassification:
    """Separate expected closures from real missing bars.

    Uses CME trading schedule.
    """
    ts = df["timestamp"].sort().to_list()
    actual_index = pd.DatetimeIndex(ts)
    if actual_index.tz is None:
        actual_index = actual_index.tz_localize("UTC")
    else:
        actual_index = actual_index.tz_convert("UTC")

    cal = _resolve_market_calendar(calendar_name)
    schedule = cal.schedule(
        start_date=actual_index.min().date(),
        end_date=actual_index.max().date(),
    )

    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        expected_index = mcal.date_range(schedule, frequency=f"{timeframe_ms}ms")
        for warn in warns:
            captured_warnings.append(str(warn.message))

    expected_set = set(expected_index)
    calendar_gap_count = 0
    real_gap_count = 0
    estimated_missing_bars = 0
    largest_gap_bars = 0

    for prev_ts, curr_ts in zip(actual_index[:-1], actual_index[1:]):
        delta_ms = int((curr_ts - prev_ts).total_seconds() * 1000)
        if delta_ms <= timeframe_ms:
            continue
        gap_bars = delta_ms // timeframe_ms
        largest_gap_bars = max(largest_gap_bars, gap_bars)
        interior = pd.date_range(
            start=prev_ts + pd.Timedelta(milliseconds=timeframe_ms),
            end=curr_ts - pd.Timedelta(milliseconds=timeframe_ms),
            freq=pd.Timedelta(milliseconds=timeframe_ms),
            tz="UTC",
        )
        expected_missing = sum(1 for t in interior if t in expected_set)
        if expected_missing > 0:
            real_gap_count += 1
            estimated_missing_bars += expected_missing
        else:
            calendar_gap_count += 1

    return GapClassification(
        calendar_gap_count=calendar_gap_count,
        real_gap_count=real_gap_count,
        estimated_missing_bars=estimated_missing_bars,
        largest_gap_bars=largest_gap_bars,
        warnings=captured_warnings,
    )


def _classify_gaps_with_heuristic(
    df: pl.DataFrame, timeframe_ms: int, reason: str
) -> GapClassification:
    """Weekend heuristic — fallback when market calendar library unavailable."""
    ts_col = df["timestamp"].sort()
    deltas = ts_col.diff().drop_nulls().dt.total_milliseconds().to_list()
    ts_values = ts_col.to_list()

    calendar_gap_count = 0
    real_gap_count = 0
    missing = 0
    largest = 0

    for i, delta in enumerate(deltas):
        if delta <= timeframe_ms:
            continue
        gap_bars = int(delta // timeframe_ms)
        largest = max(largest, gap_bars)
        if ts_values[i].weekday() >= 5 or ts_values[i + 1].weekday() >= 5:
            calendar_gap_count += 1
        else:
            real_gap_count += 1
            missing += max(0, gap_bars - 1)

    return GapClassification(
        calendar_gap_count=calendar_gap_count,
        real_gap_count=real_gap_count,
        estimated_missing_bars=missing,
        largest_gap_bars=largest,
        warnings=[f"calendar_fallback:{reason}"],
    )


def classify_calendar_gaps(
    df: pl.DataFrame, timeframe_ms: int, *, calendar_name: str | None = None
) -> GapClassification:
    """Calendar-expected closures vs real missing data bars."""
    if "timestamp" not in df.columns or len(df) < 2:
        return GapClassification(0, 0, 0, 0, [])
    try:
        return _classify_gaps_with_calendar(df, timeframe_ms, calendar_name)
    except Exception as exc:
        return _classify_gaps_with_heuristic(
            df, timeframe_ms, f"{type(exc).__name__}:{exc}"
        )


def _discover_files(raw_dir: Path, ohlcv_path: Path) -> list[Path]:
    if not raw_dir.exists():
        if ohlcv_path.exists():
            logger.warning("Raw dir missing but OHLCV cached — skip.")
            return []
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        if ohlcv_path.exists():
            logger.warning("No parquet files but OHLCV cached — skip.")
            return []
        raise FileNotFoundError(f"No parquet files in {raw_dir}")
    return files


def _parse_dt(value: str, tz: str) -> pl.Expr:
    """Config datetime string → tz-naive UTC Polars expr matching OHLCV dtype."""
    if not value:
        raise ValueError("datetime bound must not be empty")
    value = value.strip()
    if "T" not in value and " " not in value and ":" not in value:
        value = value + "T23:59:59"
    return (
        pl.lit(value)
        .str.to_datetime(time_unit="ms", time_zone=tz)
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
    )


def _microprice(ticks: pl.DataFrame) -> pl.DataFrame:
    """Volume-weighted mid: bid/ask weighted by opposing side depth."""
    return ticks.with_columns(
        (
            (
                pl.col("ask") * pl.col("bid_volume")
                + pl.col("bid") * pl.col("ask_volume")
            )
            / (pl.col("ask_volume") + pl.col("bid_volume") + FEATURE_EPS)
        ).alias("microprice"),
        (pl.col("ask_volume") + pl.col("bid_volume")).alias("volume"),
    )


def _clip_to_month(df: pl.DataFrame, stem: str) -> pl.DataFrame:
    """Boundary ticks from adjacent files bleed in — clip to nominal month."""
    year, month = int(stem[:4]), int(stem[5:7])
    start = pl.datetime(year, month, 1)
    end = (
        pl.datetime(year + 1, 1, 1) - pl.duration(seconds=1)
        if month == 12
        else pl.datetime(year, month + 1, 1) - pl.duration(seconds=1)
    )
    return df.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))


def _aggregate_file(file_path: Path, group_every: str) -> pl.DataFrame:
    """Single monthly tick file → OHLCV bars with volume imbalance and spread."""
    ticks = pl.read_parquet(
        file_path,
        columns=["timestamp", "bid", "ask", "ask_volume", "bid_volume"],
    )
    n_raw = len(ticks)
    ticks = ticks.filter(
        (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
        & (pl.col("ask") >= pl.col("bid"))
        & (pl.col("ask_volume") >= 0)
        & (pl.col("bid_volume") >= 0)
    )
    if (dropped := n_raw - len(ticks)) > 0:
        logger.warning("%s: dropped %d invalid quotes", file_path.name, dropped)

    ticks = _microprice(ticks).sort("timestamp")

    bars = (
        ticks.group_by_dynamic(
            "timestamp",
            every=group_every,
            period=group_every,
            closed="left",
            label="left",
            start_by="window",
        )
        .agg(
            pl.col("microprice").first().alias("open"),
            pl.col("microprice").max().alias("high"),
            pl.col("microprice").min().alias("low"),
            pl.col("microprice").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("bid_volume").sum().alias("bid_volume"),
            pl.col("ask_volume").sum().alias("ask_volume"),
            pl.col("microprice").count().alias("tick_count"),
            (pl.col("ask") - pl.col("bid")).mean().alias("avg_spread"),
        )
        .with_columns(
            [
                (
                    (pl.col("bid_volume") - pl.col("ask_volume"))
                    / (pl.col("bid_volume") + pl.col("ask_volume") + FEATURE_EPS)
                ).alias("volume_imbalance"),
                (pl.col("avg_spread") / (pl.col("close") + FEATURE_EPS)).alias(
                    "spread_pct_close"
                ),
            ]
        )
        .drop_nulls()
    )
    return _clip_to_month(bars, file_path.stem)


def _aggregate_all(files: list[Path], group_every: str) -> pl.DataFrame:
    bars = []
    for i, f in enumerate(files, 1):
        logger.info("Aggregating %d/%d: %s", i, len(files), f.name)
        bars.append(_aggregate_file(f, group_every))
    return pl.concat(bars, how="vertical").sort("timestamp")


def _dedupe_and_filter(ohlcv: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Remove duplicate timestamps and corrupted bars (year < 2000 or > 2100)."""
    n = len(ohlcv)
    dupes = n - ohlcv.get_column("timestamp").n_unique()
    if dupes > 0:
        logger.warning("Found %d duplicate timestamps — keeping first", dupes)
    ohlcv = ohlcv.unique(subset=["timestamp"], keep="first").sort("timestamp")
    before = len(ohlcv)
    ohlcv = ohlcv.filter(
        (pl.col("timestamp").dt.year() >= 2000)
        & (pl.col("timestamp").dt.year() <= 2100)
    )
    if (dropped := before - len(ohlcv)) > 0:
        logger.warning("Dropped %d bars with corrupted timestamps", dropped)
    return ohlcv, dupes


def _filter_range(ohlcv: pl.DataFrame, config: Config) -> pl.DataFrame:
    tz = config.data.market_tz
    start = _parse_dt(config.data_range.start, tz)
    end = _parse_dt(config.data_range.end, tz)
    before = len(ohlcv)
    ohlcv = ohlcv.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    if (dropped := before - len(ohlcv)) > 0:
        logger.info(
            "Dropped %d bars outside range [%s, %s]",
            dropped,
            config.data_range.start,
            config.data_range.end,
        )
    if ohlcv.is_empty():
        raise ValueError(
            f"No OHLCV bars after date filter "
            f"[{config.data_range.start}, {config.data_range.end}]"
        )
    return ohlcv


def _log_gap(ohlcv: pl.DataFrame, group_ms: int) -> None:
    if len(ohlcv) < 2:
        logger.warning("Gap report skipped: < 2 bars")
        return
    result = check_gap_report(ohlcv, group_ms)
    diffs = (
        ohlcv.select((pl.col("timestamp").diff().dt.total_milliseconds()).alias("d"))
        .drop_nulls()
        .get_column("d")
    )
    non_inc = int((diffs <= 0).sum())
    logger.info(
        "Gap report: expected=%dms gaps=%d missing=%d largest=%.2f bars non_inc=%d",
        group_ms,
        result["gap_count"],
        result["estimated_missing_bars"],
        result["largest_gap_bars"],
        non_inc,
    )


def _log_quality(ohlcv: pl.DataFrame) -> None:
    if ohlcv.is_empty():
        return
    v = validate_ohlcv(ohlcv)
    if v["invalid_count"] > 0:
        logger.warning("%d invalid candles", v["invalid_count"])
    s = ohlcv.select(
        (pl.col("high") - pl.col("low")).median().alias("med_range"),
        (pl.col("high") - pl.col("low")).quantile(0.99).alias("p99_range"),
        pl.col("avg_spread").median().alias("med_spread"),
        pl.col("avg_spread").quantile(0.99).alias("p99_spread"),
        pl.col("tick_count").quantile(0.01).alias("p01_ticks"),
    ).row(0, named=True)
    logger.info(
        "Quality: med_range=%.6f p99_range=%.6f "
        "med_spread=%.6f p99_spread=%.6f p01_ticks=%.1f",
        s["med_range"] or 0,
        s["p99_range"] or 0,
        s["med_spread"] or 0,
        s["p99_spread"] or 0,
        s["p01_ticks"] or 0,
    )


def _save_json(ohlcv: pl.DataFrame, config: Config, group_ms: int, dupes: int) -> None:
    total = len(ohlcv)
    stats: dict[str, Any] = {
        "total_bars": total,
        "deduped_timestamps": dupes,
        "start_date": str(ohlcv["timestamp"].min()),
        "end_date": str(ohlcv["timestamp"].max()),
        "calendar_gaps": 0,
        "weekend_gaps": 0,
        "real_gaps": 0,
        "estimated_missing_bars": 0,
        "largest_gap_bars": 0,
    }
    if total >= 2:
        g = classify_calendar_gaps(ohlcv, group_ms)
        stats.update(
            calendar_gaps=g.calendar_gap_count + g.real_gap_count,
            weekend_gaps=g.calendar_gap_count,
            real_gaps=g.real_gap_count,
            estimated_missing_bars=g.estimated_missing_bars,
            largest_gap_bars=g.largest_gap_bars,
            calendar_warnings=g.warnings,
        )
    path = Path(config.paths.data_processed) / "data_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Saved quality JSON: %s (bars=%d)", path, total)


def _persist(ohlcv: pl.DataFrame, config: Config, dupes: int) -> None:
    group_ms = _timeframe_to_ms(config.data.timeframe)
    _log_gap(ohlcv, group_ms)
    _log_quality(ohlcv)
    _save_json(ohlcv, config, group_ms, dupes)
    logger.info(
        "OHLCV: %d bars | %s to %s",
        len(ohlcv),
        ohlcv["timestamp"].min(),
        ohlcv["timestamp"].max(),
    )
    out = Path(config.paths.ohlcv)
    out.parent.mkdir(parents=True, exist_ok=True)
    ohlcv.write_parquet(out)
    logger.info("Saved: %s", out)


def prepare_dataset(config: Config) -> None:
    """Raw ticks → validated OHLCV parquet + quality JSON."""
    raw_dir = Path(config.paths.data_raw)
    ohlcv_path = Path(config.paths.ohlcv)

    if config.workflow.force_rerun and ohlcv_path.exists():
        ohlcv_path.unlink()
        logger.info("force_rerun — removed old OHLCV")

    files = _discover_files(raw_dir, ohlcv_path)
    if not files:
        return
    logger.info("Found %d tick files", len(files))

    ohlcv = _aggregate_all(files, config.data.timeframe.lower())
    ohlcv, dupes = _dedupe_and_filter(ohlcv)
    ohlcv = _filter_range(ohlcv, config)
    _persist(ohlcv, config, dupes)
