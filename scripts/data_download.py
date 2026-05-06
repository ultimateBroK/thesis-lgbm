#!/usr/bin/env python3
"""Download tick data using dukascopy-python library.

Downloads historical tick data from Dukascopy for any supported instrument,
saving as monthly parquet files with verify & repair support.

Defaults are resolved from ``config.toml`` when available.

Usage:
    pixi run python scripts/data_download.py
    pixi run python scripts/data_download.py --workers 4
    pixi run python scripts/data_download.py --force
    pixi run python scripts/data_download.py --no-verify
    pixi run python scripts/data_download.py --instrument XAG/USD --asset-class fx
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Add src/ to path for thesis module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

from thesis.shared.config import load_config

logger = logging.getLogger(__name__)

# Constants for dukascopy-python
INTERVAL_TICK = dukascopy_python.INTERVAL_TICK
OFFER_SIDE_BID = dukascopy_python.OFFER_SIDE_BID


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

STATE_FILE = "download_state.json"


def load_state(state_path: Path) -> dict[str, dict[str, int]]:
    """Read the downloaded-month tracking state from disk."""
    if state_path.exists() and state_path.stat().st_size > 0:
        try:
            with state_path.open() as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("State file %s is corrupted, starting fresh", state_path)
    return {}


def save_state(state_path: Path, state: dict[str, dict[str, int]]) -> None:
    """Persist the month-tracking state dictionary as JSON."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Hour-coverage verification
# ---------------------------------------------------------------------------


def _trading_hour_slots(
    year: int, month: int, asset_class: str = "fx"
) -> list[tuple[int, int]]:
    """Return expected (day, hour) slots for a month.

    For FX: skip Saturday entirely; on Sunday only include hours >= 21 UTC.
    For crypto: all 24 hours every day.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    if asset_class == "crypto":
        return [(d, h) for d in range(1, days_in_month + 1) for h in range(24)]

    slots: list[tuple[int, int]] = []
    for d in range(1, days_in_month + 1):
        wd = calendar.weekday(year, month, d)
        if wd == 5:  # Saturday — no trading
            continue
        if wd == 6:  # Sunday — market opens at 21:00 UTC
            slots.extend((d, h) for h in range(21, 24))
        else:
            slots.extend((d, h) for h in range(24))
    return slots


def _check_hour_coverage(
    df: pl.DataFrame, year: int, month: int, asset_class: str = "fx"
) -> tuple[int, int]:
    """Check which expected trading-hour slots are missing from *df*.

    Returns:
        (missing_count, confirmed_absent) where *confirmed_absent* equals
        *missing_count* (we cannot HEAD-check the JSON3 API, so all gaps
        are treated as potentially absent on the server until repair proves
        otherwise).
    """
    expected = set(_trading_hour_slots(year, month, asset_class))
    if len(df) == 0:
        return len(expected), len(expected)

    covered = set(
        df.with_columns(
            [
                pl.col("timestamp").dt.day().alias("_d"),
                pl.col("timestamp").dt.hour().alias("_h"),
            ]
        )
        .select(["_d", "_h"])
        .unique()
        .rows()
    )
    missing = expected - covered
    return len(missing), len(missing)


def _repair_missing_hours(
    df: pl.DataFrame,
    year: int,
    month: int,
    asset_class: str,
    instrument: str,
    max_retries: int,
) -> tuple[pl.DataFrame, int, int]:
    """Attempt to re-fetch data for missing hour slots and merge into *df*.

    Returns:
        (merged_df, rows_added, still_missing)
    """
    expected = set(_trading_hour_slots(year, month, asset_class))
    if len(df) > 0:
        covered = set(
            df.with_columns(
                [
                    pl.col("timestamp").dt.day().alias("_d"),
                    pl.col("timestamp").dt.hour().alias("_h"),
                ]
            )
            .select(["_d", "_h"])
            .unique()
            .rows()
        )
    else:
        covered = set()

    missing_slots = sorted(expected - covered)
    if not missing_slots:
        return df, 0, 0

    logger.info("  Repairing %d missing hour slots...", len(missing_slots))

    patch_frames: list[pl.DataFrame] = []
    still_missing = 0

    for day, hour in missing_slots:
        start = datetime(year, month, day, hour, tzinfo=ZoneInfo("UTC"))
        # End is start of next hour (or next day if hour=23)
        if hour < 23:
            end = datetime(year, month, day, hour + 1, tzinfo=ZoneInfo("UTC"))
        else:
            try:
                end = datetime(year, month, day + 1, 0, tzinfo=ZoneInfo("UTC"))
            except ValueError:
                # Last day of month overflow — use first of next month
                if month == 12:
                    end = datetime(year + 1, 1, 1, 0, tzinfo=ZoneInfo("UTC"))
                else:
                    end = datetime(year, month + 1, 1, 0, tzinfo=ZoneInfo("UTC"))

        try:
            df_patch_pd = dukascopy_python.fetch(
                instrument,
                INTERVAL_TICK,
                OFFER_SIDE_BID,
                start,
                end,
                max_retries=max_retries,
                limit=30_000_000,
                debug=False,
            )
            if df_patch_pd is not None and len(df_patch_pd) > 0:
                df_patch = pl.from_pandas(df_patch_pd.reset_index())
                df_patch = df_patch.rename(
                    {
                        "timestamp": "timestamp",
                        "askPrice": "ask",
                        "bidPrice": "bid",
                        "askVolume": "ask_volume",
                        "bidVolume": "bid_volume",
                    }
                )
                df_patch = df_patch.select(
                    ["timestamp", "ask", "bid", "ask_volume", "bid_volume"]
                )
                df_patch = df_patch.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("ms")).alias("timestamp")
                )
                patch_frames.append(df_patch)
            else:
                still_missing += 1
        except Exception as e:
            logger.debug(
                "  Repair failed for %d-%02d %02d:00: %s", year, month, day, hour, e
            )
            still_missing += 1

    if patch_frames:
        all_frames = [df] + patch_frames
        merged = (
            pl.concat(all_frames, how="diagonal")
            .unique(subset=["timestamp"], keep="first")
            .sort("timestamp")
        )
        rows_added = len(merged) - len(df)
        return merged, rows_added, still_missing

    return df, 0, still_missing


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _pandas_to_polars(df_pandas) -> pl.DataFrame:
    """Convert a dukascopy-python pandas DataFrame to canonical Polars schema."""
    df = pl.from_pandas(df_pandas.reset_index())
    df = df.rename(
        {
            "timestamp": "timestamp",
            "askPrice": "ask",
            "bidPrice": "bid",
            "askVolume": "ask_volume",
            "bidVolume": "bid_volume",
        }
    )
    df = df.select(["timestamp", "ask", "bid", "ask_volume", "bid_volume"])
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")).alias("timestamp"))
    df = df.sort("timestamp")
    return df


def download_month(
    year: int,
    month: int,
    output_dir: Path,
    instrument: str = INSTRUMENT_FX_METALS_XAU_USD,
    asset_class: str = "fx",
    max_retries: int = 7,
    force: bool = False,
    verify: bool = True,
) -> tuple[int, int, int]:
    """Download tick data for a specific month with verify & repair.

    Args:
        year: Year to download.
        month: Month to download (1-12).
        output_dir: Directory to save parquet file.
        instrument: Dukascopy instrument identifier.
        asset_class: Asset class for trading-hour assumptions ("fx", "crypto").
        max_retries: Maximum retry attempts for failed downloads.
        force: If True, re-download even if file exists.
        verify: If True, check hour coverage and repair gaps.

    Returns:
        Tuple of (rows, missing_hours, confirmed_absent).
    """
    key = f"{year}-{month:02d}"
    file_path = output_dir / f"{key}.parquet"
    state_path = output_dir / STATE_FILE

    # --- Skip if already complete (unless force) ---
    if file_path.exists() and not force:
        try:
            df = pl.read_parquet(file_path)
            rows = len(df)
            if rows > 0:
                # Check state for known completeness
                state = load_state(state_path)
                entry = state.get(key)
                if entry and entry.get("missing_hours", -1) == 0:
                    logger.info("Skip     %s  rows=%10s  missing=0", key, f"{rows:,}")
                    return rows, 0, 0
                # File exists but state says incomplete or no state — verify
                if verify:
                    missing, confirmed = _check_hour_coverage(
                        df, year, month, asset_class
                    )
                    if missing == 0:
                        logger.info(
                            "Skip     %s  rows=%10s  verified complete",
                            key,
                            f"{rows:,}",
                        )
                        state[key] = {
                            "rows": rows,
                            "missing_hours": 0,
                            "confirmed_absent": 0,
                        }
                        save_state(state_path, state)
                        return rows, 0, 0
                    logger.info(
                        "Check    %s  rows=%10s  %d hours missing",
                        key,
                        f"{rows:,}",
                        missing,
                    )
                    # Fall through to repair
                else:
                    logger.info("Skip     %s  rows=%10s", key, f"{rows:,}")
                    return rows, 0, 0
        except Exception:
            logger.warning("Existing file %s is corrupted, re-downloading", file_path)

    # --- Calculate date range for the month ---
    start_dt = datetime(year, month, 1, tzinfo=ZoneInfo("UTC"))
    if month == 12:
        end_dt = datetime(year + 1, 1, 1, tzinfo=ZoneInfo("UTC"))
    else:
        end_dt = datetime(year, month + 1, 1, tzinfo=ZoneInfo("UTC"))

    # Don't download future dates
    now = datetime.now(timezone.utc)
    if start_dt > now:
        logger.info("Skip     %s  (future date)", key)
        return 0, 0, 0
    if end_dt > now:
        end_dt = now

    # --- Download ---
    logger.info("Download %s  (%s → %s)...", key, start_dt.date(), end_dt.date())

    df: pl.DataFrame | None = None

    # If file exists (repair path), load existing data first
    if file_path.exists():
        try:
            df = pl.read_parquet(file_path)
        except Exception:
            df = None

    if df is None:
        try:
            df_pandas = dukascopy_python.fetch(
                instrument,
                INTERVAL_TICK,
                OFFER_SIDE_BID,
                start_dt,
                end_dt,
                max_retries=max_retries,
                limit=30_000_000,
                debug=False,
            )

            if df_pandas is None or len(df_pandas) == 0:
                logger.warning("No data returned for %s", key)
                return 0, 0, 0

            df = _pandas_to_polars(df_pandas)
        except Exception as e:
            logger.error("Failed to download %s: %s", key, e)
            return 0, 0, 0

    # --- Verify & repair ---
    missing_hours = 0
    confirmed_absent = 0

    if verify and df is not None and len(df) > 0:
        missing, confirmed = _check_hour_coverage(df, year, month, asset_class)
        if missing > 0:
            df, rows_added, still_missing = _repair_missing_hours(
                df, year, month, asset_class, instrument, max_retries
            )
            if rows_added > 0:
                logger.info("  Repaired +%s rows for %s", f"{rows_added:,}", key)
            missing_hours = still_missing
            confirmed_absent = still_missing
        else:
            missing_hours = 0
            confirmed_absent = 0

    # --- Save ---
    if df is not None and len(df) > 0:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(file_path)
        rows = len(df)
    else:
        rows = 0

    flag = "complete" if missing_hours == 0 else f"{missing_hours} hrs missing"
    logger.info("   %s  rows=%10s  %s", key, f"{rows:,}", flag)

    # --- Update state ---
    state = load_state(state_path)
    is_past = end_dt < now
    if is_past:
        state[key] = {
            "rows": rows,
            "missing_hours": missing_hours,
            "confirmed_absent": confirmed_absent,
        }
        save_state(state_path, state)

    return rows, missing_hours, confirmed_absent


def _resolve_instrument(symbol: str) -> str:
    """Resolve a canonical symbol to a dukascopy-python instrument constant.

    Falls back to the raw symbol string if no matching constant is found.
    """
    from dukascopy_python import instruments as instr_module

    # Build lookup: e.g. INSTRUMENT_FX_METALS_XAU_USD -> "XAUUSD"
    for attr_name in dir(instr_module):
        if not attr_name.startswith("INSTRUMENT"):
            continue
        value = getattr(instr_module, attr_name)
        if (
            isinstance(value, str)
            and value.replace("/", "").upper() == symbol.replace("/", "").upper()
        ):
            return value
    # Fallback: try common FX metals naming convention
    if symbol.upper().startswith("XAU"):
        return INSTRUMENT_FX_METALS_XAU_USD
    # Return as-is — dukascopy-python may still accept it
    return symbol


def run_download(
    start_year: int | None = None,
    start_month: int | None = None,
    end_year: int | None = None,
    end_month: int | None = None,
    output_dir: Path | None = None,
    instrument: str | None = None,
    asset_class: str | None = None,
    workers: int = 4,
    max_retries: int | None = None,
    force: bool | None = None,
    verify: bool = True,
    skip_current_month: bool | None = None,
) -> bool:
    """Run the download job for specified date range.

    Any parameter left as ``None`` is resolved from ``config.toml``.
    Falls back to sensible defaults if the config file is unavailable.

    Args:
        start_year: Start year.
        start_month: Start month (1-12).
        end_year: End year.
        end_month: End month (1-12).
        output_dir: Directory for parquet files.
        instrument: Dukascopy instrument identifier.
        asset_class: Asset class ("fx", "crypto", "index").
        workers: Number of parallel workers.
        max_retries: Retry attempts per month.
        force: Force re-download existing files.
        verify: Check hour coverage and repair gaps.
        skip_current_month: Skip the current (incomplete) month.

    Returns:
        True if all months downloaded without errors, False otherwise.
    """
    # --- Resolve defaults from config.toml ---
    cfg = None
    try:
        cfg = load_config("config.toml")
    except FileNotFoundError:
        logger.warning("config.toml not found, using built-in defaults")

    if cfg is not None:
        data_cfg = cfg.data
        start_dt = datetime.strptime(data_cfg.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(data_cfg.end_date, "%Y-%m-%d")
        _defaults = {
            "start_year": start_dt.year,
            "start_month": start_dt.month,
            "end_year": end_dt.year,
            "end_month": end_dt.month,
            "output_dir": Path(cfg.paths.data_raw),
            "instrument": _resolve_instrument(
                data_cfg.symbol_download or data_cfg.symbol
            ),
            "asset_class": data_cfg.asset_class,
            "max_retries": data_cfg.download_max_retries,
            "force": data_cfg.download_force,
            "skip_current_month": data_cfg.download_skip_current_month,
        }
    else:
        _defaults = {
            "start_year": 2018,
            "start_month": 1,
            "end_year": datetime.now(timezone.utc).year,
            "end_month": datetime.now(timezone.utc).month,
            "output_dir": Path("data/raw/XAUUSD"),
            "instrument": INSTRUMENT_FX_METALS_XAU_USD,
            "asset_class": "fx",
            "max_retries": 7,
            "force": False,
            "skip_current_month": True,
        }

    start_year = start_year if start_year is not None else _defaults["start_year"]
    start_month = start_month if start_month is not None else _defaults["start_month"]
    end_year = end_year if end_year is not None else _defaults["end_year"]
    end_month = end_month if end_month is not None else _defaults["end_month"]
    output_dir = output_dir if output_dir is not None else _defaults["output_dir"]
    instrument = instrument if instrument is not None else _defaults["instrument"]
    asset_class = asset_class if asset_class is not None else _defaults["asset_class"]
    max_retries = max_retries if max_retries is not None else _defaults["max_retries"]
    force = force if force is not None else _defaults["force"]
    skip_current_month = (
        skip_current_month
        if skip_current_month is not None
        else _defaults["skip_current_month"]
    )

    # Derive output subdirectory from instrument if user didn't specify --output-dir
    if output_dir == _defaults["output_dir"]:
        symbol_dir = instrument.replace("/", "").upper()
        output_dir = Path("data/raw") / symbol_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of months to download
    months_to_download: list[tuple[int, int]] = []
    now = datetime.now(timezone.utc)
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        if skip_current_month and year == now.year and month == now.month:
            logger.info("Skip     %d-%02d  (current month)", year, month)
        else:
            months_to_download.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    total_rows = 0
    total_missing = 0
    total_errors = 0

    logger.info(
        "Starting download of %d months to %s  [instrument=%s, asset_class=%s, workers=%d]",
        len(months_to_download),
        output_dir,
        instrument,
        asset_class,
        workers,
    )

    if workers == 1:
        # Sequential download
        for year, month in months_to_download:
            rows, missing, confirmed = download_month(
                year,
                month,
                output_dir,
                instrument,
                asset_class,
                max_retries,
                force,
                verify,
            )
            total_rows += rows
            total_missing += missing
            if rows == 0 and missing > 0:
                total_errors += 1
    else:
        # Parallel download with process pool
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_month,
                    year,
                    month,
                    output_dir,
                    instrument,
                    asset_class,
                    max_retries,
                    force,
                    verify,
                ): (year, month)
                for year, month in months_to_download
            }

            for future in as_completed(futures):
                year, month = futures[future]
                try:
                    rows, missing, confirmed = future.result()
                    total_rows += rows
                    total_missing += missing
                    if rows == 0 and missing > 0:
                        total_errors += 1
                except Exception as e:
                    logger.error("Exception for %d-%02d: %s", year, month, e)
                    total_errors += 1

    logger.info(
        "Download complete: %s total rows, %d missing hours, %d errors",
        f"{total_rows:,}",
        total_missing,
        total_errors,
    )

    return total_errors == 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download tick data from Dukascopy via dukascopy-python",
    )

    # Load config for defaults
    cfg = None
    try:
        cfg = load_config("config.toml")
    except FileNotFoundError:
        pass

    if cfg is not None:
        data_cfg = cfg.data
        _start_dt = datetime.strptime(data_cfg.start_date, "%Y-%m-%d")
        _end_dt = datetime.strptime(data_cfg.end_date, "%Y-%m-%d")
        _def_start_year = _start_dt.year
        _def_start_month = _start_dt.month
        _def_end_year = _end_dt.year
        _def_end_month = _end_dt.month
        _def_output = str(cfg.paths.data_raw)
        _def_instrument = _resolve_instrument(
            data_cfg.symbol_download or data_cfg.symbol
        )
        _def_asset_class = data_cfg.asset_class
        _def_max_retries = data_cfg.download_max_retries
        _def_force = data_cfg.download_force
        _def_skip_current = data_cfg.download_skip_current_month
    else:
        _def_start_year = 2018
        _def_start_month = 1
        _def_end_year = datetime.now(timezone.utc).year
        _def_end_month = datetime.now(timezone.utc).month
        _def_output = "data/raw/XAUUSD"
        _def_instrument = INSTRUMENT_FX_METALS_XAU_USD
        _def_asset_class = "fx"
        _def_max_retries = 7
        _def_force = False
        _def_skip_current = True

    parser.add_argument(
        "--start-year",
        type=int,
        default=_def_start_year,
        help=f"Start year (default: {_def_start_year})",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=_def_start_month,
        help=f"Start month 1-12 (default: {_def_start_month})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=_def_end_year,
        help=f"End year (default: {_def_end_year})",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=_def_end_month,
        help=f"End month 1-12 (default: {_def_end_month})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {_def_output})",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help=f"Dukascopy instrument (default: {_def_instrument})",
    )
    parser.add_argument(
        "--asset-class",
        choices=["fx", "crypto", "index"],
        default=None,
        help=f"Asset class (default: {_def_asset_class})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=f"Max retries per request (default: {_def_max_retries})",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Force re-download (default: {_def_force})",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip hour-coverage verification and repair",
    )
    parser.add_argument(
        "--skip-current-month",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Skip current month (default: {_def_skip_current})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    success = run_download(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        output_dir=args.output_dir,
        instrument=args.instrument,
        asset_class=args.asset_class,
        workers=args.workers,
        max_retries=args.max_retries,
        force=args.force,
        verify=not args.no_verify,
        skip_current_month=args.skip_current_month,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
