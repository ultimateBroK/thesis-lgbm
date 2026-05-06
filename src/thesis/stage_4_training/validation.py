"""Bar-based walk-forward sliding window validation with purge and embargo.

Generates rolling train/test window splits suitable for time-series
cross-validation. Windows are defined by row counts (bars), not fixed calendar
durations; calendar span can vary when weekends, holidays, or missing bars are
present. Each window applies *purge* and *embargo* gaps at the train/test
boundary to prevent information leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import polars as pl

logger = logging.getLogger("thesis.validation")


# Data model


@dataclass(frozen=True)
class WalkForwardWindow:
    """Index-based train / test slice for one walk-forward fold.

    Attributes:
        train_start_idx: Inclusive start index of the training period.
        train_end_idx: Exclusive end index of the training period
            (after purge has been applied).
        test_start_idx: Inclusive start index of the test period
            (after embargo has been applied).
        test_end_idx: Exclusive end index of the test period.
    """

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int


# Window generation


def generate_windows(
    total_bars: int,
    train_window_bars: int = 26_280,
    test_window_bars: int = 4_380,
    step_bars: int = 4_380,
    purge_bars: int = 25,
    embargo_bars: int = 50,
    min_train_bars: int = 5_000,
    event_end: np.ndarray | None = None,
) -> list[WalkForwardWindow]:
    """Create bar-count walk-forward windows across *total_bars* observations.

    Windows slide forward by *step_bars* each fold.  The parameters are counts
    of observed rows/bars, not guaranteed calendar durations.  Purge trims the
    tail of each training window and embargo skips the head of each test
    window so that no overlapping information leaks across the boundary.

    Args:
        total_bars: Total number of bars (rows) in the dataset.
        train_window_bars: Desired training window length in observed bars.
        test_window_bars: Desired test window length in observed bars.
        step_bars: Number of bars to advance between successive windows.
        purge_bars: Bars removed from the end of the training period.
        embargo_bars: Additional gap inserted after the purge zone.
        min_train_bars: Minimum training bars required to yield a window.
        event_end: Optional array of event-end bar indices used to apply
            label-aware purging instead of bar-count purging.

    Returns:
        Ordered list of :class:`WalkForwardWindow` objects.
    """
    windows: list[WalkForwardWindow] = []

    test_start = 0

    while test_start < total_bars:
        test_end = min(test_start + test_window_bars, total_bars)

        # Raw training region ends right before the test region
        raw_train_end = test_start
        train_start = max(0, raw_train_end - train_window_bars)

        if event_end is None:
            window = apply_purge_embargo(
                train_start=train_start,
                raw_train_end=raw_train_end,
                test_start=test_start,
                test_end=test_end,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        else:
            window = apply_event_time_purge(
                train_start=train_start,
                raw_train_end=raw_train_end,
                test_start=test_start,
                test_end=test_end,
                event_end=event_end,
                embargo_bars=embargo_bars,
            )

        if (
            window is not None
            and (window.train_end_idx - window.train_start_idx) >= min_train_bars
        ):
            windows.append(window)

        test_start += step_bars

    purge_mode = "event-time" if event_end is not None else "fixed-bar"
    logger.info("Generated %d %s walk-forward window(s)", len(windows), purge_mode)
    return windows


# Purge / embargo


def apply_purge_embargo(
    train_start: int,
    raw_train_end: int,
    test_start: int,
    test_end: int,
    purge_bars: int = 25,
    embargo_bars: int = 50,
) -> WalkForwardWindow | None:
    """Adjust a raw window to account for purge and embargo gaps.

    * **Purge** removes the last *purge_bars* from the training period.
    * **Embargo** skips the first *purge_bars + embargo_bars* from the
      test period, creating an additional information barrier.

    The total gap between the adjusted train end and adjusted test start
    is ``2 × purge_bars + embargo_bars``.  This is intentional: the
    extra *purge_bars* on the test side accounts for label lookahead
    (the label at the train boundary uses *horizon_bars* of future
    data, so both sides of the boundary need at least that many bars
    of clearance).

    Args:
        train_start: Raw training start index.
        raw_train_end: Raw training end index (exclusive).
        test_start: Raw test start index.
        test_end: Raw test end index (exclusive).
        purge_bars: Bars to trim from training tail.
        embargo_bars: Extra bars to skip after purge in the test head.

    Returns:
        A :class:`WalkForwardWindow` with adjusted indices, or ``None``
        if the resulting train or test period is empty.
    """
    adjusted_train_end = raw_train_end - purge_bars
    adjusted_test_start = test_start + purge_bars + embargo_bars

    if adjusted_train_end <= train_start:
        if raw_train_end == train_start:
            logger.debug(
                "Skipping pre-training window (start=%d, end=%d)",
                train_start,
                raw_train_end,
            )
            return None
        logger.warning(
            "Purge exhausted training period (start=%d, end=%d, purge=%d)",
            train_start,
            raw_train_end,
            purge_bars,
        )
        return None

    if adjusted_test_start >= test_end:
        logger.warning(
            "Purge+embargo exhausted test period (start=%d, end=%d, gap=%d)",
            test_start,
            test_end,
            purge_bars + embargo_bars,
        )
        return None

    return WalkForwardWindow(
        train_start_idx=train_start,
        train_end_idx=adjusted_train_end,
        test_start_idx=adjusted_test_start,
        test_end_idx=test_end,
    )


def apply_event_time_purge(
    train_start: int,
    raw_train_end: int,
    test_start: int,
    test_end: int,
    event_end: np.ndarray,
    embargo_bars: int = 50,
) -> WalkForwardWindow | None:
    """Adjust a window using label event-end times instead of fixed purge bars.

    Training samples are retained only when their triple-barrier event ends
    strictly before the raw test boundary. This prevents label lookahead from
    reaching into the test period while avoiding unnecessary fixed-bar trimming.
    Embargo still skips the first ``embargo_bars`` test rows.

    Args:
        train_start: Raw training start index.
        raw_train_end: Raw training end index (exclusive).
        test_start: Raw test start index.
        test_end: Raw test end index (exclusive).
        event_end: Array of event-end indices (one per bar).
        embargo_bars: Number of bars to skip at test head.

    Returns:
        A :class:`WalkForwardWindow` with adjusted indices, or ``None``
        if the resulting train or test period is empty.

    Raises:
        ValueError: If *event_end* is shorter than *raw_train_end*.
    """
    if raw_train_end <= train_start:
        logger.debug(
            "Skipping pre-training window (start=%d, end=%d)",
            train_start,
            raw_train_end,
        )
        return None

    if len(event_end) < raw_train_end:
        raise ValueError(
            f"event_end length ({len(event_end)}) is shorter than raw_train_end "
            f"({raw_train_end})"
        )

    train_event_end = event_end[train_start:raw_train_end]
    safe_offsets = np.flatnonzero(train_event_end < test_start)
    if safe_offsets.size == 0:
        logger.warning(
            "Event-time purge exhausted training period"
            " (start=%d, end=%d, test_start=%d)",
            train_start,
            raw_train_end,
            test_start,
        )
        return None

    adjusted_train_end = train_start + int(safe_offsets[-1]) + 1
    adjusted_test_start = test_start + embargo_bars

    if adjusted_test_start >= test_end:
        logger.warning(
            "Embargo exhausted test period (start=%d, end=%d, embargo=%d)",
            test_start,
            test_end,
            embargo_bars,
        )
        return None

    return WalkForwardWindow(
        train_start_idx=train_start,
        train_end_idx=adjusted_train_end,
        test_start_idx=adjusted_test_start,
        test_end_idx=test_end,
    )


# DataFrame splitting


def split_data(
    df: pl.DataFrame,
    windows: list[WalkForwardWindow],
    timestamp_col: str = "datetime",
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Slice a Polars DataFrame into (train, test) pairs per window.

    Uses integer-row slicing (not date-based filtering) so the indices
    in each :class:`WalkForwardWindow` map directly to row positions.
    The default Stage 4 pipeline slices inline for readability; this helper is
    retained for tests, notebooks, and external callers that need reusable
    train/test frame pairs.

    Args:
        df: Source DataFrame containing all bars.
        windows: Pre-computed walk-forward windows.
        timestamp_col: Name of the timestamp column (used only for
            logging; slicing is index-based).

    Returns:
        List of ``(train_df, test_df)`` tuples, one per window.
    """
    splits: list[tuple[pl.DataFrame, pl.DataFrame]] = []

    for i, w in enumerate(windows):
        train_df = df.slice(w.train_start_idx, w.train_end_idx - w.train_start_idx)
        test_df = df.slice(w.test_start_idx, w.test_end_idx - w.test_start_idx)
        splits.append((train_df, test_df))

        logger.debug(
            "Bar-based window %d — train rows [%d:%d] (%d), test rows [%d:%d] (%d)",
            i,
            w.train_start_idx,
            w.train_end_idx,
            len(train_df),
            w.test_start_idx,
            w.test_end_idx,
            len(test_df),
        )

    logger.info("Split DataFrame into %d bar-based (train, test) pair(s)", len(splits))
    return splits


# Logging / diagnostics


def log_windows(
    windows: list[WalkForwardWindow],
    df: pl.DataFrame,
    timestamp_col: str = "datetime",
) -> None:
    """Log human-readable date ranges for every walk-forward window.

    Useful for verifying that windows align with expected calendar
    boundaries and that purge / embargo gaps are reasonable.

    Args:
        windows: Walk-forward windows to log.
        df: Source DataFrame (must contain *timestamp_col*).
        timestamp_col: Name of the timestamp/Datetime column.
    """
    if timestamp_col not in df.columns:
        logger.warning(
            "Timestamp column %r not found — skipping window log", timestamp_col
        )
        return

    ts = df[timestamp_col]

    for i, w in enumerate(windows):
        train_start_dt = ts[w.train_start_idx]
        train_end_dt = ts[min(w.train_end_idx - 1, len(ts) - 1)]
        test_start_dt = ts[w.test_start_idx]
        test_end_dt = ts[min(w.test_end_idx - 1, len(ts) - 1)]

        train_bars = w.train_end_idx - w.train_start_idx
        test_bars = w.test_end_idx - w.test_start_idx

        logger.info(
            "Bar-based window %d | train: %s → %s (%d bars) | test: %s → %s (%d bars)",
            i,
            train_start_dt,
            train_end_dt,
            train_bars,
            test_start_dt,
            test_end_dt,
            test_bars,
        )
