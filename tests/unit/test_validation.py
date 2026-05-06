"""Tests for validation module — walk-forward sliding window."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.stage_4_training.validation import (
    WalkForwardWindow,
    apply_event_time_purge,
    apply_purge_embargo,
    generate_windows,
    log_windows,
    split_data,
)


def _make_df(n: int = 500) -> pl.DataFrame:
    """Create a DataFrame with n hourly rows starting 2020-01-01."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
                interval="1h",
                eager=True,
            ),
            "value": list(range(n)),
        }
    )


class TestWalkForwardWindow:
    def test_window_creation(self):
        w = WalkForwardWindow(
            train_start_idx=0,
            train_end_idx=100,
            test_start_idx=100,
            test_end_idx=120,
        )
        assert w.train_start_idx == 0
        assert w.train_end_idx == 100
        assert w.test_start_idx == 100
        assert w.test_end_idx == 120

    def test_window_is_frozen(self):
        w = WalkForwardWindow(0, 100, 100, 120)
        with pytest.raises(AttributeError):
            w.train_start_idx = 50


class TestApplyPurgeEmbargo:
    def test_basic_purge(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=1000,
            test_start=1000,
            test_end=1200,
            purge_bars=25,
            embargo_bars=50,
        )
        assert result is not None
        assert result.train_end_idx == 975  # 1000 - 25
        assert result.test_start_idx == 1075  # 1000 + 25 + 50

    def test_no_purge(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=1000,
            test_start=1000,
            test_end=1200,
            purge_bars=0,
            embargo_bars=0,
        )
        assert result is not None
        assert result.train_end_idx == 1000
        assert result.test_start_idx == 1000

    def test_returns_none_if_train_too_small(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=10,
            test_start=10,
            test_end=100,
            purge_bars=25,
            embargo_bars=50,
        )
        assert result is None


class TestApplyEventTimePurge:
    def test_removes_train_events_reaching_test_start(self):
        event_end = np.arange(200)
        event_end[90:100] = 105
        result = apply_event_time_purge(
            train_start=0,
            raw_train_end=100,
            test_start=100,
            test_end=150,
            event_end=event_end,
            embargo_bars=10,
        )
        assert result is not None
        assert result.train_end_idx == 90
        assert result.test_start_idx == 110

    def test_returns_none_when_all_train_events_overlap(self):
        event_end = np.full(100, 100)
        result = apply_event_time_purge(
            train_start=0,
            raw_train_end=50,
            test_start=50,
            test_end=80,
            event_end=event_end,
            embargo_bars=5,
        )
        assert result is None

    def test_generate_windows_accepts_event_end(self):
        event_end = np.arange(10000)
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
            event_end=event_end,
        )
        assert len(windows) >= 1
        assert windows[0].test_start_idx % 1000 == 5

    def test_returns_none_if_test_too_small(self):
        result = apply_purge_embargo(
            train_start=0,
            raw_train_end=1000,
            test_start=1000,
            test_end=1020,
            purge_bars=25,
            embargo_bars=50,
        )
        assert result is None


class TestGenerateWindows:
    def test_generates_windows(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        assert len(windows) >= 1

    def test_window_train_starts_at_zero(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        assert windows[0].train_start_idx == 0

    def test_windows_step_forward(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        if len(windows) >= 2:
            assert windows[1].test_start_idx > windows[0].test_start_idx

    def test_no_windows_if_too_small(self):
        windows = generate_windows(
            total_bars=100,
            train_window_bars=5000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=1000,
        )
        assert len(windows) == 0

    def test_purge_embargo_adjusts_indices(self):
        windows = generate_windows(
            total_bars=10000,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=25,
            embargo_bars=50,
            min_train_bars=500,
        )
        assert len(windows) >= 1
        w = windows[0]
        # train_end should be less than the raw boundary
        assert w.train_end_idx < w.test_start_idx
        # test_start should be adjusted by purge+embargo
        assert w.test_start_idx - w.train_end_idx >= 25 + 50

    def test_windows_dont_exceed_total(self):
        windows = generate_windows(
            total_bars=5500,
            train_window_bars=3000,
            test_window_bars=1000,
            step_bars=1000,
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=500,
        )
        for w in windows:
            assert w.test_end_idx <= 5500


class TestSplitData:
    def test_basic_split(self):
        df = _make_df(500)
        windows = [WalkForwardWindow(0, 300, 300, 400)]
        splits = split_data(df, windows, "timestamp")
        assert len(splits) == 1
        train_df, test_df = splits[0]
        assert len(train_df) == 300
        assert len(test_df) == 100

    def test_multiple_splits(self):
        df = _make_df(1000)
        windows = [
            WalkForwardWindow(0, 400, 400, 500),
            WalkForwardWindow(100, 500, 500, 600),
        ]
        splits = split_data(df, windows, "timestamp")
        assert len(splits) == 2

    def test_empty_windows(self):
        df = _make_df(500)
        splits = split_data(df, [], "timestamp")
        assert len(splits) == 0


class TestConsecutiveWindowsNoOverlap:
    """Verify walk-forward test windows are disjoint across folds."""

    def test_consecutive_test_windows_do_not_overlap(self):
        """No index can appear in two different test windows."""
        windows = generate_windows(
            total_bars=20000,
            train_window_bars=8000,
            test_window_bars=2000,
            step_bars=2000,  # step == test_window → non-overlapping
            purge_bars=10,
            embargo_bars=5,
            min_train_bars=2000,
        )
        assert len(windows) >= 2, "Need at least 2 windows to test overlap"

        for i in range(len(windows) - 1):
            w_a = windows[i]
            w_b = windows[i + 1]
            # test range of window i must end before test range of window i+1
            assert w_a.test_end_idx <= w_b.test_start_idx, (
                f"Window {i} test [{w_a.test_start_idx}, {w_a.test_end_idx}) "
                f"overlaps window {i + 1} test [{w_b.test_start_idx}, {w_b.test_end_idx})"
            )

    def test_all_test_indices_are_disjoint_sets(self):
        """Collect all test indices and verify zero duplicates."""
        windows = generate_windows(
            total_bars=15000,
            train_window_bars=5000,
            test_window_bars=1500,
            step_bars=1500,
            purge_bars=5,
            embargo_bars=3,
            min_train_bars=1000,
        )
        all_test_indices: set[int] = set()
        for w in windows:
            for idx in range(w.test_start_idx, w.test_end_idx):
                assert idx not in all_test_indices, (
                    f"Index {idx} appears in multiple test windows"
                )
                all_test_indices.add(idx)


class TestOOFUniquenessGuard:
    """Verify duplicate-timestamp detection logic for OOF predictions."""

    def test_oof_timestamps_unique_after_concat(self):
        """OOF predictions must have unique timestamps — no double-counting."""
        # Simulate two fold predictions with one overlapping timestamp
        fold1 = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "pred_label": [1, -1, 0],
            }
        )
        fold2 = pl.DataFrame(
            {
                "timestamp": [3, 4, 5],  # timestamp 3 overlaps
                "pred_label": [1, -1, 0],
            }
        )
        oof_df = pl.concat([fold1, fold2])

        ts_col = oof_df["timestamp"]
        assert ts_col.n_unique() < len(ts_col), "Expected duplicates"
        dup_count = len(ts_col) - ts_col.n_unique()
        assert dup_count == 1

    def test_oof_no_duplicates_passes(self):
        """Non-overlapping folds produce unique timestamps."""
        fold1 = pl.DataFrame(
            {
                "timestamp": [1, 2, 3],
                "pred_label": [1, -1, 0],
            }
        )
        fold2 = pl.DataFrame(
            {
                "timestamp": [4, 5, 6],
                "pred_label": [1, -1, 0],
            }
        )
        oof_df = pl.concat([fold1, fold2])

        ts_col = oof_df["timestamp"]
        assert ts_col.n_unique() == len(ts_col)


class TestLogWindows:
    def test_log_runs_without_error(self):
        df = _make_df(500)
        windows = [WalkForwardWindow(0, 300, 300, 400)]
        log_windows(windows, df, "timestamp")

    def test_log_missing_column(self):
        df = _make_df(500)
        windows = [WalkForwardWindow(0, 300, 300, 400)]
        # Should log warning but not crash
        log_windows(windows, df, "nonexistent")


# ---------------------------------------------------------------------------
# Current-config window overlap integrity test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWindowTestPeriodsNoOverlap:
    """Verify consecutive walk-forward test windows are non-overlapping.

    Uses ``config.toml`` validation-section parameters as defaults to match
    the production walk-forward setup.  A non-overlapping guarantee depends on
    ``step_bars >= test_window_bars``; this test will **fail** if
    ``step_bars < test_window_bars``, acting as a guardrail against config
    changes that would overstate OOF validity.
    """

    # ── current config defaults (config.toml section [validation]) ──────
    TRAIN_WINDOW_BARS = 17520  # ~2 years of H1 bars
    TEST_WINDOW_BARS = 4380  # ~6 months of H1 bars
    STEP_BARS = 4380  # must be >= test_window_bars to prevent overlap
    PURGE_BARS = 48
    EMBARGO_BARS = 50
    MIN_TRAIN_BARS = 10000

    def test_window_test_periods_no_overlap(self):
        """No bar index may appear in more than one test window.

        Generates walk-forward windows using the current ``config.toml``
        validation parameters and asserts that for every consecutive pair
        ``test_end_idx[i] <= test_start_idx[i+1]``.
        """
        total_bars = 100_000  # generous — well above 2018-01→2026-04 1H range

        windows = generate_windows(
            total_bars=total_bars,
            train_window_bars=self.TRAIN_WINDOW_BARS,
            test_window_bars=self.TEST_WINDOW_BARS,
            step_bars=self.STEP_BARS,
            purge_bars=self.PURGE_BARS,
            embargo_bars=self.EMBARGO_BARS,
            min_train_bars=self.MIN_TRAIN_BARS,
        )

        assert len(windows) >= 2, (
            f"Need at least 2 windows to check overlap; got {len(windows)}"
        )

        overlapping_pairs: list[tuple[int, int, int, int, int]] = []
        for i in range(len(windows) - 1):
            prev = windows[i]
            nxt = windows[i + 1]
            if prev.test_end_idx > nxt.test_start_idx:
                overlapping_pairs.append(
                    (
                        i,
                        prev.test_start_idx,
                        prev.test_end_idx,
                        nxt.test_start_idx,
                        nxt.test_end_idx,
                    )
                )

        if overlapping_pairs:
            import logging

            _log = logging.getLogger(__name__)
            for i, ps, pe, ns, ne in overlapping_pairs:
                _log.warning(
                    "OVERLAP DETECTED: window %d test [%d, %d) overlaps "
                    "window %d test [%d, %d) by %d bars",
                    i,
                    ps,
                    pe,
                    i + 1,
                    ns,
                    ne,
                    pe - ns,
                )

        assert not overlapping_pairs, (
            f"Found {len(overlapping_pairs)} overlapping test-window pair(s). "
            f"This means step_bars ({self.STEP_BARS}) is likely too small "
            f"relative to test_window_bars ({self.TEST_WINDOW_BARS}), "
            f"causing OOF double-counting. "
            f"First overlap: window {overlapping_pairs[0][0]} test "
            f"[{overlapping_pairs[0][1]}, {overlapping_pairs[0][2]}) "
            f"overlaps window {overlapping_pairs[0][0] + 1} test "
            f"[{overlapping_pairs[0][3]}, {overlapping_pairs[0][4]}) "
            f"by {overlapping_pairs[0][2] - overlapping_pairs[0][3]} bars"
        )

    def test_overlap_detected_when_step_less_than_test(self):
        """Overlap MUST be detected when step_bars < test_window_bars.

        This is a companion safety check: it deliberately creates a
        config where step < test_window to prove our detection logic
        actually catches the problem.
        """
        # step < test_window → test regions will overlap
        total_bars = 50_000
        step_bars = 2000  # intentionally smaller than test_window_bars

        windows = generate_windows(
            total_bars=total_bars,
            train_window_bars=self.TRAIN_WINDOW_BARS,
            test_window_bars=self.TEST_WINDOW_BARS,
            step_bars=step_bars,
            purge_bars=self.PURGE_BARS,
            embargo_bars=self.EMBARGO_BARS,
            min_train_bars=self.MIN_TRAIN_BARS,
        )

        assert len(windows) >= 2, f"Need at least 2 windows; got {len(windows)}"

        overlapping = False
        for i in range(len(windows) - 1):
            if windows[i].test_end_idx > windows[i + 1].test_start_idx:
                overlapping = True
                break

        assert overlapping, (
            "Expected overlapping test windows when "
            f"step_bars ({step_bars}) < test_window_bars ({self.TEST_WINDOW_BARS}), "
            "but none were found.  Detection logic may be broken."
        )
