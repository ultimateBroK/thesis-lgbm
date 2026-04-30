"""Tests for validation module — walk-forward sliding window."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.validation import (
    WalkForwardWindow,
    apply_purge_embargo,
    generate_windows,
    log_windows,
    split_data,
)


def _make_df(n: int = 500) -> pl.DataFrame:
    """Create a DataFrame with n hourly rows starting 2020-01-01."""
    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            start=pl.datetime(2020, 1, 1),
            end=pl.datetime(2020, 1, 1) + pl.duration(hours=n - 1),
            interval="1h",
            eager=True,
        ),
        "value": list(range(n)),
    })


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
