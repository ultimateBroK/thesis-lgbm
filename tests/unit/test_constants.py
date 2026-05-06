"""Tests for shared/constants.py — timeframe_to_ms and constant values."""

from __future__ import annotations

import pytest

from thesis.shared.constants import (
    CALIB_LR,
    CALIB_MAX_ITER,
    CENSORED_LABEL,
    CHART_COLORS,
    DIST_SHIFT_CLIP_MAX,
    DIST_SHIFT_CLIP_MIN,
    ECE_N_BINS,
    EXCLUDE_COLS,
    FEATURE_EPS,
    H1_BARS_PER_YEAR,
    SAMPLE_WEIGHT_MIN,
    STD_EPS,
    timeframe_to_ms,
)


# ---------------------------------------------------------------------------
# timeframe_to_ms
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeframeToMs:
    def test_1h(self) -> None:
        assert timeframe_to_ms("1H") == 3_600_000

    def test_4h(self) -> None:
        assert timeframe_to_ms("4H") == 4 * 3_600_000

    def test_case_insensitive(self) -> None:
        assert timeframe_to_ms("1h") == 3_600_000
        assert timeframe_to_ms("4h") == 4 * 3_600_000

    def test_5min(self) -> None:
        assert timeframe_to_ms("5MIN") == 5 * 60_000

    def test_15min(self) -> None:
        assert timeframe_to_ms("15MIN") == 15 * 60_000

    def test_minutes_with_m_suffix(self) -> None:
        assert timeframe_to_ms("5M") == 5 * 60_000
        assert timeframe_to_ms("15M") == 15 * 60_000

    def test_day(self) -> None:
        assert timeframe_to_ms("D") == 86_400_000
        assert timeframe_to_ms("1D") == 86_400_000

    def test_invalid_zero_hours(self) -> None:
        with pytest.raises(ValueError, match="hours must be > 0"):
            timeframe_to_ms("0H")

    def test_invalid_zero_minutes(self) -> None:
        with pytest.raises(ValueError, match="minutes must be > 0"):
            timeframe_to_ms("0MIN")

    def test_invalid_zero_minutes_m_suffix(self) -> None:
        with pytest.raises(ValueError, match="minutes must be > 0"):
            timeframe_to_ms("0M")

    def test_unsupported_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            timeframe_to_ms("1W")

    def test_unsupported_seconds(self) -> None:
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            timeframe_to_ms("30S")

    def test_negative_hours(self) -> None:
        with pytest.raises(ValueError):
            timeframe_to_ms("-1H")


# ---------------------------------------------------------------------------
# Constant sanity checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstants:
    def test_h1_bars_per_year(self) -> None:
        assert H1_BARS_PER_YEAR == 24 * 5 * 52

    def test_exclude_cols_is_frozenset(self) -> None:
        assert isinstance(EXCLUDE_COLS, frozenset)
        assert "timestamp" in EXCLUDE_COLS
        assert "label" in EXCLUDE_COLS
        assert "open" in EXCLUDE_COLS

    def test_chart_colors_keys(self) -> None:
        expected = {
            "primary",
            "secondary",
            "success",
            "danger",
            "warning",
            "gray",
            "long",
            "short",
            "flat",
        }
        assert set(CHART_COLORS.keys()) == expected

    def test_censored_label(self) -> None:
        assert CENSORED_LABEL == -2

    def test_eps_values(self) -> None:
        assert FEATURE_EPS < STD_EPS
        assert FEATURE_EPS > 0
        assert STD_EPS > 0

    def test_clip_range_ordering(self) -> None:
        assert DIST_SHIFT_CLIP_MIN < DIST_SHIFT_CLIP_MAX
        assert DIST_SHIFT_CLIP_MIN > 0

    def test_calibration_constants(self) -> None:
        assert ECE_N_BINS == 10
        assert CALIB_LR > 0
        assert CALIB_MAX_ITER > 0

    def test_sample_weight_min(self) -> None:
        assert 0 < SAMPLE_WEIGHT_MIN < 1
