"""Tests for labels module.

Tests triple-barrier labeling logic directly.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.stage_3_labels import (
    compute_average_uniqueness,
    compute_event_end,
)
from thesis.stage_3_labels.labeling import (
    _compute_labels,
    _filter_censored,
    _merge_label_columns,
)
from thesis.stage_4_training.walk_forward.hybrid import _compute_regression_target
from thesis.shared.config import Config, GRUConfig, LGBMConfig, LabelsConfig
from thesis.shared.constants import CENSORED_LABEL


@pytest.mark.unit
@pytest.mark.data
def test_labels_in_valid_set() -> None:
    """Test that labels are in {-1, 0, 1}."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    unique_labels = np.unique(labels)

    # All labels should be in {-1, 0, 1}; -2 may appear for right-censored rows
    # (last `horizon` bars with insufficient forward data to evaluate)
    valid = np.all(np.isin(unique_labels, [-2, -1, 0, 1]))
    assert valid, (
        f"Unexpected labels {unique_labels}: expected subset of {{-2, -1, 0, 1}}"
    )


@pytest.mark.unit
@pytest.mark.data
def test_upper_lower_barrier_relationship() -> None:
    """Test that upper_barrier > close and lower_barrier < close for each bar (the bug fix!)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    _, upper_barriers, lower_barriers, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # For every bar, upper barrier should be above close and lower barrier below close
    for i in range(n):
        assert upper_barriers[i] > close[i], (
            f"upper barrier {upper_barriers[i]} not > close {close[i]} at index {i}"
        )
        assert lower_barriers[i] < close[i], (
            f"lower barrier {lower_barriers[i]} not < close {close[i]} at index {i}"
        )


@pytest.mark.unit
@pytest.mark.data
def test_touched_bars_for_hold() -> None:
    """Test that touched_bars is -1 for Hold labels."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # Make high/low very close to close so barriers are never hit
    high = close + 0.1
    low = close - 0.1
    atr = np.ones(n) * 10  # Barriers will be at +/- 15

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # For Hold labels (0), touched_bar should be -1
    for i in range(n):
        if labels[i] == 0:
            assert touched_bars[i] == -1, (
                f"Hold label at {i} should have touched_bar=-1"
            )


@pytest.mark.unit
@pytest.mark.data
def test_same_bar_both_hit_counted_as_ambiguous_hold() -> None:
    """Same-bar upper/lower hit is neutral and counted for diagnostics."""
    close = np.array([100.0, 100.0, 100.0, 100.0])
    high = np.array([100.0, 103.0, 100.0, 100.0])
    low = np.array([100.0, 97.0, 100.0, 100.0])
    atr = np.ones(len(close))

    labels, _, _, touched_bars, ambiguous_count = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=2.0,
        sl_mult=2.0,
        horizon=2,
        min_atr=0.0001,
    )

    assert labels[0] == 0
    assert touched_bars[0] == 1
    assert ambiguous_count == 1


@pytest.mark.unit
@pytest.mark.data
def test_label_columns_do_not_emit_legacy_tp_sl_aliases() -> None:
    """New label output uses upper/lower barrier names only."""
    df = pl.DataFrame({"timestamp": [1, 2, 3]})
    result = _merge_label_columns(
        df,
        labels_arr=np.array([1, 0, -1], dtype=np.int32),
        upper_arr=np.array([102.0, 103.0, 104.0]),
        lower_arr=np.array([98.0, 97.0, 96.0]),
        touched_bars_arr=np.array([1, -1, 2], dtype=np.int32),
        event_end_arr=np.array([1, 3, 4], dtype=np.int32),
        sample_weight_arr=np.array([1.0, 0.8, 1.2]),
    )

    assert "upper_barrier" in result.columns
    assert "lower_barrier" in result.columns
    assert "event_end" in result.columns
    assert "sample_weight" in result.columns
    assert "tp_price" not in result.columns
    assert "sl_price" not in result.columns


@pytest.mark.unit
@pytest.mark.data
def test_event_end_uses_touch_or_horizon() -> None:
    """Touched labels end at touch offset; Hold/censored use full horizon."""
    touched = np.array([1, -1, 3, -2], dtype=np.int32)
    event_end = compute_event_end(touched, horizon=5)
    np.testing.assert_array_equal(event_end, np.array([1, 6, 5, 8], dtype=np.int32))


@pytest.mark.unit
@pytest.mark.data
def test_average_uniqueness_no_overlap_is_one() -> None:
    """Non-overlapping events keep unit sample weights after normalization."""
    event_end = np.array([0, 1, 2, 3], dtype=np.int32)
    weights = compute_average_uniqueness(event_end)
    np.testing.assert_allclose(weights, np.ones(4), rtol=1e-6)


@pytest.mark.unit
@pytest.mark.data
def test_average_uniqueness_downweights_overlap() -> None:
    """Overlapping events get lower relative uniqueness than isolated events."""
    event_end = np.array([3, 3, 3, 3, 4], dtype=np.int32)
    weights = compute_average_uniqueness(event_end)
    assert weights[1] < weights[4]
    assert weights[2] < weights[4]
    assert np.all(weights > 0)


@pytest.mark.unit
@pytest.mark.data
def test_touched_bars_for_non_hold() -> None:
    """Test that touched_bars >= 0 for non-Hold labels."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # Make high hit upper barrier quickly
    high = close + 20  # Will hit upper barrier
    low = close - 1
    atr = np.ones(n) * 10

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # For non-Hold labels, touched_bar should be >= 0
    for i in range(n - 10):  # Last 'horizon' bars may not have enough future
        if labels[i] != 0:
            assert touched_bars[i] >= 0, (
                f"Non-Hold label at {i} should have touched_bar >= 0"
            )
            assert touched_bars[i] < 10, f"touched_bar at {i} should be < horizon"


@pytest.mark.unit
@pytest.mark.data
def test_zero_atr_handled() -> None:
    """Test with zero ATR (min_atr kicks in)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.zeros(n)  # Zero ATR

    labels, upper_barriers, lower_barriers, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.1,  # min_atr should kick in
    )

    # Should still produce valid labels; -2 may appear for right-censored rows
    assert len(labels) == n
    assert np.all(np.isin(labels, [-2, -1, 0, 1]))

    # upper barrier and lower barrier should still be valid
    for i in range(n):
        assert upper_barriers[i] > close[i]
        assert lower_barriers[i] < close[i]


@pytest.mark.unit
@pytest.mark.data
def test_extreme_volatility_all_long() -> None:
    """Test with extreme volatility (all Long)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # High always hits upper barrier
    high = close + 100
    low = close - 1
    atr = np.ones(n) * 10

    labels, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # Most labels should be Long (1)
    long_count = np.sum(labels == 1)
    assert long_count > 0, "Should have some Long labels"


@pytest.mark.unit
@pytest.mark.data
def test_extreme_volatility_all_short() -> None:
    """Test with extreme volatility (all Short)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 1
    # Low always hits lower barrier
    low = close - 100
    atr = np.ones(n) * 10

    labels, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # Most labels should be Short (-1)
    short_count = np.sum(labels == -1)
    assert short_count > 0, "Should have some Short labels"


@pytest.mark.unit
@pytest.mark.data
def test_horizon_boundary() -> None:
    """Test that horizon is respected."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10
    horizon = 5

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=horizon,
        min_atr=0.0001,
    )

    # For non-Hold labels, touched_bar should be <= horizon
    # (range is i+1 to i+1+horizon, so max touched_bar = horizon)
    for i in range(n):
        if labels[i] != 0 and touched_bars[i] >= 0:
            assert touched_bars[i] <= horizon, (
                f"touched_bar {touched_bars[i]} exceeds horizon {horizon}"
            )


@pytest.mark.unit
@pytest.mark.data
def test_atr_multiplier_effect_asymmetric() -> None:
    """Test that asymmetric TP/SL multipliers create correct barrier widths."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels_small, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.0,
        sl_mult=1.0,
        horizon=10,
        min_atr=0.0001,
    )

    labels_large, _, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=3.0,
        sl_mult=3.0,
        horizon=10,
        min_atr=0.0001,
    )

    # Larger multiplier should result in more Hold labels (wider barriers)
    holds_small = np.sum(labels_small == 0)
    holds_large = np.sum(labels_large == 0)

    assert holds_large >= holds_small, (
        "Larger multiplier should produce at least as many Hold labels"
    )


@pytest.mark.unit
@pytest.mark.data
def test_no_lookahead_bias() -> None:
    """Test that labels don't use future information (no lookahead bias)."""
    n = 100
    close = np.linspace(1800, 1900, n)
    # Create a pattern where future is predictable
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels, _, _, touched_bars, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=1.5,
        sl_mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # Check that touched_bar is always in the future relative to current index
    for i in range(n):
        if labels[i] != 0 and touched_bars[i] >= 0:
            # touched_bar is relative to current position
            absolute_touch = i + touched_bars[i]
            assert absolute_touch > i, "Touch must be in the future"
            assert absolute_touch < n, "Touch must be within bounds"


@pytest.mark.unit
@pytest.mark.data
def test_asymmetric_barriers_tp_sl_ratio() -> None:
    """Test that asymmetric TP/SL multipliers create correct barrier distances."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    _, upper_barriers, lower_barriers, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        tp_mult=2.0,
        sl_mult=1.0,
        horizon=10,
        min_atr=0.0001,
    )

    for i in range(n):
        upper_dist = upper_barriers[i] - close[i]
        lower_dist = close[i] - lower_barriers[i]
        assert abs(upper_dist - 20.0) < 1e-10, (
            f"Upper barrier distance should be 20.0 (2.0 * 10.0 ATR), got {upper_dist}"
        )
        assert abs(lower_dist - 10.0) < 1e-10, (
            f"Lower barrier distance should be 10.0 (1.0 * 10.0 ATR), got {lower_dist}"
        )


# ── Regression tail censoring ───────────────────────────────────────────


def _make_regression_config(horizon_bars: int = 5) -> Config:
    """Build a minimal Config with regression objective and a given horizon."""
    cfg = Config()
    cfg.labels = LabelsConfig(horizon_bars=horizon_bars)
    cfg.model = LGBMConfig(objective="regression")
    return cfg


def _make_labeled_df(
    n: int = 30,
    close_start: float = 100.0,
    close_step: float = 1.0,
) -> pl.DataFrame:
    """Build a minimal labeled Polars DataFrame for regression testing.

    Creates a monotonically increasing close series and dummy columns
    required by ``_compute_regression_target``.
    """
    close = np.linspace(close_start, close_start + close_step * (n - 1), n)
    return pl.DataFrame(
        {
            "close": close,
            "label": np.full(n, 0, dtype=np.int32),
            "event_end": np.arange(n, dtype=np.int32),
            "timestamp": np.arange(n, dtype=np.int64),
        }
    )


@pytest.mark.unit
class TestRegressionTailCensoring:
    """Tests for regression-target computation and tail censoring."""

    # ── 1. horizon_bars NaN rows ───────────────────────────────────────

    def test_regression_drops_horizon_bars_nan_rows(self) -> None:
        """Regression-mode drops exactly ``horizon_bars`` censored tail rows."""
        horizon = 7
        n = 50
        df_in = _make_labeled_df(n=n)
        cfg = _make_regression_config(horizon_bars=horizon)

        result_df, is_regression = _compute_regression_target(df_in, cfg)

        assert is_regression is True
        assert len(result_df) == n - horizon, (
            f"Expected {n - horizon} rows after dropping {horizon} censored rows, "
            f"got {len(result_df)}"
        )
        # No NaN regression_target should remain in the result
        assert result_df["regression_target"].is_nan().sum() == 0

    # ── 2. Non-zero regression target ──────────────────────────────────

    def test_regression_target_nonzero_for_valid_rows(self) -> None:
        """Regression target mean is non-zero for a trending price series."""
        horizon = 5
        n = 100
        df_in = _make_labeled_df(n=n, close_step=0.5)  # trending up
        cfg = _make_regression_config(horizon_bars=horizon)

        result_df, is_regression = _compute_regression_target(df_in, cfg)

        assert is_regression is True
        mean_target = result_df["regression_target"].mean()
        assert mean_target is not None
        assert mean_target > 0.0, (
            f"Expected positive mean regression target for uptrend, got {mean_target}"
        )
        std_target = result_df["regression_target"].std()
        assert std_target is not None and std_target > 0.0, (
            "Expected non-zero std for regression target"
        )

    def test_regression_target_nonzero_for_volatile_series(self) -> None:
        """Regression target is non-zero even for a volatile non-monotonic series."""
        rng = np.random.default_rng(42)
        n = 100
        horizon = 5
        close = 100.0 + rng.standard_normal(n).cumsum() * 2.0
        df_in = pl.DataFrame(
            {
                "close": close,
                "label": np.full(n, 0, dtype=np.int32),
                "event_end": np.arange(n, dtype=np.int32),
                "timestamp": np.arange(n, dtype=np.int64),
            }
        )

        cfg = _make_regression_config(horizon_bars=horizon)
        result_df, is_regression = _compute_regression_target(df_in, cfg)

        assert is_regression is True
        # The mean could be positive or negative, but it should not be zero
        mean_target = result_df["regression_target"].mean()
        assert mean_target is not None
        assert abs(mean_target) > 1e-10, (
            "Expected non-zero mean regression target for volatile series"
        )

    # ── 3. _filter_censored removes NaN regression_target ──────────────

    def test_filter_censored_removes_label_censored(self) -> None:
        """_filter_censored drops rows where label == CENSORED_LABEL (-2)."""
        n = 20
        labels = np.full(n, 0, dtype=np.int32)
        labels[[3, 7, 15]] = CENSORED_LABEL  # rows 3, 7, 15 are censored
        df_in = pl.DataFrame(
            {
                "close": np.ones(n),
                "label": labels,
                "event_end": np.arange(n, dtype=np.int32),
            }
        )

        result_df = _filter_censored(df_in)

        assert result_df["label"].min() >= -1, (
            f"Censored labels should be removed; got {result_df['label'].to_list()}"
        )
        assert len(result_df) == n - 3, (
            f"Expected {n - 3} rows after dropping 3 censored, got {len(result_df)}"
        )

    def test_filter_censored_removes_nan_regression_target(self) -> None:
        """_filter_censored drops rows where regression_target is NaN."""
        n = 20
        reg_target = np.full(n, 0.02, dtype=np.float64)
        reg_target[[2, 8, 14]] = np.nan  # rows 2, 8, 14 have NaN
        df_in = pl.DataFrame(
            {
                "close": np.ones(n),
                "label": np.full(n, 0, dtype=np.int32),
                "regression_target": reg_target,
                "event_end": np.arange(n, dtype=np.int32),
            }
        )

        result_df = _filter_censored(df_in)

        assert result_df["regression_target"].is_nan().sum() == 0, (
            "All NaN regression_target rows should be removed"
        )
        assert len(result_df) == n - 3, (
            f"Expected {n - 3} rows after dropping 3 NaN rows, got {len(result_df)}"
        )

    def test_filter_censored_preserves_valid_rows(self) -> None:
        """_filter_censored leaves rows without censored labels or NaN target untouched."""
        n = 20
        reg_target = np.full(n, 0.03, dtype=np.float64)
        df_in = pl.DataFrame(
            {
                "close": np.ones(n),
                "label": np.full(n, 1, dtype=np.int32),
                "regression_target": reg_target,
                "event_end": np.arange(n, dtype=np.int32),
            }
        )

        result_df = _filter_censored(df_in)

        assert len(result_df) == n, (
            f"No rows should be dropped; expected {n}, got {len(result_df)}"
        )
        np.testing.assert_allclose(
            result_df["regression_target"].to_numpy(), reg_target, rtol=1e-12
        )

    # ── 4. _compute_regression_target correctness ──────────────────────

    def test_compute_regression_target_correct_forward_returns(self) -> None:
        """_compute_regression_target computes exact forward returns."""
        horizon = 4
        n = 20
        close = np.arange(100.0, 100.0 + n, 1.0, dtype=np.float64)
        df_in = pl.DataFrame(
            {
                "close": close,
                "label": np.full(n, 0, dtype=np.int32),
                "event_end": np.arange(n, dtype=np.int32),
                "timestamp": np.arange(n, dtype=np.int64),
            }
        )

        cfg = _make_regression_config(horizon_bars=horizon)
        result_df, is_regression = _compute_regression_target(df_in, cfg)

        assert is_regression is True
        reg_result = result_df["regression_target"].to_numpy()
        close_orig = close[: n - horizon]

        # Manual computation: reg_target[i] = (close[i+h] - close[i]) / close[i]
        expected = (close[horizon:] - close_orig) / close_orig
        np.testing.assert_allclose(reg_result, expected, rtol=1e-12)

    def test_compute_regression_target_zero_return_for_flat_prices(self) -> None:
        """_compute_regression_target yields near-zero returns for flat prices."""
        horizon = 3
        n = 30
        close = np.full(n, 50.0, dtype=np.float64)
        df_in = pl.DataFrame(
            {
                "close": close,
                "label": np.full(n, 0, dtype=np.int32),
                "event_end": np.arange(n, dtype=np.int32),
                "timestamp": np.arange(n, dtype=np.int64),
            }
        )

        cfg = _make_regression_config(horizon_bars=horizon)
        result_df, is_regression = _compute_regression_target(df_in, cfg)

        assert is_regression is True
        reg_result = result_df["regression_target"].to_numpy()
        np.testing.assert_allclose(reg_result, np.zeros(n - horizon), atol=1e-15)

    def test_compute_regression_target_full_horizon_censored(self) -> None:
        """When horizon equals data length, all rows are censored (empty result)."""
        horizon = 10
        n = 10
        df_in = _make_labeled_df(n=n)
        cfg = _make_regression_config(horizon_bars=horizon)

        result_df, is_regression = _compute_regression_target(df_in, cfg)

        assert is_regression is True
        assert len(result_df) == 0, (
            f"All rows should be censored when horizon={horizon} == n={n}"
        )

    def test_non_regression_objective_noop(self) -> None:
        """When BOTH objectives are NOT regression, _compute_regression_target is a no-op."""
        n = 30
        df_in = _make_labeled_df(n=n)
        cfg = Config()
        cfg.labels = LabelsConfig(horizon_bars=5)
        cfg.model = LGBMConfig(objective="multiclass")  # LGBM not regression
        cfg.gru = GRUConfig(objective="multiclass")  # GRU not regression

        result_df, is_regression = _compute_regression_target(df_in, cfg)

        assert is_regression is False
        assert "regression_target" not in result_df.columns
        assert len(result_df) == n, (
            f"Expected no rows dropped for non-regression objective, "
            f"got {len(result_df)}"
        )

    def test_regression_missing_close_raises(self) -> None:
        """Regression mode raises ValueError when 'close' column is absent."""
        df_in = pl.DataFrame(
            {
                "label": np.full(10, 0, dtype=np.int32),
                "timestamp": np.arange(10, dtype=np.int64),
            }
        )
        cfg = _make_regression_config(horizon_bars=3)

        with pytest.raises(ValueError, match="close"):
            _compute_regression_target(df_in, cfg)
