"""Tests for labels module.

Tests triple-barrier labeling logic directly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.labels import _compute_labels


@pytest.mark.unit
@pytest.mark.data
def test_labels_in_valid_set() -> None:
    """Test that labels are in {-1, 0, 1}."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
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
def test_tp_sl_price_relationship() -> None:
    """Test that tp_price > close and sl_price < close for each bar (the bug fix!)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    _, tp_prices, sl_prices, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
        horizon=10,
        min_atr=0.0001,
    )

    # For every bar, TP should be above close and SL below close
    for i in range(n):
        assert tp_prices[i] > close[i], (
            f"TP {tp_prices[i]} not > close {close[i]} at index {i}"
        )
        assert sl_prices[i] < close[i], (
            f"SL {sl_prices[i]} not < close {close[i]} at index {i}"
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

    labels, _, _, touched_bars = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
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
def test_touched_bars_for_non_hold() -> None:
    """Test that touched_bars >= 0 for non-Hold labels."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # Make high hit TP quickly
    high = close + 20  # Will hit TP
    low = close - 1
    atr = np.ones(n) * 10

    labels, _, _, touched_bars = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
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

    labels, tp_prices, sl_prices, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
        horizon=10,
        min_atr=0.1,  # min_atr should kick in
    )

    # Should still produce valid labels; -2 may appear for right-censored rows
    assert len(labels) == n
    assert np.all(np.isin(labels, [-2, -1, 0, 1]))

    # TP and SL should still be valid
    for i in range(n):
        assert tp_prices[i] > close[i]
        assert sl_prices[i] < close[i]


@pytest.mark.unit
@pytest.mark.data
def test_extreme_volatility_all_long() -> None:
    """Test with extreme volatility (all Long)."""
    n = 50
    close = np.linspace(1800, 1900, n)
    # High always hits TP
    high = close + 100
    low = close - 1
    atr = np.ones(n) * 10

    labels, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
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
    # Low always hits SL
    low = close - 100
    atr = np.ones(n) * 10

    labels, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
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

    labels, _, _, touched_bars = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
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
def test_atr_multiplier_effect() -> None:
    """Test that larger ATR multiplier creates wider barriers."""
    n = 50
    close = np.linspace(1800, 1900, n)
    high = close + 5
    low = close - 5
    atr = np.ones(n) * 10

    labels_small, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.0,
        horizon=10,
        min_atr=0.0001,
    )

    labels_large, _, _, _ = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=3.0,
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

    labels, _, _, touched_bars = _compute_labels(
        close=close,
        high=high,
        low=low,
        atr=atr,
        mult=1.5,
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
