"""Tests for labels module.

Tests triple-barrier labeling logic directly.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.stage_3_labels.labeling import (
    _compute_labels,
    _filter_censored,
    _merge_label_columns,
    compute_average_uniqueness,
    compute_event_end,
    generate_labels,
)
from thesis.shared.config import Config, LGBMConfig, LabelsConfig
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


# ── generate_labels join-path tests ─────────────────────────────────────


def _make_features_with_ohlc_atr(n: int = 50) -> pl.DataFrame:
    """Build a features DataFrame with OHLC + ATR columns."""
    rng = np.random.default_rng(42)
    close = 1800 + rng.normal(0, 5, n).cumsum()
    return pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": close + 0.5,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
            "atr_14": np.ones(n) * 10.0,
        }
    )


def _make_minimal_config(tmp_path: Path, features_df: pl.DataFrame) -> Config:
    """Build a Config pointing to tmp parquet files for generate_labels tests."""
    feat_path = tmp_path / "features.parquet"
    ohlcv_path = tmp_path / "ohlcv.parquet"
    labels_path = tmp_path / "labels.parquet"

    features_df.write_parquet(feat_path)
    # Minimal OHLCV so _validate_paths succeeds even when the join is skipped
    pl.DataFrame(
        {
            "timestamp": [0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    ).write_parquet(ohlcv_path)

    cfg = Config()
    cfg.paths.features = str(feat_path)
    cfg.paths.ohlcv = str(ohlcv_path)
    cfg.paths.labels = str(labels_path)
    return cfg


_SCHEMA_PATCHES = (
    "thesis.stage_3_labels.labeling.FeaturesSchema.validate",
    "thesis.stage_3_labels.labeling.LabelsSchema.validate",
)


@pytest.mark.unit
def test_labels_skip_ohlcv_join_when_features_have_ohlc(
    tmp_path: Path,
) -> None:
    """When features already contain OHLC columns, ohlcv.parquet is NOT loaded."""
    features_df = _make_features_with_ohlc_atr()
    cfg = _make_minimal_config(tmp_path, features_df)

    with (
        patch(_SCHEMA_PATCHES[0]),
        patch(_SCHEMA_PATCHES[1]),
        patch(
            "thesis.stage_3_labels.labeling.pl.read_parquet",
            wraps=pl.read_parquet,
        ) as mock_read,
    ):
        generate_labels(cfg)

    # Only one read (features); OHLCV file should never be opened
    assert mock_read.call_count == 1, (
        f"Expected exactly 1 parquet read (features only), "
        f"got {mock_read.call_count}"
    )


@pytest.mark.unit
def test_labels_join_ohlcv_when_features_missing_ohlc(
    tmp_path: Path,
) -> None:
    """When features lack OHLC columns, OHLCV is loaded and joined."""
    n = 50
    rng = np.random.default_rng(42)
    close = 1800 + rng.normal(0, 5, n).cumsum()

    features_df = pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "volume": rng.integers(100, 1000, n).astype(float),
            "atr_14": np.ones(n) * 10.0,
        }
    )
    ohlcv_df = pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": close + 0.5,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
        }
    )

    feat_path = tmp_path / "features.parquet"
    ohlcv_path = tmp_path / "ohlcv.parquet"
    labels_path = tmp_path / "labels.parquet"

    features_df.write_parquet(feat_path)
    ohlcv_df.write_parquet(ohlcv_path)

    cfg = Config()
    cfg.paths.features = str(feat_path)
    cfg.paths.ohlcv = str(ohlcv_path)
    cfg.paths.labels = str(labels_path)

    with patch(_SCHEMA_PATCHES[0]), patch(_SCHEMA_PATCHES[1]):
        generate_labels(cfg)

    result = pl.read_parquet(labels_path)
    for col in ("open", "high", "low", "close"):
        assert col in result.columns, f"Missing OHLC column after join: {col}"


@pytest.mark.unit
def test_labels_no_right_columns(tmp_path: Path) -> None:
    """Output DataFrame has zero columns matching the *_right join-artifact pattern."""
    features_df = _make_features_with_ohlc_atr()
    cfg = _make_minimal_config(tmp_path, features_df)

    with patch(_SCHEMA_PATCHES[0]), patch(_SCHEMA_PATCHES[1]):
        generate_labels(cfg)

    result = pl.read_parquet(cfg.paths.labels)
    right_cols = [c for c in result.columns if c.endswith("_right")]
    assert len(right_cols) == 0, f"Found *_right columns in output: {right_cols}"


@pytest.mark.unit
def test_labels_missing_atr_raises(tmp_path: Path) -> None:
    """Missing ATR column raises ValueError before any labeling work."""
    n = 50
    rng = np.random.default_rng(42)
    close = 1800 + rng.normal(0, 5, n).cumsum()
    # Features WITH OHLC but WITHOUT atr_14
    features_df = pl.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.int64),
            "open": close + 0.5,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )
    cfg = _make_minimal_config(tmp_path, features_df)

    with patch(_SCHEMA_PATCHES[0]), patch(_SCHEMA_PATCHES[1]):
        with pytest.raises((ValueError, pl.exceptions.ColumnNotFoundError), match="atr"):
            generate_labels(cfg)
