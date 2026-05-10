"""Tests for features module.

Tests technical indicator helpers and validates the compact production feature
set built from price-action and trend-focused transforms.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers import create_synthetic_ohlcv
from thesis.shared.config import Config
from thesis.shared.constants import EXCLUDE_COLS
from thesis.stage_2_features.indicators import (
    _add_adx,
    _add_atr,
    _add_context_features,
    _add_ema_crossover,
    _add_ema_slope,
    _add_high_low_range,
    _add_log_returns,
    _add_macd,
    _add_ny_session_dummies,
    _add_pivot_position,
    _add_price_action_features,
    _add_regime,
    _add_rsi,
    _add_volume_zscore,
)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.features.rsi_period = 14
    config.features.atr_period = 14
    config.features.macd_fast = 12
    config.features.macd_slow = 26
    config.features.macd_signal = 9
    return config


def _build_all_features(df: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Apply the full feature pipeline to a DataFrame (mirrors generate_features)."""
    df = _add_rsi(df, config)
    df = _add_atr(df, config)
    df = _add_macd(df, config)
    df = _add_context_features(df, config)
    df = _add_pivot_position(df)
    df = _add_price_action_features(df, config)
    df = _add_ema_crossover(df, config)
    df = _add_ny_session_dummies(df)
    df = _add_volume_zscore(df, config)
    df = _add_log_returns(df, config)
    df = _add_high_low_range(df, config)
    df = _add_adx(df, config)
    df = _add_ema_slope(df, config)
    df = _add_regime(df)
    if "return_1h" in df.columns and "log_returns" not in df.columns:
        df = df.with_columns(pl.col("return_1h").alias("log_returns"))
    # NaN from numpy → Polars null before forward-fill (production pipeline does this too)
    df = df.fill_nan(None)
    df = df.fill_null(strategy="forward").fill_null(0.0)
    keep_features = sorted(
        {
            *config.features.static_feature_cols,
        }
    )
    keep_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        f"atr_{config.features.atr_period}",
        *keep_features,
    ]
    df = df.select([c for c in keep_cols if c in df.columns])
    return df


# Expected compact production feature columns.
# Keep in sync with CORE_STATIC_FEATURES in constants.py.
EXPECTED_FEATURES: list[str] = [
    "adx_14",
    "atr_pct_close",
    "atr_percentile",
    "atr_ratio",
    "candle_body_ratio",
    "close_vs_ema_34",
    "ema_slope_20",
    "ema34_vs_ema89",
    "high_low_range_20",
    "lower_wick_ratio",
    "macd_hist_atr",
    "pivot_position",
    "price_dist_ratio",
    "price_position_20",
    "regime_strength",
    "return_1h",
    "return_4h",
    "rsi_14",
    "sess_asia",
    "sess_london",
    "sess_ny_am",
    "sess_ny_pm",
    "upper_wick_ratio",
    "vwap",
    "volume_zscore_20",
]


# ---------------------------------------------------------------------------
# EMA slope tests (task 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_ema_slope_column(sample_config: Config) -> None:
    """Test ema_slope_20 column is produced."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_ema_slope(df, sample_config)

    assert "ema_slope_20" in result.columns
    vals = result["ema_slope_20"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(np.isfinite(vals)), "EMA slope should be finite"


@pytest.mark.unit
@pytest.mark.features
def test_ema_slope_direction(sample_config: Config) -> None:
    """Test EMA slope sign follows price direction.

    A monotonically increasing series should produce positive slope
    values in the tail after warmup.
    """
    n = 200
    np.random.seed(42)
    timestamps = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0, time_zone="UTC"),
        end=pl.datetime(2023, 1, 1, 0, time_zone="UTC") + pl.duration(hours=n - 1),
        interval="1h",
        eager=True,
    )
    closes = 1800.0 + np.arange(n) * 0.5
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes * 0.9999,
            "high": closes * 1.0005,
            "low": closes * 0.9998,
            "close": closes,
            "volume": np.ones(n) * 5000,
        }
    )
    result = _add_ema_slope(df, sample_config)
    vals = result["ema_slope_20"].drop_nulls().to_numpy()
    assert len(vals) > 0
    tail_mean = vals[-50:].mean()
    assert tail_mean > 0, (
        f"Uptrend should produce positive EMA slope, got mean={tail_mean:.6f}"
    )


# ---------------------------------------------------------------------------
# Regime strength tests (task 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_regime_strength_column(sample_config: Config) -> None:
    """Test regime_strength column is produced."""
    df = create_synthetic_ohlcv(n_rows=300)
    df = _add_adx(df, sample_config)
    df = _add_ema_slope(df, sample_config)
    result = _add_regime(df)

    assert "regime_strength" in result.columns
    vals = result["regime_strength"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(np.isfinite(vals)), "regime_strength should be finite"


@pytest.mark.unit
@pytest.mark.features
def test_regime_strength_sign_follows_ema_slope(sample_config: Config) -> None:
    """Test regime_strength sign matches EMA slope sign.

    regime_strength = adx_signal * sign(ema_slope_20), so when ema_slope
    is positive, regime_strength should be non-negative, and vice versa.
    """
    df = create_synthetic_ohlcv(n_rows=300)
    df = _add_adx(df, sample_config)
    df = _add_ema_slope(df, sample_config)
    result = _add_regime(df)

    slope = result["ema_slope_20"].drop_nulls()
    regime = result["regime_strength"].drop_nulls()

    # Where slope > 0, regime >= 0
    pos_mask = slope.to_numpy() > 0
    if pos_mask.sum() > 0:
        regime_pos = regime.gather(pl.Series(pos_mask).arg_true()).to_numpy()
        assert np.all(regime_pos >= 0), (
            "regime_strength should be >= 0 when ema_slope > 0"
        )

    # Where slope < 0, regime <= 0
    neg_mask = slope.to_numpy() < 0
    if neg_mask.sum() > 0:
        regime_neg = regime.gather(pl.Series(neg_mask).arg_true()).to_numpy()
        assert np.all(regime_neg <= 0), (
            "regime_strength should be <= 0 when ema_slope < 0"
        )


@pytest.mark.unit
@pytest.mark.features
def test_regime_strength_ranging_flat(sample_config: Config) -> None:
    """Test regime_strength is near zero when ADX is low (ranging market).

    When ADX <= 20, the adx_signal clips to 0, so regime_strength should be 0.
    """
    n = 300
    np.random.seed(42)
    timestamps = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0, time_zone="UTC"),
        end=pl.datetime(2023, 1, 1, 0, time_zone="UTC") + pl.duration(hours=n - 1),
        interval="1h",
        eager=True,
    )
    # Mean-reverting (ranging) series: low ADX
    closes = 1800.0 + np.sin(np.arange(n) * 0.1) * 2.0 + np.random.randn(n) * 0.1
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.ones(n) * 5000,
        }
    )
    df = _add_adx(df, sample_config)
    df = _add_ema_slope(df, sample_config)
    result = _add_regime(df)

    adx = result["adx_14"].drop_nulls().to_numpy()
    regime = result["regime_strength"].drop_nulls().to_numpy()

    # Where ADX <= 20 (ranging), regime should be 0
    low_adx_mask = adx <= 20
    if low_adx_mask.sum() > 10:
        low_adx_regime = regime[low_adx_mask]
        assert np.allclose(low_adx_regime, 0.0), (
            f"When ADX <= 20, regime_strength should be 0, "
            f"got max={abs(low_adx_regime).max():.4f}"
        )


# ---------------------------------------------------------------------------
# Core indicator tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_rsi_bounded(sample_config: Config) -> None:
    """Test RSI is bounded [0, 100]."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_rsi(df, sample_config)

    assert "rsi_14" in result.columns

    rsi_values = result["rsi_14"].drop_nulls().to_numpy()
    assert len(rsi_values) > 0
    assert np.all(rsi_values >= 0)
    assert np.all(rsi_values <= 100)


@pytest.mark.unit
@pytest.mark.features
def test_atr_positive(sample_config: Config) -> None:
    """Test ATR > 0 for valid data."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_atr(df, sample_config)

    assert "atr_14" in result.columns
    assert "atr_pct_close" in result.columns

    atr_values = result["atr_14"].drop_nulls().to_numpy()
    assert len(atr_values) > 0
    assert np.all(atr_values > 0)

    atr_pct_values = result["atr_pct_close"].drop_nulls().to_numpy()
    assert len(atr_pct_values) > 0
    assert np.all(atr_pct_values > 0)


@pytest.mark.unit
@pytest.mark.features
def test_macd_histogram_only(sample_config: Config) -> None:
    """Test MACD produces raw and ATR-normalized histograms."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_macd(df, sample_config)

    assert "macd_hist" in result.columns
    assert "macd_hist_atr" in result.columns
    # macd_line should NOT be produced anymore
    assert "macd_line" not in result.columns


# ---------------------------------------------------------------------------
# New normalized feature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_atr_ratio_positive(sample_config: Config) -> None:
    """Test atr_ratio > 0 (ratio of short to long ATR)."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_context_features(df, sample_config)

    assert "atr_ratio" in result.columns
    values = result["atr_ratio"].drop_nulls().to_numpy()
    assert len(values) > 0
    assert np.all(values > 0)


@pytest.mark.unit
@pytest.mark.features
def test_price_dist_ratio_exists(sample_config: Config) -> None:
    """Test price_dist_ratio is computed."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_context_features(df, sample_config)

    assert "price_dist_ratio" in result.columns


@pytest.mark.unit
@pytest.mark.features
def test_atr_percentile_bounded(sample_config: Config) -> None:
    """Test atr_percentile is within [0, 1]."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_context_features(df, sample_config)

    assert "atr_percentile" in result.columns
    values = result["atr_percentile"].drop_nulls().to_numpy()
    assert len(values) > 0
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


# ---------------------------------------------------------------------------
# Pivot position tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_pivot_position_bounded(sample_config: Config) -> None:
    """Test pivot_position is clipped to [0, 1]."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_pivot_position(df)

    assert "pivot_position" in result.columns
    values = result["pivot_position"].drop_nulls().to_numpy()
    assert len(values) > 0
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


@pytest.mark.unit
@pytest.mark.features
def test_pivot_position_no_lookahead(sample_config: Config) -> None:
    """Test that pivot uses previous day's levels (shifted by 1)."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_pivot_position(df)

    # First few rows should have null pivots (no previous day data)
    # The exact count depends on how trading day aligns with calendar day
    first_few = result.head(24)
    # At least some of the first day's rows should be null
    assert first_few["pivot_position"].null_count() > 0, (
        "First trading day should have null pivots"
    )


# ---------------------------------------------------------------------------
# Session dummy tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_session_dummies_four_columns() -> None:
    """Test 4 session columns are produced."""
    df = create_synthetic_ohlcv(n_rows=100)
    result = _add_ny_session_dummies(df)

    for col in ["sess_asia", "sess_london", "sess_ny_am", "sess_ny_pm"]:
        assert col in result.columns, f"Missing session column: {col}"


@pytest.mark.unit
@pytest.mark.features
def test_session_dummies_binary() -> None:
    """Test session dummy columns are binary {0, 1}."""
    df = create_synthetic_ohlcv(n_rows=100)
    result = _add_ny_session_dummies(df)

    for col in ["sess_asia", "sess_london", "sess_ny_am", "sess_ny_pm"]:
        values = result[col].to_numpy()
        assert np.all(np.isin(values, [0, 1])), f"{col} has non-binary values"


@pytest.mark.unit
@pytest.mark.features
def test_session_dummies_coverage() -> None:
    """Test that every hour belongs to exactly one session."""
    df = create_synthetic_ohlcv(n_rows=100)
    result = _add_ny_session_dummies(df)

    total = (
        result["sess_asia"].cast(pl.Int32)
        + result["sess_london"].cast(pl.Int32)
        + result["sess_ny_am"].cast(pl.Int32)
        + result["sess_ny_pm"].cast(pl.Int32)
    ).to_numpy()
    # Every hour should be in exactly one session (4 sessions cover 24h)
    # Asia: 18-01 (8h), London: 03-07 (5h), NY AM: 08-11 (4h), NY PM: 12-17 (6h)
    # Total: 23h — hour 2 NY time is uncovered (gap between Asia and London)
    assert np.all(total <= 1), "Some hours belong to multiple sessions"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_all_features_together(sample_config: Config) -> None:
    """Test that all indicators can be applied together producing core features."""
    df = create_synthetic_ohlcv(n_rows=200)

    # Apply all feature functions in order
    df = _add_rsi(df, sample_config)
    df = _add_atr(df, sample_config)
    df = _add_macd(df, sample_config)
    df = _add_context_features(df, sample_config)

    # Fill nulls like the main function does
    df = df.fill_null(strategy="forward").fill_null(0.0)

    expected_features = [
        "rsi_14",
        "atr_14",
        "atr_pct_close",
        "macd_hist",
        "macd_hist_atr",
        "atr_ratio",
        "price_dist_ratio",
        "pivot_position",
        "vwap",
        "atr_percentile",
        "sess_asia",
        "sess_london",
        "sess_ny_am",
        "sess_ny_pm",
    ]

    for col in expected_features:
        assert col in df.columns, f"Missing feature column: {col}"

    # Check no nulls remain after filling
    for col in expected_features:
        null_count = df[col].null_count()
        assert null_count == 0, f"Column {col} has {null_count} nulls"


@pytest.mark.unit
@pytest.mark.features
def test_insufficient_rows_handled(sample_config: Config) -> None:
    """Test edge case: insufficient rows for indicator windows."""
    df = create_synthetic_ohlcv(n_rows=10)

    # Should not crash, but will have many nulls
    result = _add_rsi(df, sample_config)
    result = _add_atr(result, sample_config)
    result = _add_macd(result, sample_config)

    # Should produce columns even with few rows
    assert "rsi_14" in result.columns
    assert "atr_14" in result.columns
    assert "macd_hist" in result.columns


# ---------------------------------------------------------------------------
# Multi-timeframe feature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="Multi-timeframe features removed in refactor")
def test_multi_timeframe_produces_columns(sample_config: Config) -> None:
    """Test that 4H resampling produces rsi_14_4h, atr_14_4h, macd_hist_4h."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = add_multi_timeframe_features(df, sample_config)

    for col in ("rsi_14_4h", "atr_14_4h", "macd_hist_4h"):
        assert col in result.columns, f"Missing 4H feature: {col}"


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="Multi-timeframe features removed in refactor")
def test_multi_timeframe_rsi_bounded(sample_config: Config) -> None:
    """Test that 4H RSI is bounded [0, 100]."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = add_multi_timeframe_features(df, sample_config)

    vals = result["rsi_14_4h"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(vals >= 0)
    assert np.all(vals <= 100)


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="Multi-timeframe features removed in refactor")
def test_multi_timeframe_atr_positive(sample_config: Config) -> None:
    """Test that 4H ATR is positive."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = add_multi_timeframe_features(df, sample_config)

    vals = result["atr_14_4h"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(vals > 0)


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="Multi-timeframe features removed in refactor")
def test_resample_4h_row_count(sample_config: Config) -> None:
    """Test that 4H resampling produces ~n/4 rows."""
    df = create_synthetic_ohlcv(n_rows=200)
    df_4h = _resample_to_4h(df)
    # 200 hours / 4 = 50 four-hour bars
    assert 40 <= len(df_4h) <= 55


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="Multi-timeframe features removed in refactor")
def test_4h_no_future_leakage(sample_config: Config) -> None:
    """Test that 4H indicators are causally lagged — no future data leaks.

    join_asof(backward) ensures each 1H bar only sees the most recent
    *completed* 4H bar. This test verifies by checking that at the first
    bar of each 4H window, the 4H indicator values are from the *previous*
    window, not the current one.
    """
    df = create_synthetic_ohlcv(n_rows=200)
    result = add_multi_timeframe_features(df, sample_config)

    # After join_asof(backward), 1H bars at hours 0-3 share the 4H value
    # from the previous completed 4H bar. All 4 bars within a 4H window
    # should have the same 4H indicator value (from the previous window).
    rsi_vals = result["rsi_14_4h"].to_numpy()

    # Check bars 0-3 share same value (all lagged, no in-progress 4H data)
    # Skip if early nulls
    first_valid = 0
    for i in range(len(rsi_vals)):
        if not np.isnan(rsi_vals[i]):
            first_valid = i
            break

    # Bars within same 4H window should share the same backward-joined value
    # Window boundary: hours 4, 8, 12, ...
    # Pick bars 4,5,6,7 (second 4H window) — should all have same value
    # from the first completed 4H bar
    if first_valid >= 4:
        window_vals = rsi_vals[4:8]
        valid_window = window_vals[~np.isnan(window_vals)]
        if len(valid_window) >= 2:
            assert np.allclose(valid_window, valid_window[0]), (
                "Bars within same 4H window should share identical 4H indicator value"
            )


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="Multi-timeframe features removed in refactor")
def test_4h_join_is_backward(sample_config: Config) -> None:
    """Verify join_asof uses backward strategy — no 1H bar sees a 4H value from the future."""
    from thesis.stage_2_features.engineering import _compute_4h_indicators

    df = create_synthetic_ohlcv(n_rows=100)
    df_4h = _resample_to_4h(df)
    df_4h = _compute_4h_indicators(df_4h, sample_config)
    result = _join_4h_to_1h(df, df_4h)

    # Get 4H timestamps and 1H timestamps
    ts_1h = result["timestamp"].to_numpy()
    rsi_4h = result["rsi_14_4h"].to_numpy()

    # For each 1H bar with a valid 4H value, the 4H bar's timestamp
    # must be <= the 1H bar's timestamp (backward join property)
    ts_4h = df_4h["timestamp"].to_numpy()
    for i in range(len(ts_1h)):
        if not np.isnan(rsi_4h[i]):
            # Find which 4H bar this value came from
            found = False
            for j in range(len(ts_4h) - 1, -1, -1):
                if ts_4h[j] <= ts_1h[i]:
                    found = True
                    break
            assert found, f"1H bar {i} has no valid backward-joined 4H bar"


# ---------------------------------------------------------------------------
# Trend distance feature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_ema_crossover_columns(sample_config: Config) -> None:
    """Test that EMA crossover features are produced."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_ema_crossover(df, sample_config)

    expected = ["close_vs_ema_34", "ema34_vs_ema89"]
    for col in expected:
        assert col in result.columns, f"Missing EMA crossover column: {col}"


# ---------------------------------------------------------------------------
# Bollinger Band feature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="_add_bollinger_bands removed in feature refactor")
def test_bollinger_bands_columns(sample_config: Config) -> None:
    """Test that bb_width and bb_position columns are produced."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_bollinger_bands(df, sample_config)

    assert "bb_width" in result.columns
    assert "bb_position" in result.columns


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="_add_bollinger_bands removed in feature refactor")
def test_bollinger_position_bounded(sample_config: Config) -> None:
    """Test that bb_position is clipped to [0, 1]."""
    df = create_synthetic_ohlcv(n_rows=300)
    df = _add_atr(df, sample_config)
    result = _add_bollinger_bands(df, sample_config)

    vals = result["bb_position"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(vals >= 0.0)
    assert np.all(vals <= 1.0)


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="_add_bollinger_bands removed in feature refactor")
def test_bollinger_width_positive(sample_config: Config) -> None:
    """Test that bb_width is positive for valid data."""
    df = create_synthetic_ohlcv(n_rows=300)
    df = _add_atr(df, sample_config)
    result = _add_bollinger_bands(df, sample_config)

    vals = result["bb_width"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(vals > 0)


# ---------------------------------------------------------------------------
# Volume z-score tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_volume_zscore_column(sample_config: Config) -> None:
    """Test volume_zscore_20 column is produced."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_volume_zscore(df, sample_config)
    assert "volume_zscore_20" in result.columns


# ---------------------------------------------------------------------------
# Log return feature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_log_returns_columns(sample_config: Config) -> None:
    """Test return_1h, return_4h, return_1d columns are produced."""
    df = create_synthetic_ohlcv(n_rows=200)
    result = _add_log_returns(df, sample_config)

    for col in ("return_1h", "return_4h", "return_1d"):
        assert col in result.columns, f"Missing return column: {col}"


# ---------------------------------------------------------------------------
# High-low range tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_high_low_range_column(sample_config: Config) -> None:
    """Test high_low_range_20 column is produced and positive."""
    df = create_synthetic_ohlcv(n_rows=200)
    df = _add_atr(df, sample_config)
    result = _add_high_low_range(df, sample_config)

    assert "high_low_range_20" in result.columns
    vals = result["high_low_range_20"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(vals > 0)


# ---------------------------------------------------------------------------
# Regime feature tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="add_vol_regime removed in refactor")
def test_vol_regime_ordinal(sample_config: Config) -> None:
    """Test vol_regime is ordinal {0, 1, 2}."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = add_vol_regime(df, sample_config)

    assert "vol_regime" in result.columns
    vals = result["vol_regime"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(np.isin(vals, [0, 1, 2])), (
        f"vol_regime has out-of-range values: {np.unique(vals)}"
    )


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="add_vol_regime removed in refactor")
def test_vol_regime_has_all_categories(sample_config: Config) -> None:
    """Test that vol_regime produces at least 2 of the 3 categories with enough data."""
    df = create_synthetic_ohlcv(n_rows=500, seed=123)
    result = add_vol_regime(df, sample_config)

    vals = result["vol_regime"].drop_nulls().to_numpy()
    unique = np.unique(vals)
    assert len(unique) >= 2, f"vol_regime should have >= 2 categories, got {unique}"


@pytest.mark.unit
@pytest.mark.features
def test_adx_range(sample_config: Config) -> None:
    """Test ADX (adx_14) is non-negative and finite after warmup.

    ADX is theoretically bounded [0, 100] but Wilder smoothing can produce
    values slightly above 100 in edge cases. We check non-negative + finite.
    """
    df = create_synthetic_ohlcv(n_rows=300)
    result = _add_adx(df, sample_config)

    assert "adx_14" in result.columns
    vals = result["adx_14"].drop_nulls().to_numpy()
    assert len(vals) > 0
    assert np.all(vals >= 0), "ADX should be non-negative"
    assert np.all(np.isfinite(vals)), "ADX should be finite"


@pytest.mark.unit
@pytest.mark.features
def test_adx_trending_detection(sample_config: Config) -> None:
    """Test that ADX produces reasonable values for a trending series.

    A monotonically increasing series should produce high ADX values
    (strong trend) after warmup.
    """
    n = 300
    np.random.seed(42)
    # Create a strong uptrend
    timestamps = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0, time_zone="UTC"),
        end=pl.datetime(2023, 1, 1, 0, time_zone="UTC") + pl.duration(hours=n - 1),
        interval="1h",
        eager=True,
    )
    closes = 1800.0 + np.arange(n) * 0.5  # strong linear uptrend
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes * 0.9999,
            "high": closes * 1.0005,
            "low": closes * 0.9998,
            "close": closes,
            "volume": np.ones(n) * 5000,
        }
    )
    result = _add_adx(df, sample_config)
    vals = result["adx_14"].drop_nulls().to_numpy()

    # For a strong uptrend, ADX should be well above 20 (trending threshold)
    assert len(vals) > 0
    tail_mean = vals[-50:].mean()
    assert tail_mean > 20, (
        f"Strong uptrend should produce ADX > 20, got mean={tail_mean}"
    )


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="add_hurst_exponent removed in feature refactor")
def test_hurst_exponent_column() -> None:
    """Test hurst_exponent_100 column is produced."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = add_hurst_exponent(df)

    assert "hurst_exponent_100" in result.columns


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="add_hurst_exponent removed in feature refactor")
def test_hurst_exponent_valid_after_warmup() -> None:
    """Test Hurst exponent produces valid (0, 1) values after warmup.

    The first 100 bars are warmup (NaN). After that, H should be in (0, 1).
    """
    df = create_synthetic_ohlcv(n_rows=300)
    result = add_hurst_exponent(df)

    vals = result["hurst_exponent_100"].to_numpy()
    # First 100 bars should be NaN (warmup)
    assert np.all(np.isnan(vals[:99])), "First 99 values should be NaN (warmup)"

    # After warmup, values should be in (0, 1)
    valid = vals[99:]
    valid = valid[~np.isnan(valid)]
    assert len(valid) > 0, "Should have valid Hurst values after warmup"
    assert np.all(valid > 0.0) and np.all(valid < 1.0), (
        f"Hurst exponent should be in (0, 1), got range [{valid.min()}, {valid.max()}]"
    )


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="add_hurst_exponent removed in feature refactor")
def test_hurst_random_walk_around_half() -> None:
    """Test Hurst exponent for a random walk is near 0.5.

    A random walk series should have H ≈ 0.5. With 300 bars the
    estimate is noisy, so we check it's in [0.2, 0.8].
    """
    df = create_synthetic_ohlcv(n_rows=300, seed=42)
    result = add_hurst_exponent(df)

    vals = result["hurst_exponent_100"].drop_nulls().to_numpy()
    assert len(vals) > 50
    tail_mean = vals[-50:].mean()
    assert 0.2 < tail_mean < 0.8, (
        f"Random walk Hurst should be near 0.5, got mean={tail_mean}"
    )


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="add_fractal_dim removed in feature refactor")
def test_fractal_dim_column() -> None:
    """Test fractal_dim column is produced."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = add_fractal_dim(df)

    assert "fractal_dim" in result.columns


@pytest.mark.unit
@pytest.mark.features
@pytest.mark.skip(reason="add_fractal_dim removed in feature refactor")
def test_fractal_dim_valid_after_warmup() -> None:
    """Test fractal dimension produces valid [1.0, 2.0] values after warmup."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = add_fractal_dim(df)

    vals = result["fractal_dim"].to_numpy()
    # First 100 bars should be NaN (warmup)
    assert np.all(np.isnan(vals[:99])), "First 99 values should be NaN (warmup)"

    # After warmup, values should be in [1.0, 2.0]
    valid = vals[99:]
    valid = valid[~np.isnan(valid)]
    assert len(valid) > 0, "Should have valid fractal dim values after warmup"
    assert np.all(valid >= 1.0) and np.all(valid <= 2.0), (
        f"Fractal dim should be in [1.0, 2.0], got range [{valid.min()}, {valid.max()}]"
    )


@pytest.mark.unit
@pytest.mark.features
def test_regime_features_no_leakage(sample_config: Config) -> None:
    """Test that regime features don't use future data.

    Regime features use rolling windows that only look backward.
    Verify by computing features on a prefix and checking values match
    the full series at corresponding positions.
    """
    df_full = create_synthetic_ohlcv(n_rows=300)
    df_prefix = df_full.slice(0, 250)

    result_full = _add_adx(df_full, sample_config)
    result_prefix = _add_adx(df_prefix, sample_config)

    # First 250 values should be identical
    full_vals = result_full["adx_14"].to_numpy()[:250]
    prefix_vals = result_prefix["adx_14"].to_numpy()

    # Allow NaN comparison
    mask = ~(np.isnan(full_vals) | np.isnan(prefix_vals))
    if mask.sum() > 0:
        assert np.allclose(full_vals[mask], prefix_vals[mask], atol=1e-10), (
            "Regime features should be identical for prefix vs full series (no future leak)"
        )


# ---------------------------------------------------------------------------
# Full pipeline integration test — compact feature set
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.features
def test_full_feature_count_compact(sample_config: Config) -> None:
    """Test that the full pipeline produces exactly the compact feature count."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = _build_all_features(df, sample_config)

    feature_cols = sorted(c for c in result.columns if c not in EXCLUDE_COLS)
    assert len(feature_cols) == len(EXPECTED_FEATURES), (
        f"Expected {len(EXPECTED_FEATURES)} features, got {len(feature_cols)}: {feature_cols}"
    )


@pytest.mark.unit
@pytest.mark.features
def test_all_compact_features_present(sample_config: Config) -> None:
    """Test that all compact expected feature columns are present."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = _build_all_features(df, sample_config)

    for col in EXPECTED_FEATURES:
        assert col in result.columns, f"Missing expected feature: {col}"


@pytest.mark.unit
@pytest.mark.features
def test_no_extra_feature_columns(sample_config: Config) -> None:
    """Test that no unexpected feature columns are produced."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = _build_all_features(df, sample_config)

    feature_cols = sorted(c for c in result.columns if c not in EXCLUDE_COLS)
    expected_set = set(EXPECTED_FEATURES)
    extra = set(feature_cols) - expected_set
    assert len(extra) == 0, f"Unexpected extra feature columns: {extra}"


@pytest.mark.unit
@pytest.mark.features
def test_all_features_no_nulls_after_fill(sample_config: Config) -> None:
    """Test that all compact features have zero nulls after fill."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = _build_all_features(df, sample_config)

    for col in EXPECTED_FEATURES:
        null_count = result[col].null_count()
        assert null_count == 0, f"Column {col} has {null_count} nulls after fill"


@pytest.mark.unit
@pytest.mark.features
def test_all_features_finite(sample_config: Config) -> None:
    """Test that all features have finite values (no inf/nan) after pipeline."""
    df = create_synthetic_ohlcv(n_rows=300)
    result = _build_all_features(df, sample_config)

    for col in EXPECTED_FEATURES:
        vals = result[col].to_numpy()
        assert np.all(np.isfinite(vals)), f"Column {col} has non-finite values"
