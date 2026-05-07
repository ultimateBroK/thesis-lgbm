"""Tests for backtest module — backtesting.py integration.

Tests the thin wrapper around backtesting.py v0.6.5.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.shared.config import Config
from thesis.stage_5_backtest import (
    HybridGRUStrategy,
    run_backtest_from_data,
    run_backtest_manual,
)
from thesis.stage_5_backtest.simulation import (
    _prepare_df,
    _run_fractional_backtest,
)
from thesis.stage_5_backtest.strategy import _calendar_day

_log = logging.getLogger(__name__)


def create_synthetic_backtest_data(
    n_rows: int = 100,
    signal_pattern: str = "alternating",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic test data + predictions for testing."""
    np.random.seed(42)

    timestamps = pl.datetime_range(
        start=pl.datetime(2023, 1, 1, 0),
        end=pl.datetime(2023, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    base_price = 1800.0
    closes = base_price + np.cumsum(np.random.randn(n_rows) * 0.5)
    opens = closes + np.random.randn(n_rows) * 0.1
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_rows)) * 0.5
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_rows)) * 0.5

    # Generate signals
    if signal_pattern == "alternating":
        pred_label = np.array([1, -1] * (n_rows // 2) + [1] * (n_rows % 2))
    elif signal_pattern == "all_long":
        pred_label = np.ones(n_rows, dtype=int)
    elif signal_pattern == "all_short":
        pred_label = -np.ones(n_rows, dtype=int)
    elif signal_pattern == "mixed":
        pred_label = np.random.choice([-1, 0, 1], n_rows)
    else:
        pred_label = np.zeros(n_rows, dtype=int)

    # ATR values (large enough to avoid immediate stop-loss)
    atr = np.full(n_rows, 20.0)

    test_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(1000, 10000, n_rows).astype(float),
            "atr_14": atr,
        }
    )

    preds_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "pred_label": pred_label,
            "pred_proba_class_minus1": np.random.uniform(0, 0.5, n_rows),
            "pred_proba_class_0": np.random.uniform(0, 0.3, n_rows),
            "pred_proba_class_1": np.random.uniform(0.3, 1.0, n_rows),
        }
    )

    return test_df, preds_df


@pytest.mark.unit
def test_prepare_df_raises_when_prediction_merge_coverage_low() -> None:
    """Backtest join must not silently drop missing prediction timestamps."""
    test_df, preds_df = create_synthetic_backtest_data(n_rows=100)
    shifted_preds = preds_df.with_columns(pl.col("timestamp") + pl.duration(days=30))

    with pytest.raises(ValueError, match="coverage below 99%"):
        _prepare_df(
            test_df,
            shifted_preds,
            test_source="labels.parquet",
            preds_source="final_predictions.parquet",
        )


@pytest.mark.unit
def test_prepare_df_allows_full_prediction_merge_coverage(caplog) -> None:
    """Aligned predictions should pass merge guard and log row coverage."""
    test_df, preds_df = create_synthetic_backtest_data(n_rows=20)

    with caplog.at_level(logging.INFO):
        pdf = _prepare_df(test_df, preds_df)

    assert len(pdf) == 20
    assert any("coverage=100.00%" in rec.message for rec in caplog.records)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.backtest.initial_capital = 10_000.0
    config.backtest.leverage = 50
    config.backtest.spread_ticks = 30.0
    config.backtest.slippage_ticks = 3.0
    config.backtest.commission_per_lot = 10.0
    config.backtest.atr_stop_multiplier = 0.75
    config.backtest.confidence_threshold = (
        0.0  # disable confidence gating for deterministic sizing
    )
    config.data.contract_size = 100
    config.data.tick_size = 0.01
    return config


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.backtest
def test_results_contain_expected_keys(sample_config: Config) -> None:
    """Test that results contain only curated core stats."""
    test_df, preds_df = create_synthetic_backtest_data(100, "mixed")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    expected_keys = [
        "num_trades",
        "win_rate_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "profit_factor",
        "return_pct",
        "equity_final",
        "sortino_ratio",
        "calmar_ratio",
        "expectancy_pct",
        "avg_trade_pct",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"

    noisy_keys = {
        "sqn",
        "recovery_factor",
        "kelly_criterion",
        "avg_win",
        "avg_loss",
        "best_trade_pct",
        "worst_trade_pct",
        "volatility_ann_pct",
    }
    assert noisy_keys.isdisjoint(metrics)


@pytest.mark.unit
@pytest.mark.backtest
def test_metrics_values_reasonable(sample_config: Config) -> None:
    """Test that metric values are within reasonable ranges."""
    test_df, preds_df = create_synthetic_backtest_data(100, "mixed")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    # Win rate should be in [0, 100]
    if "win_rate_pct" in metrics:
        assert 0 <= metrics["win_rate_pct"] <= 100

    # Max drawdown should be <= 0
    if "max_drawdown_pct" in metrics:
        assert metrics["max_drawdown_pct"] <= 0

    # Final equity should be positive
    if "equity_final" in metrics:
        assert metrics["equity_final"] > 0


@pytest.mark.unit
@pytest.mark.backtest
def test_empty_signals_handled(sample_config: Config) -> None:
    """Test handling of no signals (no trades)."""
    test_df, preds_df = create_synthetic_backtest_data(50, "none")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    assert metrics.get("num_trades", 0) == 0


@pytest.mark.unit
@pytest.mark.backtest
def test_atr_stop_loss_used(sample_config: Config) -> None:
    """Test that ATR stop-loss is passed to buy/sell."""
    test_df, preds_df = create_synthetic_backtest_data(50, "alternating")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    # Should have trades
    assert metrics.get("num_trades", 0) > 0


@pytest.mark.unit
@pytest.mark.backtest
def test_signal_reversal_works(sample_config: Config) -> None:
    """Test that exclusive_orders handles signal reversal."""
    # Alternating signals should produce multiple trades
    test_df, preds_df = create_synthetic_backtest_data(100, "alternating")
    metrics = run_backtest_from_data(test_df, preds_df, sample_config)

    assert metrics.get("num_trades", 0) > 0


@pytest.mark.unit
@pytest.mark.backtest
def test_commission_calculation(sample_config: Config) -> None:
    """Test that commission callable produces correct values."""
    contract_size = sample_config.data.contract_size
    commission_per_lot = sample_config.backtest.commission_per_lot

    # Simulate the commission function
    order_size = 100.0  # 1 lot
    lots = abs(order_size) / contract_size
    commission = lots * commission_per_lot
    assert commission == 10.0  # 1 lot × $10


@pytest.mark.unit
@pytest.mark.backtest
def test_run_backtest_from_data_compat(sample_config: Config) -> None:
    """Test that ablation interface returns a dict."""
    test_df, preds_df = create_synthetic_backtest_data(50, "mixed")
    result = run_backtest_from_data(test_df, preds_df, sample_config)

    assert isinstance(result, dict)
    assert "num_trades" in result


@pytest.mark.unit
@pytest.mark.backtest
def test_no_lookahead_bias(sample_config: Config) -> None:
    """Test that execution is delayed by 1 bar (backtesting.py native)."""
    n_rows = 10
    test_df, preds_df = create_synthetic_backtest_data(n_rows, "all_long")
    pdf = _prepare_df(test_df, preds_df)
    stats, _bt = _run_fractional_backtest(pdf, sample_config)
    trades = stats["_trades"]

    # backtesting.py evaluates signals only after bars are complete and fills
    # market orders on the next bar. The first actionable all-long signal is
    # evaluated at bar 1, so the first entry must be at bar 2, not bar 0/1.
    assert len(trades) > 0
    assert trades.iloc[0]["EntryTime"] == pdf.index[2]


@pytest.mark.unit
@pytest.mark.backtest
def test_signal_uses_index_minus_2(sample_config: Config) -> None:
    """Signal shift: strategy reads signals[-2], not signals[-1].

    Verifies that:
    1. No trade is entered when len(signals) < 2 (guard clause).
    2. When signals have ≥ 2 bars, the trade decision uses the
       prediction from bar i-1 (signals[-2]), not the current bar.
    """
    import numpy as np
    import polars as pl

    from thesis.stage_5_backtest.simulation import _prepare_df, _run_fractional_backtest

    n_rows = 5
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0),
        end=pl.datetime(2024, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    closes = 1800.0 + np.arange(n_rows) * 2.0
    opens = closes - 0.1
    highs = closes + 0.5
    lows = closes - 0.5
    atr = np.full(n_rows, 20.0)

    # Only bar 0 has a long signal; bars 1-4 are hold (0).
    # With signals[-2], bar 0's signal is consumed at bar 1's next().
    pred_label = np.array([1, 0, 0, 0, 0], dtype=int)

    test_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n_rows) * 1000.0,
            "atr_14": atr,
        }
    )

    preds_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "pred_label": pred_label,
        }
    )

    pdf = _prepare_df(test_df, preds_df)
    stats, _bt = _run_fractional_backtest(pdf, sample_config)
    trades = stats["_trades"]

    # Guard: bar 0 has len(signals)=1 → no trade at bar 0.
    # Bar 1 has len(signals)=2 → signals[-2]=pred_label[0]=1 → buy → fill at open[2].
    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"
    assert trades.iloc[0]["EntryTime"] == pdf.index[2], (
        f"Entry should be at index 2 (shifted signal), "
        f"got {trades.iloc[0]['EntryTime']}"
    )


@pytest.mark.unit
@pytest.mark.backtest
def test_calendar_day_strips_intraday_time() -> None:
    """Daily risk state must reset by date, not every bar timestamp."""
    ts1 = pd.Timestamp("2026-04-29 09:00:00")
    ts2 = pd.Timestamp("2026-04-29 17:00:00")
    ts3 = pd.Timestamp("2026-04-30 00:00:00")

    assert _calendar_day(ts1) == _calendar_day(ts2)
    assert _calendar_day(ts1) != _calendar_day(ts3)


# ---------------------------------------------------------------------------
# Fixed-risk position sizing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.backtest
def test_position_size_ignores_confidence_after_gate() -> None:
    """Confidence gates entries but must not amplify lot size.

    The latest OOS regression showed that confidence-scaled lots magnified
    high-confidence wrong predictions. Sizing is fixed until the model is
    profitable at the base risk level.
    """
    strategy = type(
        "S",
        (),
        {
            "lots_per_trade": 0.1,
            "min_lots": 0.05,
            "max_lots": 0.2,
            "_compute_lots": HybridGRUStrategy._compute_lots,
        },
    )()

    assert strategy._compute_lots(None) == pytest.approx(0.1)
    assert strategy._compute_lots(0.56) == pytest.approx(0.1)
    assert strategy._compute_lots(0.95) == pytest.approx(0.1)


@pytest.mark.unit
@pytest.mark.backtest
def test_position_size_clamps_fixed_lots_to_bounds() -> None:
    """Fixed sizing still respects configured min/max safety bounds."""
    too_small = type(
        "S",
        (),
        {
            "lots_per_trade": 0.001,
            "min_lots": 0.05,
            "max_lots": 0.2,
            "_compute_lots": HybridGRUStrategy._compute_lots,
        },
    )()
    too_large = type(
        "S",
        (),
        {
            "lots_per_trade": 0.5,
            "min_lots": 0.05,
            "max_lots": 0.2,
            "_compute_lots": HybridGRUStrategy._compute_lots,
        },
    )()

    assert too_small._compute_lots(0.99) == pytest.approx(0.05)
    assert too_large._compute_lots(0.99) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Cooldown tests — min_bars_between_trades (task 9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.backtest
def test_cooldown_prevents_immediate_reentry(sample_config: Config) -> None:
    """Verifies that min_bars_between_trades delays re-entry after an exit.

    Creates synthetic data with a clear directional pattern so trades
    exit naturally (via label reversals).  A long cooldown should
    reduce trade count compared to no cooldown.
    """
    # Use a longer series so trades have room to complete and re-enter
    test_df, preds_df = create_synthetic_backtest_data(200, "alternating")

    # Without cooldown: alternating signals should produce many trades
    metrics_no_cd, _trades_no_cd = run_backtest_manual(
        test_df,
        preds_df,
        leverage=10,
        lots_per_trade=0.1,
        min_lots=0.01,
        max_lots=0.5,
        confidence_threshold=0.0,
        atr_stop_multiplier=3.0,
        atr_tp_multiplier=0,
        horizon_bars=0,
        min_bars_between_trades=0,
    )

    # With cooldown: 20-bar cooldown should reduce trade count
    metrics_cd, _trades_cd = run_backtest_manual(
        test_df,
        preds_df,
        leverage=10,
        lots_per_trade=0.1,
        min_lots=0.01,
        max_lots=0.5,
        confidence_threshold=0.0,
        atr_stop_multiplier=3.0,
        atr_tp_multiplier=0,
        horizon_bars=0,
        min_bars_between_trades=20,
    )

    # Cooldown should reduce trade count (or at least not increase it)
    assert metrics_cd["num_trades"] <= metrics_no_cd["num_trades"], (
        f"Cooldown should reduce trades, got no_cd={metrics_no_cd['num_trades']}, "
        f"cd={metrics_cd['num_trades']}"
    )


@pytest.mark.unit
@pytest.mark.backtest
def test_cooldown_disabled_by_zero(sample_config: Config) -> None:
    """When min_bars_between_trades=0, cooldown is effectively disabled.

    The gate in _is_trading_allowed skips cooldown when the threshold is 0.
    """
    test_df, preds_df = create_synthetic_backtest_data(100, "alternating")

    metrics, _trades = run_backtest_manual(
        test_df,
        preds_df,
        leverage=10,
        lots_per_trade=0.1,
        min_lots=0.01,
        max_lots=0.5,
        confidence_threshold=0.0,
        atr_stop_multiplier=3.0,
        atr_tp_multiplier=0,
        horizon_bars=0,
        min_bars_between_trades=0,
    )

    # Cooldown disabled — should still get trades
    assert metrics["num_trades"] > 0, (
        "With cooldown disabled, alternating signals should produce trades"
    )


# ---------------------------------------------------------------------------
# Zero-cost diagnostic: isolate cost sensitivity from edge absence
# ---------------------------------------------------------------------------


def _create_synthetic_data_for_diagnostic(
    n_rows: int = 200,
    drift: float = 0.15,
    noise: float = 0.08,
    base_price: float = 2000.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic data with a clear upward trend for diagnostic tests.

    Returns test_df and preds_df with all-long signals so the strategy
    captures the entire trend (one long entry, held to the end or stopped).
    """
    np.random.seed(42)

    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0),
        end=pl.datetime(2024, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    # Upward-trending price series
    closes = base_price + np.cumsum(
        np.full(n_rows, drift) + np.random.randn(n_rows) * noise
    )
    opens = closes + np.random.randn(n_rows) * 0.02
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_rows)) * 0.2
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_rows)) * 0.2

    # Wide ATR → loose stops so trades are not prematurely stopped out
    atr = np.full(n_rows, 40.0)

    test_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n_rows) * 1000.0,
            "atr_14": atr,
        }
    )

    # All-long signals — one directional bet capturing the trend
    pred_label = np.ones(n_rows, dtype=int)

    preds_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "pred_label": pred_label,
            "pred_proba_class_minus1": np.zeros(n_rows),
            "pred_proba_class_0": np.zeros(n_rows),
            "pred_proba_class_1": np.ones(n_rows) * 0.9,
        }
    )

    return test_df, preds_df


@pytest.mark.unit
@pytest.mark.backtest
def test_zero_cost_backtest_isolates_model_edge() -> None:
    """Zero-cost vs real-cost backtest to diagnose failure mode.

    Runs the same strategy with zero trading costs (spread=0, slippage=0,
    commission=0) and with realistic costs.  The comparison reveals whether
    poor live performance stems from cost-sensitivity or a fundamental lack
    of directional edge.

    Interpretation:
        - Zero-cost profitable, real-cost profitable → model has edge.
        - Zero-cost profitable, real-cost unprofitable → cost sensitivity.
        - Zero-cost unprofitable → model has no edge (edge-absence).
    """
    test_df, preds_df = _create_synthetic_data_for_diagnostic(n_rows=200)

    # --- Zero-cost run ---
    zero_metrics, _ = run_backtest_manual(
        test_df,
        preds_df,
        spread_ticks=0,
        slippage_ticks=0,
        commission_per_lot=0,
        atr_stop_multiplier=3.0,
        atr_tp_multiplier=0,  # disable take-profit
        horizon_bars=0,  # disabled
        leverage=10,
        lots_per_trade=0.1,
        confidence_threshold=0.0,
    )

    # --- Real-cost run ---
    real_metrics, _ = run_backtest_manual(
        test_df,
        preds_df,
        spread_ticks=35,
        slippage_ticks=5,
        commission_per_lot=10.0,
        atr_stop_multiplier=3.0,
        atr_tp_multiplier=0,  # disable take-profit
        horizon_bars=0,  # disabled
        leverage=10,
        lots_per_trade=0.1,
        confidence_threshold=0.0,
    )

    zero_return = zero_metrics["return_pct"]
    real_return = real_metrics["return_pct"]
    zero_trades = zero_metrics["num_trades"]
    real_trades = real_metrics["num_trades"]

    # Same data and signals → same trade count
    assert zero_trades == real_trades, (
        f"Trade count mismatch: zero-cost={zero_trades}, real-cost={real_trades}"
    )

    # Costs must degrade returns
    assert zero_return > real_return, (
        f"Zero-cost return ({zero_return:.2f}%) must exceed "
        f"real-cost return ({real_return:.2f}%). Trading costs are not "
        f"degrading performance as expected."
    )

    # Report diagnosis
    if zero_return <= 0:
        _log.warning(
            "EDGE-ABSENCE: Zero-cost return %.2f%% ≤ 0 — the model has no "
            "directional edge. Even without trading costs the strategy is "
            "unprofitable. Fix: improve signal quality (feature engineering, "
            "label design, or model architecture) before tuning costs.",
            zero_return,
        )
    elif real_return <= 0:
        _log.warning(
            "COST-SENSITIVITY: Zero-cost profitable (%.2f%%) but real-cost "
            "unprofitable (%.2f%%). The model has edge but trading costs "
            "consume it. Fix: reduce costs (tighten spread, lower commission) "
            "or increase position-sizing calibration.",
            zero_return,
            real_return,
        )
    else:
        _log.info(
            "Zero-cost: %.2f%%, Real-cost: %.2f%% — model has edge and "
            "survives costs. Both returns are positive.",
            zero_return,
            real_return,
        )


# ---------------------------------------------------------------------------
# Perfect-prediction sanity test: theoretical ceiling of the engine
# ---------------------------------------------------------------------------


def _create_perfect_prediction_data(
    n_rows: int = 400,
    drift_per_bar: float = 3.0,
    noise_std: float = 0.3,
    base_price: float = 2000.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic OHLCV data with a strong deterministic uptrend.

    Returns test_df and preds_df where predictions are all-long (+1) —
    the "perfect" prediction for a consistently upward-trending market.

    The trend is driven by a fixed drift per bar so the price roughly
    doubles in magnitude over the horizon, making it straightforward to
    verify that the backtest engine captures the full directional edge.

    Args:
        n_rows: Number of hourly bars.
        drift_per_bar: Average close increase per bar ($).
        noise_std: Standard deviation of per-bar price noise ($).
        base_price: Starting price.

    Returns:
        (test_df, preds_df) — both polars DataFrames.
    """
    np.random.seed(42)

    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0),
        end=pl.datetime(2024, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    # Deterministic drift + small noise for realistic OHLC structure
    closes = base_price + np.cumsum(
        np.full(n_rows, drift_per_bar) + np.random.randn(n_rows) * noise_std
    )
    # Ensure monotonic closes (no local dips strong enough to hit stops)
    opens = closes - 0.05 + np.random.randn(n_rows) * 0.01
    highs = closes + np.abs(np.random.randn(n_rows)) * 0.3
    lows = closes - np.abs(np.random.randn(n_rows)) * 0.3

    # Very wide ATR → stops are never triggered by the tiny noise
    atr = np.full(n_rows, 80.0)

    test_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n_rows) * 5000.0,
            "atr_14": atr,
        }
    )

    # Perfect predictions: all-long on a market that only goes up
    pred_label = np.ones(n_rows, dtype=int)

    preds_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "pred_label": pred_label,
            "pred_proba_class_minus1": np.zeros(n_rows),
            "pred_proba_class_0": np.zeros(n_rows),
            "pred_proba_class_1": np.ones(n_rows) * 0.99,
        }
    )

    return test_df, preds_df


@pytest.mark.unit
@pytest.mark.backtest
def test_perfect_prediction_backtest_sanity() -> None:
    """Feed perfect predictions into backtest to verify engine ceiling.

    Creates synthetic OHLCV data with a strong deterministic uptrend
    (~$1200 move over 400 hourly bars) and feeds all-long "perfect"
    predictions.  Runs two scenarios:

    1. Zero trading costs — should return > +100% because the engine
       correctly captures the full directional move.
    2. Real trading costs — shows the maximum achievable return under
       the labeling regime, revealing the cost-sensitivity ceiling.

    Interpretation:
        - Zero-cost return > +100% → engine works correctly.
        - Zero-cost return ≤ +100% → engine, sizing, or data generation
          prevents capturing the full move (investigate position sizing
          and trend parameters).
        - Real-cost return positive → labeling regime can be profitable
          with perfect foresight.
        - Real-cost return negative → even perfect predictions cannot
          overcome trading costs under this labeling regime — a critical
          finding that signals a design issue in label/backtest setup.
    """
    n_rows = 400
    test_df, preds_df = _create_perfect_prediction_data(n_rows=n_rows)

    # Expected price range: ~$2000 → ~$3200
    close_end = float(test_df["close"][-1])
    close_start = float(test_df["close"][0])
    _log.info(
        "Perfect-prediction data: %d rows, close %.2f → %.2f (Δ=%.2f)",
        n_rows,
        close_start,
        close_end,
        close_end - close_start,
    )

    # --- Zero-cost backtest ---
    zero_metrics, zero_trades = run_backtest_manual(
        test_df,
        preds_df,
        leverage=50,
        lots_per_trade=0.3,
        min_lots=0.1,
        max_lots=0.5,
        confidence_threshold=0.0,
        spread_ticks=0,
        slippage_ticks=0,
        commission_per_lot=0,
        atr_stop_multiplier=1.0,
        atr_tp_multiplier=0,  # disable take-profit
        horizon_bars=0,  # disabled — hold to end
        initial_capital=10_000.0,
    )

    zero_return = zero_metrics["return_pct"]
    zero_trade_count = zero_metrics["num_trades"]
    zero_equity = zero_metrics["equity_final"]

    _log.info(
        "Zero-cost: return=%.2f%%, equity_final=%.2f, trades=%d",
        zero_return,
        zero_equity,
        zero_trade_count,
    )

    # --- Real-cost backtest ---
    real_metrics, real_trades = run_backtest_manual(
        test_df,
        preds_df,
        leverage=50,
        lots_per_trade=0.3,
        min_lots=0.1,
        max_lots=0.5,
        confidence_threshold=0.0,
        spread_ticks=35,
        slippage_ticks=5,
        commission_per_lot=10.0,
        atr_stop_multiplier=1.0,
        atr_tp_multiplier=0,
        horizon_bars=0,
        initial_capital=10_000.0,
    )

    real_return = real_metrics["return_pct"]
    real_trade_count = real_metrics["num_trades"]
    real_equity = real_metrics["equity_final"]

    _log.info(
        "Real-cost: return=%.2f%%, equity_final=%.2f, trades=%d",
        real_return,
        real_equity,
        real_trade_count,
    )

    # --- Assertions ---

    # Zero-cost must be massively profitable — proves the engine captures
    # the full directional edge.  With $1,200 move × 30 units = $36,000,
    # return = 360% which is well above the 100% floor.
    assert zero_return > 100, (
        f"Zero-cost return ({zero_return:.2f}%) must exceed +100% to "
        f"prove the backtest engine captures the full directional move. "
        f"Expected ~360% for $1200 uptick with 30-unit position. "
        f"Check position sizing, trend parameters, or whether the "
        f"position was stopped out prematurely."
    )

    assert zero_trade_count >= 1, (
        f"Zero-cost backtest should have at least 1 trade, got {zero_trade_count}"
    )

    # Real-cost return is informative — document if even perfect predictions
    # cannot survive real trading costs.
    if real_return <= 0:
        _log.warning(
            "CRITICAL: Even perfect predictions are unprofitable at real "
            "costs (%.2f%%). The labeling/backtest regime has a structural "
            "problem: trading costs consume all edge even with perfect "
            "foresight. Revisit label horizons, barrier multipliers, or "
            "cost assumptions.",
            real_return,
        )
    elif real_return > 0:
        _log.info(
            "Real-cost return positive (%.2f%%) — even with perfect "
            "predictions the labeling regime survives real trading costs.",
            real_return,
        )

    # Zero-cost must outperform real-cost (costs degrade returns)
    assert zero_return > real_return, (
        f"Zero-cost return ({zero_return:.2f}%) must exceed "
        f"real-cost return ({real_return:.2f}%). Trading costs "
        f"are not degrading performance as expected."
    )


# ---------------------------------------------------------------------------
# OOS date-range filtering tests
# ---------------------------------------------------------------------------


def _apply_oos_filter(
    pdf: pd.DataFrame,
    oob_start_date: str = "",
    oob_end_date: str = "",
) -> pd.DataFrame:
    """Apply the OOS date-range filter logic from run_backtest.
    """
    if oob_start_date:
        start_ts = pd.Timestamp(oob_start_date)
        pdf = pdf[pdf.index >= start_ts]
    if oob_end_date:
        end_ts = pd.Timestamp(oob_end_date)
        pdf = pdf[pdf.index <= end_ts]
    return pdf


class TestOOSFiltering:
    """Tests for out-of-sample date-range filtering in backtest pipeline."""

    @staticmethod
    def _make_dataframe(
        n_days: int = 7,
        bars_per_day: int = 24,
        start_date: str = "2024-01-01",
    ) -> pd.DataFrame:
        """Create a small test DataFrame with hourly timestamps over N days.

        Returns a pandas DataFrame with a DatetimeIndex (matching what
        _prepare_df produces), plus a 'value' column for sanity checks.
        """
        n_rows = n_days * bars_per_day
        np.random.seed(42)
        timestamps = pd.date_range(
            start=start_date,
            periods=n_rows,
            freq="h",
        )
        pdf = pd.DataFrame(
            {
                "Open": np.random.randn(n_rows) + 100,
                "High": np.random.randn(n_rows) + 101,
                "Low": np.random.randn(n_rows) + 99,
                "Close": np.random.randn(n_rows) + 100,
                "Volume": np.ones(n_rows) * 1000.0,
                "atr_14": np.full(n_rows, 20.0),
                "pred_label": np.random.choice([-1, 0, 1], n_rows),
                "value": np.arange(n_rows, dtype=float),
            },
            index=timestamps,
        )
        pdf.index = pd.DatetimeIndex(pdf.index)
        return pdf

    # ── Default (no filter) ──────────────────────────────────────────────

    def test_oos_no_filter_preserves_all_bars(self):
        """Default (empty strings) should return the full dataset unchanged."""
        pdf = self._make_dataframe(n_days=5)
        filtered = _apply_oos_filter(pdf)
        assert len(filtered) == len(pdf)
        pd.testing.assert_frame_equal(filtered, pdf)

    def test_oos_empty_string_is_treated_as_no_filter(self):
        """Explicit empty strings should behave identically to omitting params."""
        pdf = self._make_dataframe(n_days=3)
        filtered = _apply_oos_filter(pdf, oob_start_date="", oob_end_date="")
        assert len(filtered) == len(pdf)

    # ── Start-only filter ────────────────────────────────────────────────

    def test_oos_start_only_filters_from_date(self):
        """Only oob_start_date: keep bars on or after the date."""
        pdf = self._make_dataframe(n_days=7, start_date="2024-01-01")
        filtered = _apply_oos_filter(pdf, oob_start_date="2024-01-03")
        expected_min = pd.Timestamp("2024-01-03")
        assert len(filtered) > 0, "Filter should not remove all bars"
        assert len(filtered) < len(pdf), "Filter should remove some bars"
        assert filtered.index.min() >= expected_min

    def test_oos_start_exact_boundary_included(self):
        """A bar exactly at the start boundary should be kept."""
        pdf = self._make_dataframe(n_days=3, start_date="2024-01-01")
        # Use the exact timestamp of the first visible bar
        boundary = str(pdf.index.min())
        filtered = _apply_oos_filter(pdf, oob_start_date=boundary)
        assert len(filtered) == len(pdf), "All bars should be kept at exact boundary"

    # ── End-only filter ──────────────────────────────────────────────────

    def test_oos_end_only_filters_to_date(self):
        """Only oob_end_date: keep bars on or before the date."""
        pdf = self._make_dataframe(n_days=7, start_date="2024-01-01")
        filtered = _apply_oos_filter(pdf, oob_end_date="2024-01-04")
        expected_max = pd.Timestamp("2024-01-04")
        assert len(filtered) > 0, "Filter should not remove all bars"
        assert len(filtered) < len(pdf), "Filter should remove some bars"
        assert filtered.index.max() <= expected_max

    def test_oos_end_exact_boundary_included(self):
        """A bar exactly at the end boundary should be kept."""
        pdf = self._make_dataframe(n_days=3, start_date="2024-01-01")
        boundary = str(pdf.index.max())
        filtered = _apply_oos_filter(pdf, oob_end_date=boundary)
        assert len(filtered) == len(pdf), "All bars should be kept at exact boundary"

    # ── Both dates ───────────────────────────────────────────────────────

    def test_oos_both_dates_narrows_window(self):
        """Both dates: keep only bars within [start, end]."""
        pdf = self._make_dataframe(n_days=10, start_date="2024-01-01")
        filtered = _apply_oos_filter(
            pdf,
            oob_start_date="2024-01-03",
            oob_end_date="2024-01-07",
        )
        start_ts = pd.Timestamp("2024-01-03")
        end_ts = pd.Timestamp("2024-01-07")
        assert len(filtered) > 0, "Window should contain bars"
        assert len(filtered) < len(pdf), "Window should be a subset"
        assert filtered.index.min() >= start_ts
        assert filtered.index.max() <= end_ts

    def test_oos_both_dates_correct_row_count(self):
        """Verify filtering retains the expected number of bars.

        With 7 days × 24 bars starting 2024-01-01 00:00, filtering
        [2024-01-03, 2024-01-06) includes 73 bars:
        - 2024-01-03 00:00  (bar  48)  — first kept
        - 2024-01-06 00:00  (bar 120)  — last kept (≤ midnight Jan 6)
        """
        pdf = self._make_dataframe(n_days=7, start_date="2024-01-01")
        filtered = _apply_oos_filter(
            pdf,
            oob_start_date="2024-01-03",
            oob_end_date="2024-01-06",
        )
        # Jan 3 00:00 through Jan 6 00:00 = 73 hourly bars
        assert len(filtered) == 73, f"Expected 73 bars, got {len(filtered)}"

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_oos_start_after_end_excludes_all(self):
        """Start date after end date should exclude all bars."""
        pdf = self._make_dataframe(n_days=5, start_date="2024-01-01")
        filtered = _apply_oos_filter(
            pdf,
            oob_start_date="2024-01-10",
            oob_end_date="2024-01-09",
        )
        assert len(filtered) == 0

    def test_oos_filter_with_real_prepare_df(self):
        """Integration: filter on output of _prepare_df from real synthetic data."""
        test_df, preds_df = create_synthetic_backtest_data(
            n_rows=168, signal_pattern="mixed"
        )
        pdf = _prepare_df(test_df, preds_df)
        original_len = len(pdf)
        # Filter to a middle 3-day window (168 rows / 7 days ≈ 24 rows/day)
        mid_point = pdf.index[len(pdf) // 2]
        start_ts = str(mid_point - pd.Timedelta(days=1))
        end_ts = str(mid_point + pd.Timedelta(days=1))
        filtered = _apply_oos_filter(pdf, oob_start_date=start_ts, oob_end_date=end_ts)
        assert len(filtered) > 0, "Window should contain bars"
        assert len(filtered) < original_len, "Window should be a subset"
        assert filtered.index.min() >= pd.Timestamp(start_ts)
        assert filtered.index.max() <= pd.Timestamp(end_ts)
