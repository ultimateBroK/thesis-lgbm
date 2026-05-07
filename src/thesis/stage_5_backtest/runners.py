"""Stage 5 backtest helpers (prep + runner)."""

from __future__ import annotations

from collections.abc import Callable
import logging

from backtesting.lib import FractionalBacktest
import pandas as pd
import polars as pl

from thesis.shared.config import Config
from thesis.stage_5_backtest.strategy import HybridGRUStrategy

logger = logging.getLogger("thesis.backtest")


def _validate_backtest_merge(
    *,
    feature_rows: int,
    prediction_rows: int,
    merged_rows: int,
    test_source: str = "<in-memory test/features>",
    preds_source: str = "<in-memory predictions>",
) -> None:
    """Guard against silent timestamp loss in the backtest inner join."""
    coverage = merged_rows / prediction_rows if prediction_rows else 0.0
    dropped = prediction_rows - merged_rows
    logger.info(
        "Backtest merge: features_rows=%d predictions_rows=%d merged_rows=%d "
        "coverage=%.2f%% dropped_predictions=%d",
        feature_rows,
        prediction_rows,
        merged_rows,
        coverage * 100.0,
        dropped,
    )
    if coverage < 0.99:
        raise ValueError(
            "Backtest merge coverage below 99%: "
            f"expected>=99.00%, actual={coverage * 100.0:.2f}%, "
            f"features_rows={feature_rows}, predictions_rows={prediction_rows}, "
            f"merged_rows={merged_rows}, dropped_predictions={dropped}, "
            f"features_path={test_source}, predictions_path={preds_source}. "
            "Check timestamp alignment before backtesting."
        )


def _prepare_df(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    *,
    test_source: str = "<in-memory test/features>",
    preds_source: str = "<in-memory predictions>",
) -> pd.DataFrame:
    """Join features and predictions on ``timestamp``; OHLCV → Open/High/.../Volume."""
    test = test_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    preds = preds_df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    if "pred_label" not in preds.columns:
        raise ValueError("Predictions must contain 'pred_label' column")

    pred_cols = ["timestamp", "pred_label"]
    for col in [
        "pred_proba_class_minus1",
        "pred_proba_class_0",
        "pred_proba_class_1",
    ]:
        if col in preds.columns:
            pred_cols.append(col)

    merged = test.join(preds.select(pred_cols), on="timestamp", how="inner")
    _validate_backtest_merge(
        feature_rows=len(test),
        prediction_rows=len(preds),
        merged_rows=len(merged),
        test_source=test_source,
        preds_source=preds_source,
    )

    if "atr_14" not in merged.columns:
        raise ValueError(
            "atr_14 column not found in test data. "
            "Ensure feature engineering includes ATR before backtest."
        )

    logger.info("Backtest bars: %d", len(merged))

    pdf = merged.to_pandas()
    pdf = pdf.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    if "Volume" not in pdf.columns:
        pdf["Volume"] = 0

    pdf = pdf.set_index("timestamp")
    pdf.index = pd.DatetimeIndex(pdf.index)

    return pdf


def _compute_spread_rate(
    bc: Config.BacktestConfig,
    dc: Config.DataConfig,
    median_price: float,
) -> float:
    """Convert tick-based spread + slippage to relative rate for backtesting.py."""
    total_ticks = bc.spread_ticks + bc.slippage_ticks
    return total_ticks * dc.tick_size / median_price


def _make_commission_fn(
    commission_per_lot: float,
    contract_size: float,
) -> Callable[[float, float], float]:
    """Build a commission function closure for backtesting.py."""

    def commission_fn(order_size: float, price: float) -> float:  # noqa: ARG001
        lots = abs(order_size) / contract_size
        return lots * commission_per_lot

    return commission_fn


def _create_fractional_backtest(
    pdf: pd.DataFrame,
    *,
    cash: float,
    spread: float,
    commission_fn: Callable[[float, float], float],
    leverage: float | int,
) -> FractionalBacktest:
    """Configure ``FractionalBacktest`` without running it."""
    return FractionalBacktest(
        pdf,
        HybridGRUStrategy,
        cash=cash,
        spread=spread,
        commission=commission_fn,
        margin=1.0 / float(leverage),
        exclusive_orders=True,
        finalize_trades=True,
        fractional_unit=1.0,
    )


def _run_fractional_backtest(
    pdf: pd.DataFrame,
    config: Config,
) -> tuple[pd.Series, FractionalBacktest]:
    """Run the configured FractionalBacktest and return (stats, bt)."""
    bc = config.backtest
    dc = config.data

    median_price = float(pdf["Close"].median())
    spread = _compute_spread_rate(bc, dc, median_price)
    commission_fn = _make_commission_fn(bc.commission_per_lot, dc.contract_size)
    bt = _create_fractional_backtest(
        pdf,
        cash=bc.initial_capital,
        spread=spread,
        commission_fn=commission_fn,
        leverage=bc.leverage,
    )

    stats = bt.run(
        atr_stop_mult=bc.atr_stop_multiplier,
        atr_tp_mult=bc.atr_tp_multiplier,
        lots_per_trade=bc.lots_per_trade,
        min_lots=bc.min_lots,
        max_lots=bc.max_lots,
        confidence_threshold=bc.confidence_threshold,
        contract_size=dc.contract_size,
        horizon_bars=config.labels.horizon_bars,
        max_drawdown_cutoff=bc.max_drawdown_cutoff,
        dd_cooldown_bars=bc.dd_cooldown_bars,
        max_open_positions=bc.max_open_positions,
        daily_loss_limit=bc.daily_loss_limit,
        min_bars_between_trades=bc.min_bars_between_trades,
    )
    return stats, bt
