"""CFD backtest simulation via backtesting.py.

Keep SL/TP ATR multipliers aligned with the label barriers (same ATR
multiple), otherwise the model is trained on a different risk envelope
than the backtest executes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from thesis.shared.config import Config
from thesis.shared.ui import console
from thesis.stage_5_backtest.persistence import (
    _log_core_backtest_metrics,
    _normalize_stats,
    _save_bokeh_chart,
    _save_equity_curve_csv,
    _save_json_results,
    _save_trade_details_csv,
    _trades_to_list,
)
from thesis.stage_5_backtest.runners import (
    _create_fractional_backtest,
    _make_commission_fn,
    _prepare_df,
    _run_fractional_backtest,
)
from thesis.stage_5_backtest.strategy import _DEFAULT_INITIAL_CAPITAL

logger = logging.getLogger("thesis.backtest")


def run_backtest(config: Config) -> None:
    """Run a full CFD backtest from files specified in config.

    For walk-forward (sliding) validation, joins OOF predictions with the
    full labeled dataset (which contains OHLCV + features). For static
    validation, uses the traditional test split file.

    Writes normalized metrics and trade records as JSON, optional trade-detail
    and equity-curve CSV files, and an optional Bokeh HTML chart.
    """
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found: {preds_path}")
    with console.status(f"[cyan]Loading predictions[/] {preds_path}"):
        preds_df = pl.read_parquet(preds_path)

    test_path = Path(config.paths.test_data)
    is_static = config.validation.method == "static"

    if test_path.exists() and is_static:
        with console.status(f"[cyan]Loading static test data[/] {test_path}"):
            test_df = pl.read_parquet(test_path)
    elif test_path.exists() and not is_static:
        logger.warning(
            "Static test file found (%s) but workflow is walk-forward "
            "(method='%s') — ignoring stale test_data in favor of OOF predictions",
            test_path,
            config.validation.method,
        )
        labels_path = Path(config.paths.labels)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found ({labels_path})"
                " — needed for walk-forward backtest"
            )
        with console.status(f"[cyan]Loading labels for backtest[/] {labels_path}"):
            test_df = pl.read_parquet(labels_path)
    else:
        labels_path = Path(config.paths.labels)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Neither test data ({test_path}) nor labels ({labels_path}) found"
            )
        logger.info("Walk-forward mode: joining OOF predictions with labeled data")
        with console.status(f"[cyan]Loading labels for backtest[/] {labels_path}"):
            test_df = pl.read_parquet(labels_path)

    pdf = _prepare_df(
        test_df,
        preds_df,
        test_source=str(test_path if test_path.exists() and is_static else labels_path),
        preds_source=str(preds_path),
    )

    bc = config.backtest
    if bc.oob_start_date:
        import pandas as pd

        start_ts = pd.Timestamp(bc.oob_start_date)
        pdf = pdf[pdf.index >= start_ts]
        logger.info("OOS start filter: %s → %d bars", bc.oob_start_date, len(pdf))
    if bc.oob_end_date:
        import pandas as pd

        end_ts = pd.Timestamp(bc.oob_end_date)
        pdf = pdf[pdf.index <= end_ts]
        logger.info("OOS end filter: %s → %d bars", bc.oob_end_date, len(pdf))
    if bc.oob_start_date or bc.oob_end_date:
        logger.info(
            "OOS date range: %s to %s (%d bars)",
            bc.oob_start_date or "start",
            bc.oob_end_date or "end",
            len(pdf),
        )

    logger.info("Confidence threshold: %.2f", config.backtest.confidence_threshold)
    with console.status("[cyan]Running CFD backtest[/]"):
        stats, bt = _run_fractional_backtest(pdf, config)

    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats["_trades"],
        commission_per_lot=config.backtest.commission_per_lot,
        contract_size=config.data.contract_size,
    )

    out_path = Path(config.paths.backtest_results)
    _save_json_results(metrics, trades, out_path)

    if trades:
        _save_trade_details_csv(trades, out_path.parent)
        initial_capital = (
            config.backtest.initial_capital
            if hasattr(config.backtest, "initial_capital")
            else _DEFAULT_INITIAL_CAPITAL
        )
        _save_equity_curve_csv(trades, out_path.parent, initial_capital)

    _log_core_backtest_metrics(metrics, config.backtest.initial_capital)

    session_dir = Path(config.paths.session_dir) if config.paths.session_dir else None
    _save_bokeh_chart(bt, stats, session_dir)


def run_backtest_from_data(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    config: Config,
) -> dict:
    """Run the full backtest pipeline using in-memory Polars DataFrames.

    Args:
        test_df: Market/test data containing price columns and atr_14.
        preds_df: Predictions with timestamp and pred_label
            (optional pred_proba_* columns allowed).
        config: Configuration object with backtest, data, and paths sections.

    Returns:
        Normalized metrics dictionary extracted from the backtest results.
    """
    pdf = _prepare_df(test_df, preds_df)
    stats, _ = _run_fractional_backtest(pdf, config)
    return _normalize_stats(stats)


def run_backtest_manual(
    test_df: pl.DataFrame,
    preds_df: pl.DataFrame,
    *,
    leverage: int = 100,
    lots_per_trade: float = 0.2,
    min_lots: float = 0.1,
    max_lots: float = 0.5,
    confidence_threshold: float = 0.0,
    spread_ticks: int = 35,
    slippage_ticks: int = 5,
    commission_per_lot: float = 10.0,
    atr_stop_multiplier: float = 1.0,
    atr_tp_multiplier: float = 2.0,
    horizon_bars: int = 10,
    contract_size: int = 100,
    tick_size: float = 0.01,
    initial_capital: float = 10_000.0,
    max_drawdown_cutoff: float = 0.50,
    dd_cooldown_bars: int = 12,
    max_open_positions: int = 1,
    daily_loss_limit: float = 0.03,
    min_bars_between_trades: int = 6,
) -> tuple[dict, list[dict]]:
    """Run a backtest with manually specified parameters (no Config required).

    Designed for interactive use in dashboards where parameters can be tuned
    without modifying the config file.

    Returns:
        Tuple of (metrics dict, trades list).
    """
    pdf = _prepare_df(test_df, preds_df)

    median_price = float(pdf["Close"].median())
    spread_total = (spread_ticks + slippage_ticks) * tick_size / median_price

    commission_fn = _make_commission_fn(commission_per_lot, contract_size)
    bt = _create_fractional_backtest(
        pdf,
        cash=initial_capital,
        spread=spread_total,
        commission_fn=commission_fn,
        leverage=leverage,
    )

    stats = bt.run(
        atr_stop_mult=atr_stop_multiplier,
        atr_tp_mult=atr_tp_multiplier,
        lots_per_trade=lots_per_trade,
        min_lots=min_lots,
        max_lots=max_lots,
        confidence_threshold=confidence_threshold,
        contract_size=contract_size,
        horizon_bars=horizon_bars,
        max_drawdown_cutoff=max_drawdown_cutoff,
        dd_cooldown_bars=dd_cooldown_bars,
        max_open_positions=max_open_positions,
        daily_loss_limit=daily_loss_limit,
        min_bars_between_trades=min_bars_between_trades,
    )

    metrics = _normalize_stats(stats)
    trades = _trades_to_list(
        stats["_trades"],
        commission_per_lot=commission_per_lot,
        contract_size=contract_size,
    )

    return metrics, trades
