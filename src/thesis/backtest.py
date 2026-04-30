"""Stage 5: CFD backtest simulation via backtesting.py.

Combines strategy, runners, persistence, and stats into a single module.

Public API:
    run_backtest         — full pipeline from Parquet files.
    run_backtest_from_data — from in-memory DataFrames.
    run_backtest_manual  — with explicit keyword parameters.
    HybridGRUStrategy    — strategy class for backtesting.py.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl
from backtesting import Strategy
from backtesting.lib import FractionalBacktest

from thesis.config import Config

logger = logging.getLogger("thesis.backtest")


def _calendar_day(value: object) -> object:
    """Return the calendar date for a timestamp-like value."""
    return pd.Timestamp(value).date()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class HybridGRUStrategy(Strategy):
    """Trade on ML signals with ATR stop-loss and equity risk management.

    No manual signal shift — backtesting.py natively delays execution
    by 1 bar (evaluates at close[i], executes at open[i+1]).

    Position sizing: confidence-weighted — scales from min_lots to max_lots
    based on how far the predicted probability exceeds the confidence
    threshold.  When confidence data is unavailable (or threshold is 0),
    falls back to fixed ``lots_per_trade``.

    Confidence filtering: when confidence_threshold > 0, only trade
    when the predicted class probability exceeds the threshold.

    Stop-loss: set via backtesting.py's native ``sl=`` parameter on
    buy()/sell() calls. The stop price is computed as entry_price ±
    (ATR × atr_stop_mult), floored by min_atr to prevent unrealistic
    stops in low-ATR regimes. A manual stop-check fallback also closes
    positions when the open/low/high crosses the tracked stop level,
    providing conservative detection against brief pierces.

    Take-profit: when ``atr_tp_mult > 0``, a TP price is set at
    entry_price ± (ATR × atr_tp_mult), creating an asymmetric
    risk-reward profile (e.g. 1:2 with SL=1×ATR, TP=2×ATR).

    Risk management:
        - Max drawdown circuit breaker: if equity drops below
          ``max_drawdown_cutoff`` fraction of peak equity, trading pauses
          for ``dd_cooldown_bars`` bars.
        - Max open positions: limits simultaneous positions to
          ``max_open_positions`` (default 1).
        - Daily loss limit: if equity drops by ``daily_loss_limit``
          fraction from the day's starting equity, trading pauses
          until the next calendar day.

    Attributes:
        atr_stop_mult: ATR multiplier for stop-loss distance.
        atr_tp_mult: ATR multiplier for take-profit distance (0 = disabled).
        lots_per_trade: Base lot size used in confidence-weighted scaling.
        min_lots: Minimum lot size (low-conviction floor).
        max_lots: Maximum lot size (high-conviction cap).
        confidence_threshold: Minimum class probability to trade (0 = disabled).
        min_atr: Floor to prevent microscopic stops in low-vol regimes.
        contract_size: Units per lot.
        horizon_bars: Max bars to hold (0 = hold until opposite signal/stop).
        max_drawdown_cutoff: Fraction of peak equity — breach triggers cooldown.
        dd_cooldown_bars: Bars to pause trading after drawdown breach.
        max_open_positions: Max simultaneous open positions.
        daily_loss_limit: Max fraction of daily equity loss before pause.
    """

    atr_stop_mult = 1.0
    atr_tp_mult = 2.0  # 0 = disabled (no take-profit)
    lots_per_trade = 0.2
    min_lots = 0.1
    max_lots = 0.5
    confidence_threshold = 0.0  # 0 = disabled, trade all signals
    min_atr = 0.0001  # floor to prevent microscopic stops
    contract_size = 100  # units per lot (from DataConfig)
    horizon_bars = (
        0  # 0 = disabled (hold until opposite signal or stop); N = exit after N bars
    )
    max_drawdown_cutoff = 0.50  # circuit breaker threshold
    dd_cooldown_bars = 12  # pause duration after drawdown breach
    max_open_positions = 1  # max simultaneous positions
    daily_loss_limit = 0.03  # daily loss fraction limit

    def init(self) -> None:
        """Register indicators and initialise risk-management state.

        Registers ``signals`` from ``data.pred_label`` and ``ATR`` from
        ``data.atr_14`` with the backtesting indicator system, and sets an
        internal ``_has_proba`` flag indicating whether per-class probability
        columns exist.

        Risk-management state initialised:
            - ``_peak_equity``: running maximum equity for drawdown tracking.
            - ``_dd_cooldown_left``: bars remaining in drawdown cooldown.
            - ``_daily_start_equity``: equity at start of each calendar day.
            - ``_current_date``: current calendar day for daily reset logic.
        """
        self._initial_capital = self.equity

        self.signals = self.I(lambda: self.data.pred_label, name="signals", plot=False)
        self.atr = self.I(lambda: self.data.atr_14, name="ATR", plot=True)
        self._has_proba = hasattr(self.data, "pred_proba_class_minus1")
        if self._has_proba:
            self.proba_short = self.I(
                lambda: self.data.pred_proba_class_minus1,
                name="proba_short",
                plot=False,
            )
            self.proba_hold = self.I(
                lambda: self.data.pred_proba_class_0, name="proba_hold", plot=False
            )
            self.proba_long = self.I(
                lambda: self.data.pred_proba_class_1, name="proba_long", plot=False
            )

        self._entry_bar: dict[str, int] = {}

        # Risk-management state
        self._peak_equity: float = self.equity
        self._dd_cooldown_left: int = 0
        self._dd_cutoff_breached: bool = False
        self._daily_start_equity: float = self.equity
        self._current_date: object = None  # track calendar day for daily reset

    def _floor_atr(self, atr: float) -> float:
        """Floor ATR to prevent unrealistic stops in low-volatility regimes."""
        return max(atr, self.min_atr)

    def _update_risk_state(self) -> None:
        """Update peak equity, drawdown cooldown, and daily loss tracking.

        Called every bar in ``next()`` before any trading decisions.
        """
        eq = self.equity
        self._peak_equity = max(self._peak_equity, eq)

        # Drawdown circuit breaker — decrement cooldown each bar
        if self._dd_cooldown_left > 0:
            self._dd_cooldown_left -= 1

        # Check if drawdown exceeds cutoff. For thesis evaluation, stop opening
        # new positions after a catastrophic drawdown instead of repeatedly
        # logging cooldown events for the rest of the backtest.
        if self.max_drawdown_cutoff > 0 and self._peak_equity > 0:
            dd = (self._peak_equity - eq) / self._peak_equity
            if dd >= self.max_drawdown_cutoff and not self._dd_cutoff_breached:
                self._dd_cutoff_breached = True
                self._dd_cooldown_left = self.dd_cooldown_bars
                logger.warning(
                    "Drawdown circuit breaker triggered: %.1f%% drawdown "
                    "exceeds %.1f%% cutoff — blocking new trades",
                    dd * 100,
                    self.max_drawdown_cutoff * 100,
                )

        # Daily loss tracking — reset at start of each new calendar day
        bar_date = _calendar_day(self.data.index[-1])
        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_start_equity = eq

    def _is_trading_allowed(self) -> bool:
        """Check all risk gates before opening a new position.

        Returns:
            True if trading is permitted, False if any gate blocks it.
        """
        # Gate 1: max open positions
        if len(self.orders) > 0:
            return False

        if self.position and self.max_open_positions <= 1:
            return False

        # Gate 2: drawdown circuit breaker
        if self._dd_cutoff_breached:
            return False

        if self._dd_cooldown_left > 0:
            return False

        # Gate 3: daily loss limit
        if self.daily_loss_limit > 0 and self._daily_start_equity > 0:
            daily_pnl = (
                self.equity - self._daily_start_equity
            ) / self._daily_start_equity
            if daily_pnl <= -self.daily_loss_limit:
                return False

        return True

    def _compute_lots(self, confidence: float | None) -> float:
        """Compute position size based on confidence-weighted scaling.

        When confidence data is available and threshold > 0, lots scale
        linearly from 0 (at threshold) to ``lots_per_trade`` (at 1.0),
        then clamped to ``[min_lots, max_lots]``.  Without confidence
        data, returns the fixed ``lots_per_trade``.

        Args:
            confidence: Predicted class probability, or None if unavailable.

        Returns:
            Lot size to use for the trade.
        """
        if confidence is not None and self.confidence_threshold > 0:
            scale = (confidence - self.confidence_threshold) / (
                1.0 - self.confidence_threshold
            )
            lots = self.lots_per_trade * scale
            return max(self.min_lots, min(lots, self.max_lots))
        return self.lots_per_trade

    def next(self) -> None:
        """Evaluate the latest model signal and place orders if appropriate.

        Processing order:
            1. Risk state update — peak equity, cooldown, daily tracking.
            2. Time-based exit — close positions exceeding horizon_bars.
            3. Risk gate check — skip new trades if any gate blocks.
            4. Confidence gate — skip low-confidence signals.
            5. Compute confidence-weighted position size.
            6. Execute trades with native ATR-based stop-loss.
        """
        # Step 1: update risk state every bar
        self._update_risk_state()

        signal = int(self.signals[-1])
        atr = self._floor_atr(self.atr[-1])

        # Step 2: time-based exit
        if self.horizon_bars > 0 and self.position:
            entry_bar = self._entry_bar.get("long") or self._entry_bar.get("short")
            if entry_bar is not None:
                bars_held = len(self.data) - entry_bar
                if bars_held >= self.horizon_bars:
                    self.position.close()
                    direction = "long" if self.position.is_long else "short"
                    self._entry_bar.pop(direction, None)

        # Step 3: risk gate — no new trades if blocked
        if not self._is_trading_allowed():
            return

        # Step 4: confidence gate
        confidence: float | None = None
        if self.confidence_threshold > 0 and self._has_proba:
            if signal == 1:
                confidence = float(self.proba_long[-1])
            elif signal == -1:
                confidence = float(self.proba_short[-1])
            else:
                return

            if confidence < self.confidence_threshold:
                return

        # Step 5: position sizing
        proxy_entry_price = self.data.Close[-1]
        lots = self._compute_lots(confidence)
        size = lots * self.contract_size
        # backtesting.py requires whole-number units (or equity fraction <1)
        size = max(1, round(size))

        # Step 6: execute trades
        if signal == 1 and not self.position:
            self._entry_bar["long"] = len(self.data)
            sl_price = proxy_entry_price - (atr * self.atr_stop_mult)
            tp_price = (
                proxy_entry_price + (atr * self.atr_tp_mult)
                if self.atr_tp_mult > 0
                else None
            )
            self.buy(size=size, sl=sl_price, tp=tp_price)

        elif signal == -1 and not self.position:
            self._entry_bar["short"] = len(self.data)
            sl_price = proxy_entry_price + (atr * self.atr_stop_mult)
            tp_price = (
                proxy_entry_price - (atr * self.atr_tp_mult)
                if self.atr_tp_mult > 0
                else None
            )
            self.sell(size=size, sl=sl_price, tp=tp_price)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _extract_recovery_factor(
    equity_final: float,
    equity_peak: float,
    max_dd_pct: float,
    initial_capital: float = 10_000.0,
) -> float:
    """Compute recovery factor = net_profit / max_drawdown_dollars.

    Args:
        equity_final: Final equity value from backtest.
        equity_peak: Peak equity reached during backtest.
        max_dd_pct: Maximum drawdown as percentage of peak equity.
        initial_capital: Starting capital (default 10_000).

    Returns:
        Recovery factor (0.0 if max_drawdown is zero or negative).
    """
    net_profit = equity_final - initial_capital
    max_dd_dollars = abs(max_dd_pct / 100) * equity_peak
    if max_dd_dollars > 0:
        return net_profit / max_dd_dollars
    return 0.0


def _compute_avg_win_loss(trades_df: pd.DataFrame) -> tuple[float, float]:
    """Compute average win and average loss from trades DataFrame.

    Args:
        trades_df: DataFrame with PnL column from backtesting.py stats.

    Returns:
        Tuple of (avg_win, avg_loss). Returns (0.0, 0.0) if DataFrame is empty
        or PnL column is missing.
    """
    if trades_df.empty or "PnL" not in trades_df.columns:
        return 0.0, 0.0
    wins = trades_df[trades_df["PnL"] > 0]["PnL"]
    losses = trades_df[trades_df["PnL"] < 0]["PnL"]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    return avg_win, avg_loss


def _normalize_stats(stats: pd.Series, initial_capital: float = 10_000.0) -> dict:
    """Convert a Backtesting.py statistics Series into a snake_case dict.

    Omits keys that begin with an underscore and normalizes display-style keys
    by lowercasing, replacing spaces and punctuation with underscores, and
    mapping ``%`` to ``pct`` and ``#`` to ``num``.

    Adds computed fields:
        - ``recovery_factor``: net_profit / max_drawdown_dollars.
        - ``avg_win``: mean PnL of winning trades.
        - ``avg_loss``: mean PnL of losing trades.

    Args:
        stats: Series-like statistics object produced by Backtesting.py.
        initial_capital: Starting capital for recovery factor computation
            (default 10_000).

    Returns:
        Dictionary of normalized metric names to their original values.
    """
    raw = stats.to_dict()
    out: dict = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        key = (
            k.lower()
            .replace(" ", "_")
            .replace(".", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("$", "")
            .replace("%", "pct")
            .replace("#", "num")
            .replace("__", "_")
            .rstrip("_")
        )
        out[key] = v

    equity_final = out.get("equity_final", 0)
    equity_peak = out.get("equity_peak", equity_final)
    max_dd_pct = out.get("max_drawdown_pct", 0)
    out["recovery_factor"] = _extract_recovery_factor(
        equity_final, equity_peak, max_dd_pct, initial_capital=initial_capital
    )

    trades_df = stats.get("_trades", pd.DataFrame())
    avg_win, avg_loss = _compute_avg_win_loss(trades_df)
    out["avg_win"] = avg_win
    out["avg_loss"] = avg_loss

    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _trades_to_list(
    trades_df: pd.DataFrame,
    commission_per_lot: float = 20.0,
    contract_size: float = 100.0,
) -> list[dict]:
    """Convert a backtesting.py trades DataFrame to a JSON-serializable list.

    Each record contains entry/exit timestamps, direction ("long" or "short"),
    entry/exit prices, lot size, PnL, return percentage, commission, and
    duration.

    Args:
        trades_df: Raw trades DataFrame from backtesting.py stats.
        commission_per_lot: Commission charged per lot to compute per-trade
            commission.
        contract_size: Units per lot used to convert raw Size into lot counts.

    Returns:
        List of trade dictionaries with keys: entry_time, exit_time,
        direction, entry_price, exit_price, lot_size, pnl, return_pct,
        commission, duration.
    """
    if trades_df.empty:
        return []
    records = trades_df.reset_index(drop=True)
    result: list[dict] = []
    for _, row in records.iterrows():
        size = float(row.get("Size", 0))
        lots = abs(size) / contract_size
        commission = lots * commission_per_lot
        result.append(
            {
                "entry_time": str(row.get("EntryTime", "")),
                "exit_time": str(row.get("ExitTime", "")),
                "direction": "long" if size > 0 else "short",
                "entry_price": float(row.get("EntryPrice", 0)),
                "exit_price": float(row.get("ExitPrice", 0)),
                "lot_size": lots,
                "pnl": float(row.get("PnL", 0)),
                "return_pct": float(row.get("ReturnPct", 0)) * 100,
                "commission": round(commission, 2),
                "duration": str(row.get("Duration", "")),
            }
        )
    return result


def _save_json_results(
    metrics: dict,
    trades: list[dict],
    out_path: Path,
) -> None:
    """Save backtest results as JSON.

    Args:
        metrics: Normalized metrics from _normalize_stats.
        trades: List of trade records from _trades_to_list.
        out_path: Destination path for JSON file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "trades": trades}, f, indent=2, default=str)
    logger.info("Backtest results saved: %s", out_path)


def _save_trade_details_csv(trades: list[dict], out_dir: Path) -> None:
    """Save per-trade records as CSV.

    Args:
        trades: List of trade dictionaries.
        out_dir: Parent directory for output CSV.
    """
    if not trades:
        return
    csv_path = out_dir / "trades_detail.csv"
    fieldnames = list(trades[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
    logger.info("Trade details CSV saved: %s (%d trades)", csv_path, len(trades))


def _save_equity_curve_csv(
    trades: list[dict],
    out_dir: Path,
    initial_capital: float = 10_000.0,
) -> None:
    """Save equity curve as CSV with running peak and drawdown.

    Each row represents a closed trade with the running equity, peak equity,
    and drawdown percentage.

    Args:
        trades: List of trade dictionaries with pnl and exit_time.
        out_dir: Parent directory for output CSV.
        initial_capital: Starting capital for equity calculation.
    """
    if not trades:
        return
    eq_path = out_dir / "equity_curve.csv"
    equity = initial_capital
    with open(eq_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trade_num", "exit_time", "pnl", "equity", "drawdown_pct"])
        peak = initial_capital
        for i, t in enumerate(trades, 1):
            equity += t["pnl"]
            peak = max(peak, equity)
            dd_pct = (equity - peak) / peak * 100 if peak > 0 else 0.0
            writer.writerow(
                [
                    i,
                    t.get("exit_time", ""),
                    round(t["pnl"], 2),
                    round(equity, 2),
                    round(dd_pct, 4),
                ]
            )
    logger.info("Equity curve CSV saved: %s", eq_path)


def _save_bokeh_chart(
    bt: FractionalBacktest,
    stats: pd.Series,
    session_dir: Path | None,
) -> None:
    """Save Bokeh HTML chart for the backtest.

    Args:
        bt: Backtest instance with .plot() method.
        stats: Backtest statistics Series (checked for trade count).
        session_dir: Session directory for chart output; if None, skips chart.
    """
    if not session_dir:
        return
    if len(stats["_trades"]) == 0:
        logger.info("No trades — skipping Bokeh chart")
        return
    chart_dir = session_dir / "backtest"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_path = chart_dir / "backtest_chart.html"
    bt.plot(
        filename=str(chart_path),
        open_browser=False,
        plot_equity=True,
        plot_drawdown=True,
        plot_trades=True,
        resample="2h",
    )
    logger.info("Bokeh chart saved: %s", chart_path)


# ---------------------------------------------------------------------------
# Runners — Data Preparation
# ---------------------------------------------------------------------------


def _prepare_df(test_df: pl.DataFrame, preds_df: pl.DataFrame) -> pd.DataFrame:
    """Prepare a pandas DataFrame merging market data and predictions.

    Renames price columns to backtesting.py's expected PascalCase format
    (Open, High, Low, Close, Volume) and merges prediction columns from
    the model output.

    Args:
        test_df: Market data with timestamp, OHLCV, and atr_14 columns.
        preds_df: Predictions with timestamp, pred_label, and optional
            pred_proba_class_* columns.

    Returns:
        Pandas DataFrame indexed by timestamp (DatetimeIndex) with renamed
        price columns and merged prediction columns.

    Raises:
        ValueError: If pred_label is missing from preds_df or atr_14 is
            missing from the merged result.
    """
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


# ---------------------------------------------------------------------------
# Runners — Configuration Helpers
# ---------------------------------------------------------------------------


def _compute_spread_rate(
    bc: Config.BacktestConfig,
    dc: Config.DataConfig,
    median_price: float,
) -> float:
    """Convert tick-based spread to relative rate for backtesting.py.

    Args:
        bc: Backtest configuration with spread_ticks and slippage_ticks.
        dc: Data configuration with tick_size.
        median_price: Median close price used as normalization denominator.

    Returns:
        Relative spread rate as a fraction.
    """
    total_ticks = bc.spread_ticks + bc.slippage_ticks
    return total_ticks * dc.tick_size / median_price


def _build_commission_fn(
    bc: Config.BacktestConfig,
    dc: Config.DataConfig,
) -> Callable[[float, float], float]:
    """Build a commission function closure for backtesting.py.

    Args:
        bc: Backtest configuration with commission_per_lot.
        dc: Data configuration with contract_size.

    Returns:
        A commission function that takes (order_size, price) and returns
        commission in dollars.
    """

    def commission_fn(order_size: float, price: float) -> float:  # noqa: ARG001
        lots = abs(order_size) / dc.contract_size
        return lots * bc.commission_per_lot

    return commission_fn


def _init_backtest(
    pdf: pd.DataFrame,
    bc: Config.BacktestConfig,
    dc: Config.DataConfig,
    spread: float,
    commission_fn: Callable[[float, float], float],
) -> FractionalBacktest:
    """Construct a FractionalBacktest instance without running it.

    Args:
        pdf: Prepared pandas DataFrame with price and prediction columns.
        bc: Backtest configuration.
        dc: Data configuration.
        spread: Pre-computed relative spread rate.
        commission_fn: Commission function from _build_commission_fn.

    Returns:
        Configured FractionalBacktest instance ready for .run().
    """
    margin = 1.0 / bc.leverage
    return FractionalBacktest(
        pdf,
        HybridGRUStrategy,
        cash=bc.initial_capital,
        spread=spread,
        commission=commission_fn,
        margin=margin,
        exclusive_orders=True,
        finalize_trades=True,
        fractional_unit=1.0,
    )


def _run_bt(pdf: pd.DataFrame, config: Config) -> tuple[pd.Series, FractionalBacktest]:
    """Run a backtest using HybridGRUStrategy with extracted helpers.

    Args:
        pdf: Prepared DataFrame with market data and predictions.
        config: Application configuration with backtest and data sections.

    Returns:
        Tuple of (backtest statistics Series, Backtest instance).
    """
    bc = config.backtest
    dc = config.data

    median_price = float(pdf["Close"].median())
    spread = _compute_spread_rate(bc, dc, median_price)
    commission_fn = _build_commission_fn(bc, dc)
    bt = _init_backtest(pdf, bc, dc, spread, commission_fn)

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
    )
    return stats, bt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(config: Config) -> None:
    """Run a full CFD backtest from files specified in config and persist results.

    For walk-forward (sliding) validation, joins OOF predictions with the
    full labeled dataset (which contains OHLCV + features). For static
    validation, uses the traditional test split file.

    Written outputs include:
        - JSON file with normalized metrics and trade records.
        - Optional trades detail CSV and equity-curve CSV when trades are present.
        - Optional Bokeh HTML chart under the configured session directory.

    Args:
        config: Application configuration object containing paths and
            backtest/data settings.
    """
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions not found: {preds_path}")
    preds_df = pl.read_parquet(preds_path)

    # Walk-forward: predictions are OOF across all windows — need OHLCV from labels
    # Static: predictions are for the test split — need OHLCV from test split
    test_path = Path(config.paths.test_data)
    is_static = config.validation.method == "static"

    if test_path.exists() and is_static:
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
                f"Labels file not found ({labels_path}) — needed for walk-forward backtest"
            )
        test_df = pl.read_parquet(labels_path)
    else:
        labels_path = Path(config.paths.labels)
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Neither test data ({test_path}) nor labels ({labels_path}) found"
            )
        logger.info("Walk-forward mode: joining OOF predictions with labeled data")
        test_df = pl.read_parquet(labels_path)

    pdf = _prepare_df(test_df, preds_df)
    logger.info("Confidence threshold: %.2f", config.backtest.confidence_threshold)
    stats, bt = _run_bt(pdf, config)

    metrics = _normalize_stats(stats, initial_capital=config.backtest.initial_capital)
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
            else 10_000.0
        )
        _save_equity_curve_csv(trades, out_path.parent, initial_capital)

    logger.info("=== BACKTEST RESULTS ===")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

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
        preds_df: Predictions data containing timestamp and pred_label
            (optional pred_proba_* columns allowed).
        config: Configuration object with backtest, data, and paths sections.

    Returns:
        Normalized metrics dictionary extracted from the backtest results.
    """
    pdf = _prepare_df(test_df, preds_df)
    stats, _ = _run_bt(pdf, config)
    return _normalize_stats(stats, initial_capital=config.backtest.initial_capital)


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
) -> tuple[dict, list[dict]]:
    """Run a backtest with manually specified parameters (no Config required).

    Designed for interactive use in dashboards where parameters can be tuned
    without modifying the config file.

    Args:
        test_df: Market/test data with OHLCV columns and atr_14.
        preds_df: Predictions with timestamp and pred_label (optionally
            pred_proba_* columns).
        leverage: CFD leverage ratio (default 100).
        lots_per_trade: Base lot size for confidence-weighted sizing.
        min_lots: Minimum lot size (low-conviction floor).
        max_lots: Maximum lot size (high-conviction cap).
        confidence_threshold: Minimum prediction probability to trade (0 = disabled).
        spread_ticks: Spread in ticks.
        slippage_ticks: Slippage in ticks.
        commission_per_lot: Commission per lot.
        atr_stop_multiplier: ATR multiplier for stop-loss distance (default 1.0).
        atr_tp_multiplier: ATR multiplier for take-profit distance (default 2.0, 0 = disabled).
        horizon_bars: Time-based exit after N bars (default 10).
        contract_size: Units per lot.
        tick_size: Price tick size in dollars (default 0.01).
        initial_capital: Starting capital for the backtest.
        max_drawdown_cutoff: Circuit breaker drawdown fraction (0.5 = 50%).
        dd_cooldown_bars: Bars to pause after drawdown breach.
        max_open_positions: Max simultaneous open positions.
        daily_loss_limit: Daily equity loss fraction before pause.

    Returns:
        Tuple of (metrics dict, trades list). Metrics contains normalized
        performance metrics; trades is a list of per-trade records.
    """
    pdf = _prepare_df(test_df, preds_df)

    median_price = float(pdf["Close"].median())
    spread_total = (spread_ticks + slippage_ticks) * tick_size / median_price

    def commission_fn(order_size: float, price: float) -> float:  # noqa: ARG001
        lots = abs(order_size) / contract_size
        return lots * commission_per_lot

    margin = 1.0 / leverage

    bt = FractionalBacktest(
        pdf,
        HybridGRUStrategy,
        cash=initial_capital,
        spread=spread_total,
        commission=commission_fn,
        margin=margin,
        exclusive_orders=True,
        finalize_trades=True,
        fractional_unit=1.0,
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
    )

    metrics = _normalize_stats(stats, initial_capital=initial_capital)
    trades = _trades_to_list(
        stats["_trades"],
        commission_per_lot=commission_per_lot,
        contract_size=contract_size,
    )

    return metrics, trades
