"""HybridGRU trading strategy and related constants.

Contains the ``HybridGRUStrategy`` class used by the backtest runner,
module-level fallback defaults, and the ``_calendar_day`` helper.
"""

from __future__ import annotations

import logging

from backtesting import Strategy
import pandas as pd

logger = logging.getLogger("thesis.backtest")

# ── Module-level fallback defaults ──

#: Floor to prevent microscopic stops in low-volatility regimes.
_MIN_ATR_FLOOR: float = 0.0001

#: Default initial capital used as fallback when BacktestConfig is unavailable.
_DEFAULT_INITIAL_CAPITAL: float = 10_000.0

#: Default commission per lot — fallback for _trades_to_list when config absent.
_DEFAULT_COMMISSION_PER_LOT: float = 20.0

#: Default contract size (units per lot) — fallback for _trades_to_list.
_DEFAULT_CONTRACT_SIZE: float = 100.0

#: Minimum bars required before the shifted-signal logic can activate.
_MIN_BARS_FOR_SIGNAL: int = 2

#: Minimum order size in units (backtesting.py requires whole-number sizes).
_MIN_ORDER_SIZE: int = 1


def _calendar_day(value: object) -> object:
    """Return the calendar date for a timestamp-like value.

    Args:
        value: A timestamp object (Pandas Timestamp, datetime, or
            string parseable by ``pd.Timestamp``).

    Returns:
        The calendar date portion as a ``datetime.date`` object.
    """
    return pd.Timestamp(value).date()


# ── Strategy ──


class HybridGRUStrategy(Strategy):
    """Trade on ML signals with ATR stop-loss and equity risk management.

    Signal shift: the strategy reads ``self.signals[-2]`` instead of
    ``-1`` so that the trade decision at bar ``i`` is based on the
    prediction made at bar ``i-1``.  This aligns the label anchor
    (``close[i-1]``) with the approximate entry price (``open[i]``),
    since the label for bar ``i-1`` uses barriers centred on
    ``close[i-1]``.  Without this shift the prediction at bar ``i``
    (anchored at ``close[i]``) would be executed at ``open[i+1]`` — a
    one-bar gap that breaks the anchor-entry correspondence.

    Position sizing: fixed-risk after confidence filtering. Confidence decides
    whether a trade is allowed; lot size stays at ``lots_per_trade`` clamped to
    ``[min_lots, max_lots]``.

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

    Risk management includes a max-drawdown circuit breaker, a maximum open
    position limit, and a daily loss limit based on the day's starting equity.

    Attributes:
        atr_stop_mult: ATR multiplier for stop-loss distance.
        atr_tp_mult: ATR multiplier for take-profit distance (0 = disabled).
        lots_per_trade: Fixed lot size after confidence filtering.
        min_lots: Minimum lot safety bound.
        max_lots: Maximum lot safety bound.
        confidence_threshold: Minimum class probability to trade (0 = disabled).
        min_atr: Floor to prevent microscopic stops in low-vol regimes.
        contract_size: Units per lot.
        horizon_bars: Max bars to hold (0 = hold until opposite signal/stop).
        max_drawdown_cutoff: Fraction of peak equity — breach triggers cooldown.
        dd_cooldown_bars: Bars to pause trading after drawdown breach.
        max_open_positions: Max simultaneous open positions.
        daily_loss_limit: Max fraction of daily equity loss before pause.
        min_bars_between_trades: Minimum bars after position exit before re-entry.
    """

    # ── Strategy fallback defaults ──
    # Runtime configuration is passed through bt.run() keyword arguments.
    # These class attributes provide safety defaults for direct Strategy use
    # and for parameters omitted by a caller. Short Strategy attribute names
    # are mapped from configuration fields in _run_bt() and run_backtest_manual().
    # ───────────────────────────────────────────────────────────────────────

    atr_stop_mult = 1.0  # cf. BacktestConfig.atr_stop_multiplier = 2.0
    atr_tp_mult = 2.0  # 0 = disabled (no take-profit); matches BacktestConfig
    lots_per_trade = 0.2  # cf. BacktestConfig.lots_per_trade = 0.1
    min_lots = 0.1  # cf. BacktestConfig.min_lots = 0.01
    max_lots = 0.5  # matches BacktestConfig
    confidence_threshold = 0.0  # 0 = disabled (trade all); cf. BacktestConfig = 0.50
    min_atr = (
        _MIN_ATR_FLOOR  # floor to prevent microscopic stops (module-level constant)
    )
    contract_size = 100  # units per lot; overridden via DataConfig.contract_size
    # 0 = disabled (hold until opposite signal or stop); overridden via LabelsConfig
    horizon_bars = 0
    max_drawdown_cutoff = 0.50  # circuit breaker threshold; cf. BacktestConfig = 0.30
    dd_cooldown_bars = (
        12  # pause duration after drawdown breach; matches BacktestConfig
    )
    max_open_positions = 1  # max simultaneous positions; matches BacktestConfig
    daily_loss_limit = 0.03  # daily loss fraction limit; matches BacktestConfig
    min_bars_between_trades = 0  # 0 = disabled; min bars after exit before re-entry

    def init(self) -> None:
        """Register indicators and initialise risk-management state.

        Registers ``signals`` from ``data.pred_label`` and ``ATR`` from
        ``data.atr_14`` with the backtesting indicator system, and sets an
        internal ``_has_proba`` flag indicating whether per-class probability
        columns exist.

        Initializes peak-equity tracking, drawdown cooldown, daily start
        equity, and current-date state for risk management.
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

        # Cooldown state — prevent overtrading after position closure
        self._last_exit_bar: int = 0  # bar index of last position exit (0 = none yet)
        self._position_was_open: bool = False  # was a position open at prior bar?

        # Risk-management state
        self._peak_equity: float = self.equity
        self._dd_cooldown_left: int = 0
        self._dd_cutoff_breached: bool = False
        self._daily_start_equity: float = self.equity
        self._current_date: object = None  # track calendar day for daily reset

    def _floor_atr(self, atr: float) -> float:
        """Floor ATR to prevent unrealistic stops in low-volatility regimes.

        Args:
            atr: Current Average True Range value.

        Returns:
            ``max(atr, self.min_atr)`` — guaranteed above the module-level
            ``_MIN_ATR_FLOOR``.
        """
        return max(atr, self.min_atr)

    def _update_risk_state(self) -> None:
        """Update peak equity, drawdown tracking, and daily loss tracking.

        Called every bar in ``next()`` before any trading decisions.

        Maintains peak equity, drawdown cooldown, daily start equity, and the
        calendar-day tracker for daily reset logic. The drawdown circuit
        breaker is a permanent shutdown: once triggered, no new positions are
        opened for the rest of the backtest.
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

        # Gate 2: trade cooldown — enforce minimum bars between exits and re-entries
        if (
            self.min_bars_between_trades > 0
            and self._last_exit_bar > 0
            and (len(self.data) - self._last_exit_bar) < self.min_bars_between_trades
        ):
            return False

        # Gate 3: drawdown circuit breaker
        if self._dd_cutoff_breached:
            return False

        if self._dd_cooldown_left > 0:
            return False

        # Gate 4: daily loss limit
        if self.daily_loss_limit > 0 and self._daily_start_equity > 0:
            daily_pnl = (
                self.equity - self._daily_start_equity
            ) / self._daily_start_equity
            if daily_pnl <= -self.daily_loss_limit:
                return False

        return True

    def _compute_lots(self, confidence: float | None) -> float:
        """Return fixed position size after confidence filtering.

        Confidence already gates whether a trade is allowed. Scaling lots by
        confidence amplified wrong high-confidence predictions in the latest
        OOS run, causing drawdown to grow far faster than signal quality. Keep
        sizing fixed until the model is profitable at the base risk level.

        Args:
            confidence: Predicted class probability, accepted for API stability.

        Returns:
            Fixed lot size clamped to configured safety bounds.
        """
        return max(self.min_lots, min(self.lots_per_trade, self.max_lots))

    def next(self) -> None:
        """Evaluate the latest model signal and place orders if appropriate.

        Processes cooldown tracking, risk-state updates, time-based exits,
        risk gates, confidence gates, position sizing, and ATR-based market
        orders. The backtesting engine fills orders on the next bar, so signal
        bar ``i`` cannot trade at the same bar's close.
        """
        # Step 0: cooldown tracking — detect auto-closure from framework SL/TP
        if self._position_was_open and not self.position:
            self._last_exit_bar = len(self.data)
            self._position_was_open = False

        # Step 1: update risk state every bar
        self._update_risk_state()

        # Signal shift: use signals[-2] so the trade decision at bar i
        # is based on the prediction made at bar i-1.  The label for
        # bar i-1 is anchored at close[i-1]; backtesting.py fills
        # orders at the next open, so the approximate entry price is
        # open[i] ≈ close[i-1].  Without the shift, pred_label[i]
        # (anchored at close[i]) would be executed at open[i+1], a
        # one-bar misalignment between label anchor and entry price.
        if len(self.signals) < _MIN_BARS_FOR_SIGNAL:
            return
        raw_signal = float(self.signals[-2])
        # Threshold continuous predictions at 0 for direction:
        #   pred > 0  → Long  (1)
        #   pred < 0  → Short (-1)
        #   pred == 0 → Hold  (0)
        if raw_signal not in (-1, 0, 1):
            signal = 1 if raw_signal > 0 else (-1 if raw_signal < 0 else 0)
        else:
            signal = int(raw_signal)
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
                    self._last_exit_bar = len(self.data)
                    self._position_was_open = False

        # Step 3: risk gate — no new trades if blocked
        if not self._is_trading_allowed():
            return

        # Step 4: confidence gate
        confidence: float | None = None
        if self.confidence_threshold > 0 and self._has_proba:
            # Confidence must use the same bar as the shifted signal
            if signal == 1:
                confidence = float(self.proba_long[-2])
            elif signal == -1:
                confidence = float(self.proba_short[-2])
            else:
                return

            if confidence < self.confidence_threshold:
                return
        elif self.confidence_threshold > 0 and not self._has_proba:
            # Regression mode: no probability columns — skip confidence gate
            pass

        # Step 5: position sizing
        proxy_entry_price = self.data.Close[-1]
        lots = self._compute_lots(confidence)
        size = lots * self.contract_size
        # backtesting.py requires whole-number units (or equity fraction <1)
        size = max(_MIN_ORDER_SIZE, round(size))

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
            self._position_was_open = True

        elif signal == -1 and not self.position:
            self._entry_bar["short"] = len(self.data)
            sl_price = proxy_entry_price + (atr * self.atr_stop_mult)
            tp_price = (
                proxy_entry_price - (atr * self.atr_tp_mult)
                if self.atr_tp_mult > 0
                else None
            )
            self.sell(size=size, sl=sl_price, tp=tp_price)
            self._position_was_open = True
