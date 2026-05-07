"""HybridGRU trading strategy used in stage 5 backtests."""

from __future__ import annotations

import logging

from backtesting import Strategy
import pandas as pd

logger = logging.getLogger("thesis.backtest")

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
    """Return calendar date for a timestamp-like value."""
    return pd.Timestamp(value).date()


class HybridGRUStrategy(Strategy):
    """Trade on ML signals with ATR stops and simple risk gates.

    Key detail: use ``signals[-2]`` so the prediction anchored at ``close[i-1]``
    is executed at ``open[i]`` (backtesting fills on the next bar).

    Runtime configuration is passed through bt.run() keyword arguments.
    These class attributes provide safety defaults for direct Strategy use
    and for parameters omitted by a caller. Short Strategy attribute names
    are mapped from configuration fields in _run_fractional_backtest() and
    run_backtest_manual().
    """

    atr_stop_mult = 1.0  # cf. BacktestConfig.atr_stop_multiplier = 2.0
    atr_tp_mult = 2.0  # 0 = disabled (no take-profit); matches BacktestConfig
    lots_per_trade = 0.2  # cf. BacktestConfig.lots_per_trade = 0.1
    min_lots = 0.1  # cf. BacktestConfig.min_lots = 0.01
    max_lots = 0.5  # matches BacktestConfig
    confidence_threshold = 0.0  # 0 = disabled (trade all); cf. BacktestConfig = 0.50
    min_atr = _MIN_ATR_FLOOR  # floor to prevent microscopic stops
    contract_size = 100  # units per lot; overridden via DataConfig.contract_size
    horizon_bars = 0  # 0 = disabled (hold until opposite signal or stop)
    max_drawdown_cutoff = 0.50  # circuit breaker threshold; cf. BacktestConfig = 0.30
    dd_cooldown_bars = 12  # pause duration after drawdown breach
    max_open_positions = 1  # max simultaneous positions
    daily_loss_limit = 0.03  # daily loss fraction limit
    min_bars_between_trades = 0  # 0 = disabled; min bars after exit before re-entry

    def init(self) -> None:
        """Register indicators and initialise risk-management state."""
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
        self._last_exit_bar: int = 0
        self._position_was_open: bool = False

        self._peak_equity: float = self.equity
        self._dd_cooldown_left: int = 0
        self._dd_cutoff_breached: bool = False
        self._daily_start_equity: float = self.equity
        self._current_date: object = None

    def _floor_atr(self, atr: float) -> float:
        """Floor ATR to ``max(atr, self.min_atr)``."""
        return max(atr, self.min_atr)

    def _update_risk_state(self) -> None:
        """Update peak equity, drawdown tracking, and daily loss tracking.

        The drawdown circuit breaker is permanent: once triggered, no new
        positions are opened for the rest of the backtest.
        """
        eq = self.equity
        self._peak_equity = max(self._peak_equity, eq)

        if self._dd_cooldown_left > 0:
            self._dd_cooldown_left -= 1

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

        bar_date = _calendar_day(self.data.index[-1])
        if self._current_date != bar_date:
            self._current_date = bar_date
            self._daily_start_equity = eq

    def _is_trading_allowed(self) -> bool:
        """Check all risk gates before opening a new position."""
        if len(self.orders) > 0:
            return False

        if self.position and self.max_open_positions <= 1:
            return False

        if (
            self.min_bars_between_trades > 0
            and self._last_exit_bar > 0
            and (len(self.data) - self._last_exit_bar) < self.min_bars_between_trades
        ):
            return False

        if self._dd_cutoff_breached:
            return False

        if self._dd_cooldown_left > 0:
            return False

        if self.daily_loss_limit > 0 and self._daily_start_equity > 0:
            daily_pnl = (
                self.equity - self._daily_start_equity
            ) / self._daily_start_equity
            if daily_pnl <= -self.daily_loss_limit:
                return False

        return True

    def _compute_lots(self, confidence: float | None) -> float:
        """Return fixed position size after confidence filtering.

        Scaling lots by confidence amplified wrong high-confidence predictions
        in OOS runs. Keep sizing fixed until profitable at base risk level.
        """
        return max(self.min_lots, min(self.lots_per_trade, self.max_lots))

    def next(self) -> None:
        """Evaluate the latest model signal and place orders if appropriate.

        The backtesting engine fills orders on the next bar, so signal
        bar ``i`` cannot trade at the same bar's close.
        """
        # Cooldown tracking — detect auto-closure from framework SL/TP
        if self._position_was_open and not self.position:
            self._last_exit_bar = len(self.data)
            self._position_was_open = False

        self._update_risk_state()

        # Signal shift: trade at bar i uses prediction from bar i-1 (signals[-2]),
        # aligning label anchor close[i-1] with execution at open[i].
        if len(self.signals) < _MIN_BARS_FOR_SIGNAL:
            return
        raw_signal = float(self.signals[-2])
        if raw_signal not in (-1, 0, 1):
            signal = 1 if raw_signal > 0 else (-1 if raw_signal < 0 else 0)
        else:
            signal = int(raw_signal)
        atr = self._floor_atr(self.atr[-1])

        # Time-based exit
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

        if not self._is_trading_allowed():
            return

        # Confidence gate
        confidence: float | None = None
        if self.confidence_threshold > 0 and self._has_proba:
            if signal == 1:
                confidence = float(self.proba_long[-2])
            elif signal == -1:
                confidence = float(self.proba_short[-2])
            else:
                return

            if confidence < self.confidence_threshold:
                return

        # Position sizing
        proxy_entry_price = self.data.Close[-1]
        lots = self._compute_lots(confidence)
        size = lots * self.contract_size
        size = max(_MIN_ORDER_SIZE, round(size))

        # Execute trades
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
