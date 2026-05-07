"""Benchmark comparison helpers — naive strategies vs model.

Provides equity-curve construction, annualized Sharpe, max drawdown,
random-signal simulation, and the public `compute_benchmark_comparison`
entry point used by the report builder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from polars.exceptions import ComputeError

from thesis.shared.config import Config
from thesis.shared.constants import H1_BARS_PER_YEAR

logger = logging.getLogger("thesis.report")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_BARS_PER_YEAR = H1_BARS_PER_YEAR


# ---------------------------------------------------------------------------
# Model label helper (shared with _impl.py)
# ---------------------------------------------------------------------------


def _model_label(config: Config) -> str:
    """Human-readable model family label for reports."""
    architecture = config.model.architecture
    if architecture in ("static", "lgbm"):
        return "LightGBM"
    if architecture == "gru":
        return "GRU-only"
    if architecture == "hybrid":
        return "Hybrid GRU + LightGBM"
    return f"{architecture.title()} Model"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _annualized_sharpe(
    returns: np.ndarray, bars_per_year: int = _BARS_PER_YEAR
) -> float:
    """Compute annualized Sharpe ratio from bar returns.

    Args:
        returns: 1-D array of per-bar returns.
        bars_per_year: Number of bars in a trading year (default
            ``H1_BARS_PER_YEAR``).

    Returns:
        Annualized Sharpe ratio, or 0.0 if the standard deviation is
        zero or NaN.
    """
    std = float(np.std(returns, ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(bars_per_year))


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Compute maximum drawdown as a percentage from an equity curve.

    Args:
        equity: 1-D array representing cumulative equity over time.

    Returns:
        Maximum drawdown as a non-negative percentage (e.g. 15.3 for
        15.3%), or 0.0 if fewer than 2 data points.
    """
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    return float(abs(dd.min()))


def _equity_curve_from_bar_returns(
    returns: np.ndarray,
    initial_capital: float,
) -> np.ndarray:
    """Cumulative equity from per-bar fractional returns (length n+1)."""
    equity = np.empty(len(returns) + 1)
    equity[0] = initial_capital
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def _compute_random_strategy(
    returns: np.ndarray,
    initial_capital: float,
    leverage: int,
    seed: int,
) -> dict:
    """Simulate a random long/short signal strategy.

    Args:
        returns: 1-D array of per-bar returns.
        initial_capital: Starting equity value.
        leverage: CFD leverage multiplier.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``return_pct``, ``sharpe``, ``max_dd_pct``,
        ``win_rate_pct``, and ``num_trades``.
    """
    rng = np.random.default_rng(seed)
    signals = rng.choice([-1, 1], size=len(returns))
    leveraged = returns * signals * leverage

    equity = _equity_curve_from_bar_returns(leveraged, initial_capital)
    ret = (equity[-1] / initial_capital - 1) * 100
    sharpe = _annualized_sharpe(leveraged)
    max_dd = _max_drawdown_pct(equity)

    active = leveraged[signals != 0]
    win_rate = float((active > 0).sum() / len(active) * 100) if len(active) > 0 else 0.0

    return {
        "return_pct": ret,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "win_rate_pct": win_rate,
        "num_trades": int(np.abs(np.diff(signals)).sum() / 2 + 1),
    }


def _load_close_prices_for_benchmark(
    test_data_path: Path,
    hybrid_metrics: dict,
    config: Config,
) -> np.ndarray | None:
    """Load close prices for benchmark comparison.

    Walk-forward validation does not produce a static ``test.parquet``.
    Fall back to the full OHLCV dataset filtered by the backtest period
    recorded in the metrics.

    Args:
        test_data_path: Path to static test parquet (may not exist).
        hybrid_metrics: Backtest metrics containing ``start``/``end`` timestamps.
        config: Application configuration for resolving OHLCV path.

    Returns:
        1-D array of close prices, or ``None`` when no data is available.
    """
    # 1. Try static test split — only when validation method is actually "static"
    is_static = config.validation.method == "static"
    if test_data_path.exists() and is_static:
        try:
            df = pl.read_parquet(test_data_path, columns=["close"])
            return df["close"].to_numpy()
        except (ComputeError, OSError):
            logger.warning(
                "Failed to load static test data for benchmarks: %s",
                test_data_path,
                exc_info=True,
            )
    elif test_data_path.exists() and not is_static:
        logger.warning(
            "Static test file found (%s) but workflow is walk-forward "
            "(method='%s') — ignoring stale test_data for benchmarks",
            test_data_path,
            config.validation.method,
        )

    # 2. Walk-forward fallback: load OHLCV and filter to backtest period
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        logger.warning("No OHLCV data available for benchmark fallback: %s", ohlcv_path)
        return None

    try:
        df = pl.read_parquet(ohlcv_path)
    except (ComputeError, OSError):
        logger.warning(
            "Failed to load OHLCV for benchmarks: %s", ohlcv_path, exc_info=True
        )
        return None

    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()

    bt_start = hybrid_metrics.get("start")
    bt_end = hybrid_metrics.get("end")

    if bt_start and bt_end:
        start_dt = pl.lit(str(bt_start)[:19]).str.to_datetime()
        end_dt = pl.lit(str(bt_end)[:19]).str.to_datetime()
        df = df.filter((ts_col >= start_dt) & (ts_col <= end_dt))

    if len(df) < 2:
        logger.warning("OHLCV fallback for benchmarks: insufficient bars (%d)", len(df))
        return None

    logger.info("Benchmark using OHLCV fallback: %d bars", len(df))
    return df["close"].to_numpy()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_benchmark_comparison(
    test_data_path: Path,
    hybrid_metrics: dict,
    config: Config,
) -> list[dict]:
    """Compute benchmark comparison metrics for naive strategies vs hybrid model.

    Computes buy-and-hold, always-long, random-signal, and hybrid-model rows.

    Args:
        test_data_path: Path to the static test parquet file.
        hybrid_metrics: Backtest metrics from the hybrid model run.
        config: Application configuration.

    Returns:
        List of strategy dictionaries, each with ``strategy``,
        ``return_pct``, ``sharpe``, ``max_dd_pct``, ``win_rate_pct``,
        and ``num_trades``. Returns an empty list if no price data is
        available.
    """
    close = _load_close_prices_for_benchmark(test_data_path, hybrid_metrics, config)
    if close is None or len(close) < 2:
        return []
    if len(close) < 2:
        return []

    initial = config.backtest.initial_capital
    leverage = config.backtest.leverage
    seed = config.workflow.random_seed

    bar_returns = np.diff(close) / close[:-1]

    # 1. Buy & Hold (unleveraged, no costs)
    bh_equity = _equity_curve_from_bar_returns(bar_returns, initial)
    bh_return = (bh_equity[-1] / initial - 1) * 100

    # 2. Always Long (leveraged, no timing/costs)
    al_returns = bar_returns * leverage
    al_equity = _equity_curve_from_bar_returns(al_returns, initial)
    al_return = (al_equity[-1] / initial - 1) * 100

    # 3. Random Signal
    random_result = _compute_random_strategy(bar_returns, initial, leverage, seed)

    results: list[dict] = [
        {
            "strategy": "Buy & Hold",
            "return_pct": bh_return,
            "sharpe": _annualized_sharpe(bar_returns),
            "max_dd_pct": _max_drawdown_pct(bh_equity),
            "win_rate_pct": float("nan"),
            "num_trades": 1,
        },
        {
            "strategy": "Always Long",
            "return_pct": al_return,
            "sharpe": _annualized_sharpe(al_returns),
            "max_dd_pct": _max_drawdown_pct(al_equity),
            "win_rate_pct": float("nan"),
            "num_trades": 1,
        },
        {
            "strategy": "Random Signal",
            **random_result,
        },
        {
            "strategy": _model_label(config),
            "return_pct": hybrid_metrics.get("return_pct", 0),
            "sharpe": hybrid_metrics.get("sharpe_ratio", 0),
            "max_dd_pct": abs(hybrid_metrics.get("max_drawdown_pct", 0)),
            "win_rate_pct": hybrid_metrics.get("win_rate_pct", 0),
            "num_trades": int(hybrid_metrics.get("num_trades", 0)),
        },
    ]

    return results
