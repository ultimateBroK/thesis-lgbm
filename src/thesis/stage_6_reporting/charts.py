"""Chart rendering helpers for the thesis report.

Contains equity-curve and feature-importance plot generators used by
:func:`generate_report`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from thesis.shared.config import Config

logger = logging.getLogger("thesis.report")


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def _build_equity_series(
    trades: list[dict], initial_capital: float
) -> tuple[list, list]:
    """Build timestamp and cumulative equity series from trades.

    The equity curve is trade-by-trade closed PnL, not mark-to-market, so
    intra-trade drawdowns are not visible.
    """
    times = [pd.to_datetime(trades[0]["entry_time"])]
    equity = [initial_capital]
    for t in trades:
        times.append(pd.to_datetime(t["exit_time"]))
        equity.append(equity[-1] + t["pnl"])
    return times, equity


def _plot_equity_curve(trades: list[dict], config: Config, out_dir: Path) -> None:
    """Render and save an equity curve image from trade history.

    Args:
        trades: List of trade dictionaries with ``pnl``,
            ``entry_time``, and ``exit_time``.
        config: Application configuration for initial capital.
        out_dir: Output directory for the saved PNG.
    """
    if not trades:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, equity = _build_equity_series(trades, config.backtest.initial_capital)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, equity, linewidth=1)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity (USD)")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: equity_curve.png")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def _load_feature_importance(config: Config, out_dir: Path) -> dict:
    """Load feature-importance JSON from session report outputs.

    Args:
        config: Application configuration.
        out_dir: Fallback directory when no session dir is configured.

    Returns:
        Feature-importance dictionary, or an empty dict if the JSON
        file is not found.
    """
    fi_path = (
        Path(config.paths.session_dir) / "reports" / "feature_importance.json"
        if config.paths.session_dir
        else out_dir.parent / "feature_importance.json"
    )
    if not fi_path.exists():
        return {}
    with open(fi_path) as f:
        return json.load(f)


def _plot_feature_importance(feature_importance: dict, out_dir: Path) -> None:
    """Render and save a top-20 feature-importance chart.

    Args:
        feature_importance: Dictionary mapping feature names to
            importance scores.
        out_dir: Output directory for the saved PNG.
    """
    if not feature_importance:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(list(top.keys()), list(top.values()))
    ax.set_title("Feature Importance (Top 20)")
    ax.invert_yaxis()
    fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: feature_importance.png")
