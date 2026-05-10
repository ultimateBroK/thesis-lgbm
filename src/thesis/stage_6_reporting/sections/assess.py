"""Assessment helpers and constants for backtest verdict logic.

Contains quality thresholds, edge classification, issue identification,
and the verdict/recommendation engine used by both section renderers
and table builders.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np

from thesis.shared.zones import _get_metric_zone

# ---------------------------------------------------------------------------
# Module-level constants (re-exported by __init__.py for _tables.py)
# ---------------------------------------------------------------------------

_QUALITY_ACC_DELTA: float = 0.05
_QUALITY_DIR_ACC_GOOD: float = 0.55
_QUALITY_MACRO_F1_GOOD: float = 0.45
_QUALITY_DIR_ACC_FAIR: float = 0.50

_EDGE_PF_NEGATIVE: float = 1.0
_EDGE_SHARPE_MARGINAL: float = 1.0
_EDGE_PF_MARGINAL: float = 1.5

_MIN_TRADES_DEPLOYABLE: int = 30

_ISSUE_DD_CATASTROPHIC: float = 50.0
_ISSUE_DD_ELEVATED: float = 30.0
_ISSUE_DD_CFD_ELEVATED: float = 20.0
_ISSUE_RET_SEVERE_LOSS: float = -50.0
_ISSUE_RET_SUSPICIOUS: float = 500.0
_ISSUE_WIN_RATE_VIABILITY: float = 40.0
_ISSUE_TRADES_MARGINAL: int = 100
_ISSUE_SHARPE_POOR: float = 0.5
_ISSUE_PF_MARGINAL_EDGE: float = 1.2

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
_SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "✅"}
_PRIORITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "✅"}

_ZONE_EMOJI_MAP: dict[str, str] = {
    "excellent": "✅",
    "good": "🟢",
    "moderate": "🟡",
    "poor": "🟠",
    "dangerous": "🔴",
}


# ---------------------------------------------------------------------------
# Assessment functions
# ---------------------------------------------------------------------------


def _get_zone_info(metric_name: str, value: float | None) -> tuple[str, str, str]:
    """Return (emoji, zone_label, recommended_range) for a backtest metric.

    Delegates to zones._get_metric_zone for consistent threshold application.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("⚪", "N/A", "N/A")

    color, label, rec = _get_metric_zone(metric_name, value)
    emoji = _ZONE_EMOJI_MAP.get(color, "⚪")
    return (emoji, label, rec)


def _assess_model_quality(pred_stats: dict) -> tuple[str, str]:
    """Classify ML quality into POOR / FAIR / GOOD with a short reason."""
    acc = pred_stats["accuracy"]
    baseline = pred_stats["majority_baseline"]
    dir_acc = pred_stats["directional_accuracy"]
    per_class = pred_stats["per_class"]
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))

    gap = acc - baseline
    if gap < 0:
        return ("POOR", "acc below baseline")
    if (
        acc > baseline + _QUALITY_ACC_DELTA
        and dir_acc > _QUALITY_DIR_ACC_GOOD
        and macro_f1 >= _QUALITY_MACRO_F1_GOOD
    ):
        return ("GOOD", "above baseline with directional edge")
    if dir_acc >= _QUALITY_DIR_ACC_FAIR:
        return ("FAIR", "slightly above baseline, marginal edge")
    return ("POOR", "no reliable directional edge")


def _assess_trading_edge(metrics: dict) -> tuple[str, str]:
    """Classify trading edge into NEGATIVE / MARGINAL / POSITIVE."""
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    ret = metrics.get("return_pct", 0)

    if pf < _EDGE_PF_NEGATIVE or sharpe < 0 or ret < 0:
        return ("NEGATIVE", f"PF={pf:.2f}" if pf > 0 else f"PF<{_EDGE_PF_NEGATIVE:.1f}")
    if sharpe < _EDGE_SHARPE_MARGINAL or pf < _EDGE_PF_MARGINAL:
        return ("MARGINAL", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")
    return ("POSITIVE", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")


def _derive_recommendation(ml_quality: str, trading_edge: str, metrics: dict) -> str:
    """Produce a deployment recommendation from model quality + trading edge."""
    n_trades = int(metrics.get("num_trades", 0)) if metrics else 0

    if ml_quality == "POOR" or trading_edge == "NEGATIVE":
        return "NOT DEPLOYABLE without fixes"
    if n_trades < _MIN_TRADES_DEPLOYABLE:
        return "NOT DEPLOYABLE — insufficient trades for validation"
    if ml_quality == "FAIR" and trading_edge == "MARGINAL":
        return "DEPLOYABLE with caution — marginal edge"
    if ml_quality == "GOOD" and trading_edge == "POSITIVE":
        return "DEPLOYABLE"
    return "DEPLOYABLE with caution"


def _identify_primary_issue(metrics: dict, pred_stats: dict | None) -> str | None:
    """Return the single most critical issue description, or None."""
    nt = int(metrics.get("num_trades", 0)) if metrics else 0
    sh = metrics.get("sharpe_ratio", 0) if metrics else 0
    pf = metrics.get("profit_factor", 0) if metrics else 0
    dd = abs(metrics.get("max_drawdown_pct", 0)) if metrics else 0
    ret = metrics.get("return_pct", 0) if metrics else 0
    wr = metrics.get("win_rate_pct", 0) if metrics else 0
    da = pred_stats.get("directional_accuracy", 0) if pred_stats else 0

    # Ordered check: first match is most critical
    checks: list[tuple[Callable[[], bool], Callable[[], str]]] = [
        (
            lambda: nt == 0,
            lambda: "Zero trades executed — model produces no actionable signals",
        ),
        (
            lambda: nt > 0 and nt < _MIN_TRADES_DEPLOYABLE,
            lambda: f"Only {nt} trades — statistically unreliable results",
        ),
        (
            lambda: sh < 0,
            lambda: (
                f"Sharpe {sh:.2f} is negative — strategy underperforms risk-free rate"
            ),
        ),
        (
            lambda: dd > _ISSUE_DD_CATASTROPHIC,
            lambda: (
                f"Max drawdown {dd:.1f}% > {_ISSUE_DD_CATASTROPHIC:.0f}%"
                " — catastrophic capital erosion"
            ),
        ),
        (
            lambda: pf < _EDGE_PF_NEGATIVE,
            lambda: (
                f"Profit factor {pf:.2f} < {_EDGE_PF_NEGATIVE:.1f}"
                " — strategy loses money on average"
            ),
        ),
        (
            lambda: da > 0 and da < _QUALITY_DIR_ACC_FAIR,
            lambda: (
                f"Directional accuracy {da:.1%} < {_QUALITY_DIR_ACC_FAIR:.0%}"
                " — predicts worse than random"
            ),
        ),
        (
            lambda: ret < _ISSUE_RET_SEVERE_LOSS,
            lambda: f"Return {ret:.0f}% — severe capital loss",
        ),
        (
            lambda: pf < _ISSUE_PF_MARGINAL_EDGE and pf >= _EDGE_PF_NEGATIVE,
            lambda: (
                f"Profit factor {pf:.2f} < {_ISSUE_PF_MARGINAL_EDGE:.1f}"
                " — barely covers transaction costs"
            ),
        ),
        (
            lambda: sh < _ISSUE_SHARPE_POOR and sh >= 0,
            lambda: (
                f"Sharpe {sh:.2f} < {_ISSUE_SHARPE_POOR:.1f}"
                " — poor risk-adjusted returns"
            ),
        ),
        (
            lambda: dd > _ISSUE_DD_ELEVATED and dd <= _ISSUE_DD_CATASTROPHIC,
            lambda: (
                f"Max drawdown {dd:.1f}% exceeds {_ISSUE_DD_ELEVATED:.0f}% threshold"
            ),
        ),
        (
            lambda: nt >= _MIN_TRADES_DEPLOYABLE and nt < _ISSUE_TRADES_MARGINAL,
            lambda: f"Only {nt} trades — marginal sample size",
        ),
        (
            lambda: sh < _EDGE_SHARPE_MARGINAL and sh >= _ISSUE_SHARPE_POOR,
            lambda: (
                f"Sharpe {sh:.2f} < {_EDGE_SHARPE_MARGINAL:.1f}"
                " — below professional threshold"
            ),
        ),
        (
            lambda: ret > _ISSUE_RET_SUSPICIOUS,
            lambda: f"Return {ret:.0f}% suspiciously high — verify for overfitting",
        ),
        (
            lambda: dd > _ISSUE_DD_CFD_ELEVATED and dd <= _ISSUE_DD_ELEVATED,
            lambda: (
                f"Max drawdown {dd:.1f}% > {_ISSUE_DD_CFD_ELEVATED:.0f}%"
                " — elevated for CFD trading"
            ),
        ),
        (
            lambda: wr < _ISSUE_WIN_RATE_VIABILITY and wr >= 0,
            lambda: (
                f"Win rate {wr:.1f}% < {_ISSUE_WIN_RATE_VIABILITY:.0f}%"
                " — below trading viability"
            ),
        ),
        (
            lambda: da > 0 and da < _QUALITY_DIR_ACC_GOOD,
            lambda: (
                f"Directional accuracy {da:.1%} < {_QUALITY_DIR_ACC_GOOD:.0%}"
                " — unreliable"
            ),
        ),
        (
            lambda: pf < _EDGE_PF_MARGINAL and pf >= _ISSUE_PF_MARGINAL_EDGE,
            lambda: f"Profit factor {pf:.2f} < {_EDGE_PF_MARGINAL:.1f} — marginal edge",
        ),
    ]
    for cond_fn, msg_fn in checks:
        if cond_fn():
            return msg_fn()
    return None
