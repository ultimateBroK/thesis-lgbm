"""Stage 6: Report generation — markdown builder, statistics, charts, and orchestrator.

Moved from ``thesis.report`` into ``thesis.stage_6_reporting``.
"""

from __future__ import annotations

import json
import logging
import math
import tomllib
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from thesis._shared.config import Config
from thesis._shared.constants import H1_BARS_PER_YEAR
from thesis._shared.ui import console
from thesis._shared.zones import _get_metric_zone
from thesis.stage_4_training import _baselines
from thesis.stage_6_reporting import _calibration, _data_quality, _model_metrics

logger = logging.getLogger("thesis.report")

# ---------------------------------------------------------------------------
# Module-level constants — extracted from function bodies
# ---------------------------------------------------------------------------

# Confidence & baseline
_HIGH_CONFIDENCE_THRESHOLD: float = 0.70
_DIRECTIONAL_BASELINE: float = 0.5

# Model quality assessment
_QUALITY_ACC_DELTA: float = 0.05
_QUALITY_DIR_ACC_GOOD: float = 0.55
_QUALITY_MACRO_F1_GOOD: float = 0.45
_QUALITY_DIR_ACC_FAIR: float = 0.50

# Trading edge classification
_EDGE_PF_NEGATIVE: float = 1.0
_EDGE_SHARPE_MARGINAL: float = 1.0
_EDGE_PF_MARGINAL: float = 1.5

# Deployment recommendation
_MIN_TRADES_DEPLOYABLE: int = 30

# Issue identification thresholds
_ISSUE_DD_CATASTROPHIC: float = 50.0
_ISSUE_DD_ELEVATED: float = 30.0
_ISSUE_DD_CFD_ELEVATED: float = 20.0
_ISSUE_RET_SEVERE_LOSS: float = -50.0
_ISSUE_RET_SUSPICIOUS: float = 500.0
_ISSUE_WIN_RATE_VIABILITY: float = 40.0
_ISSUE_TRADES_MARGINAL: int = 100
_ISSUE_SHARPE_POOR: float = 0.5
_ISSUE_PF_MARGINAL_EDGE: float = 1.2

# Hybrid-vs-static comparison
_MIN_WINDOWS_COMPARISON: int = 3
_SIGNIFICANCE_ALPHA: float = 0.05
_MAX_PER_WINDOW_DISPLAY: int = 10

# Expected Calibration Error (ECE)
_ECE_N_BINS: int = 10
_ECE_WELL_CALIBRATED: float = 0.05
_ECE_MODERATELY_CALIBRATED: float = 0.15

# ---------------------------------------------------------------------------
# Stats helpers (formerly report/stats.py)
# ---------------------------------------------------------------------------


def _load_label_distribution(labels_path: Path) -> dict | None:
    """Compute class distribution from the labels parquet file.

    Args:
        labels_path: Path to the labels parquet file.

    Returns:
        A dictionary with class counts/percentages for ``Short``, ``Hold``, and
        ``Long``, plus ``total``; returns ``None`` when unavailable.
    """
    if not labels_path.exists():
        return None
    try:
        df = pl.read_parquet(labels_path, columns=["label"])
        total = len(df)
        dist: dict[str, Any] = {}
        for label_val, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            count = int((df["label"] == label_val).sum())
            pct = count / total * 100 if total > 0 else 0.0
            dist[name] = (count, pct)
        dist["total"] = total
        return dist
    except Exception:
        logger.warning(
            "Failed to load label distribution: %s", labels_path, exc_info=True
        )
        return None


def _load_prediction_stats(preds_path: Path) -> dict | None:
    """Compute prediction quality statistics from a predictions parquet file.

    Args:
        preds_path: Path to predictions parquet containing ``true_label``,
            ``pred_label``, and optional class-probability columns.

    Returns:
        A dictionary with overall accuracy, directional accuracy, baselines,
        per-class metrics, confusion matrix, and optional high-confidence
        stats; returns ``None`` if the file is unavailable or unreadable.
    """
    if not preds_path.exists():
        return None
    try:
        df = pl.read_parquet(preds_path)
        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()

        proba_cols = [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]
        proba = (
            df.select(proba_cols).to_numpy()
            if all(c in df.columns for c in proba_cols)
            else None
        )

        raw_metrics = _model_metrics.compute_all_classification_metrics(
            true,
            pred,
            y_proba=proba,
        )
        per_class_metrics = raw_metrics["precision_recall_f1_per_class"]
        class_map = {-1: "Short", 0: "Hold", 1: "Long"}
        per_class_counts = {
            class_map[c]: {
                "true_count": int((true == c).sum()),
                "pred_count": int((pred == c).sum()),
                "precision": float(per_class_metrics[class_map[c]]["precision"]),
                "recall": float(per_class_metrics[class_map[c]]["recall"]),
                "f1": float(per_class_metrics[class_map[c]]["f1"]),
            }
            for c in (-1, 0, 1)
        }

        result: dict[str, Any] = {
            "total": int(raw_metrics["total"]),
            "accuracy": float(raw_metrics["accuracy"]),
            "balanced_accuracy": float(raw_metrics["balanced_accuracy"]),
            "directional_accuracy": float(raw_metrics["directional_accuracy"]),
            "directional_baseline": _DIRECTIONAL_BASELINE,
            "majority_baseline": float(raw_metrics["majority_baseline_accuracy"]),
            "macro_f1": float(raw_metrics["macro_f1"]),
            "weighted_f1": float(raw_metrics["weighted_f1"]),
            "per_class": per_class_counts,
            "confusion_matrix": raw_metrics["confusion_matrix"],
            "direction_confusion_matrix": raw_metrics["direction_confusion_matrix"],
        }

        if proba is not None:
            max_proba = proba.max(axis=1)
            hc_mask = max_proba >= _HIGH_CONFIDENCE_THRESHOLD
            hc_count = int(hc_mask.sum())
            hc_acc = float((true[hc_mask] == pred[hc_mask]).mean()) if hc_count else 0.0
            hc_non_hold = hc_mask & (pred != 0)
            hc_non_hold_count = int(hc_non_hold.sum())
            hc_dir_acc = (
                float((true[hc_non_hold] == pred[hc_non_hold]).mean())
                if hc_non_hold_count
                else 0.0
            )
            result["high_confidence"] = {
                "threshold": _HIGH_CONFIDENCE_THRESHOLD,
                "count": hc_count,
                "pct_of_total": (hc_count / len(true) * 100.0) if len(true) else 0.0,
                "accuracy": hc_acc,
                "directional_accuracy": hc_dir_acc,
            }
        return result
    except Exception:
        logger.warning(
            "Failed to load prediction statistics: %s", preds_path, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Confusion cost matrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Benchmark comparison helpers (formerly report/stats.py)
# ---------------------------------------------------------------------------

_BARS_PER_YEAR = H1_BARS_PER_YEAR


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


def _build_equity_curve(
    returns: np.ndarray,
    initial_capital: float,
) -> np.ndarray:
    """Build equity curve from bar returns and initial capital.

    Args:
        returns: 1-D array of per-bar returns as fractional changes.
        initial_capital: Starting equity value.

    Returns:
        Equity curve array with length ``len(returns) + 1``, where
        ``equity[0]`` equals ``initial_capital``.
    """
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

    equity = _build_equity_curve(leveraged, initial_capital)
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
        except Exception:
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
    except Exception:
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


def compute_benchmark_comparison(
    test_data_path: Path,
    hybrid_metrics: dict,
    config: Config,
) -> list[dict]:
    """Compute benchmark comparison metrics for naive strategies vs hybrid model.

    Strategies computed:
        1. Buy & Hold — unleveraged, no costs.
        2. Always Long — leveraged, no timing.
        3. Random Signal — random long/short with leverage.
        4. Hybrid Model — actual backtest results.

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
    bh_equity = _build_equity_curve(bar_returns, initial)
    bh_return = (bh_equity[-1] / initial - 1) * 100

    # 2. Always Long (leveraged, no timing/costs)
    al_returns = bar_returns * leverage
    al_equity = _build_equity_curve(al_returns, initial)
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


# ---------------------------------------------------------------------------
# Markdown builder (formerly report/builder.py)
# ---------------------------------------------------------------------------

_ZONE_EMOJI = {
    "excellent": "✅",
    "good": "🟢",
    "moderate": "🟡",
    "poor": "🟠",
    "dangerous": "🔴",
}


def _zone(key: str, value: float) -> str:
    """Zone emoji for a metric value."""
    if value is None or (
        isinstance(value, float)
        and (math.isnan(value) if isinstance(value, float) else False)
    ):
        return "⚪"
    color, _, _ = _get_metric_zone(key, value)
    return _ZONE_EMOJI.get(color, "⚪")


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def _fmt_f2(v: float) -> str:
    return f"{v:.2f}"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


def _tbl_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _model_label(config: Config) -> str:
    """Human-readable model family label for reports."""
    architecture = config.model.architecture
    if architecture == "static":
        return "Static LightGBM"
    if architecture == "hybrid":
        return "Hybrid GRU + LightGBM"
    return f"{architecture.title()} Model"


def _static_vs_hybrid_comparison(L: list[str], config: Config) -> None:
    """Render hybrid-vs-static statistical comparison section.

    Loads walk-forward history from the current session and a sibling session
    of the opposite architecture, performs a paired t-test on per-window
    accuracy, and appends markdown lines to ``L``.

    Args:
        L: Output markdown lines.
        config: Loaded runtime configuration.
    """
    current_arch = config.model.architecture
    # Only meaningful for hybrid vs static
    if current_arch not in ("hybrid", "static"):
        return

    target_arch = "static" if current_arch == "hybrid" else "hybrid"
    current_session = config.paths.session_dir

    if not current_session:
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — no session directory configured.*")
        L.append("")
        return

    current_wf_path = Path(current_session) / "reports" / "walk_forward_history.json"
    if not current_wf_path.exists():
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            "*Comparison unavailable — walk-forward history not found for "
            f"current {current_arch} session.*"
        )
        L.append("")
        return

    # Find sibling session with opposite architecture
    results_dir = Path(current_session).parent
    sibling_session = _find_architecture_session(
        results_dir, target_arch, current_session
    )

    if sibling_session is None:
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — run both static and hybrid first.*")
        L.append(f"*No `{target_arch}` session found under `{results_dir}`.*")
        L.append("")
        return

    sibling_wf_path = sibling_session / "reports" / "walk_forward_history.json"
    if not sibling_wf_path.exists():
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            f"*Comparison unavailable — walk-forward history not found "
            f"for {target_arch} session `{sibling_session.name}`.*"
        )
        L.append("")
        return

    # Load both histories
    try:
        current_history = json.loads(current_wf_path.read_text())
        sibling_history = json.loads(sibling_wf_path.read_text())
    except Exception:
        logger.warning(
            "Failed to load walk-forward history for hybrid-vs-static comparison",
            exc_info=True,
        )
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — failed to load walk-forward history.*")
        L.append("")
        return

    current_windows = current_history.get("window_details", [])
    sibling_windows = sibling_history.get("window_details", [])

    if (
        len(current_windows) < _MIN_WINDOWS_COMPARISON
        or len(sibling_windows) < _MIN_WINDOWS_COMPARISON
    ):
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            "*Comparison unavailable — need at least "
            f"{_MIN_WINDOWS_COMPARISON} windows in each "
            f"session (have {len(current_windows)}/{len(sibling_windows)}).*"
        )
        L.append("")
        return

    # Pair windows by matching test date ranges
    paired = _pair_windows_by_date(current_windows, sibling_windows)

    if len(paired) < _MIN_WINDOWS_COMPARISON:
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append(
            f"*Comparison unavailable — only {len(paired)} overlapping "
            f"test windows found (need ≥{_MIN_WINDOWS_COMPARISON}).*"
        )
        L.append("")
        return

    current_accs = [p[0] for p in paired]
    sibling_accs = [p[1] for p in paired]

    # Paired t-test
    try:
        from scipy.stats import ttest_rel

        t_stat, p_value = ttest_rel(current_accs, sibling_accs)
    except Exception:
        logger.warning("ttest_rel failed", exc_info=True)
        L.append("#### Hybrid vs Static Comparison")
        L.append("")
        L.append("*Comparison unavailable — statistical test failed.*")
        L.append("")
        return

    current_mean = np.mean(current_accs)
    sibling_mean = np.mean(sibling_accs)
    delta_mean = current_mean - sibling_mean

    # Determine significance
    alpha = _SIGNIFICANCE_ALPHA
    if p_value < alpha:
        if delta_mean > 0:
            result_line = (
                f"{_model_label(config)} **significantly outperforms** "
                f"{target_arch.title()} (p={p_value:.4f})"
            )
        else:
            result_line = (
                f"{target_arch.title()} **significantly outperforms** "
                f"{_model_label(config)} (p={p_value:.4f})"
            )
    else:
        result_line = (
            f"{_model_label(config)} is **not significantly different** from "
            f"{target_arch.title()} (p={p_value:.4f})"
        )

    L.append("#### Hybrid vs Static Comparison")
    L.append("")
    L.append(result_line)
    L.append("")
    L.append(_tbl_row("Metric", _model_label(config), target_arch.title(), "Delta"))
    L.append(_tbl_row("------", "------", "------", "------"))
    L.append(
        _tbl_row(
            "Mean Accuracy",
            f"{current_mean * 100:.1f}%",
            f"{sibling_mean * 100:.1f}%",
            f"{delta_mean * 100:+.1f}pp",
        )
    )
    L.append(
        _tbl_row(
            "Paired Windows",
            str(len(paired)),
            str(len(paired)),
            "",
        )
    )
    L.append(
        _tbl_row(
            "t-statistic",
            "",
            "",
            f"{t_stat:.4f}",
        )
    )
    L.append(
        _tbl_row(
            "p-value",
            "",
            "",
            f"{p_value:.4f}",
        )
    )
    L.append("")

    # Per-window delta table (first N windows)
    L.append(
        f"**Per-Window Accuracy Delta** (first {_MAX_PER_WINDOW_DISPLAY} windows):"
    )
    L.append("")
    L.append(
        _tbl_row(
            "Window",
            _model_label(config),
            target_arch.title(),
            "Delta",
        )
    )
    L.append(_tbl_row("------", "------", "------", "------"))
    for i, (c_acc, s_acc) in enumerate(paired[:_MAX_PER_WINDOW_DISPLAY], 1):
        delta = c_acc - s_acc
        L.append(
            _tbl_row(
                str(i),
                f"{c_acc * 100:.1f}%",
                f"{s_acc * 100:.1f}%",
                f"{delta * 100:+.1f}pp",
            )
        )
    if len(paired) > _MAX_PER_WINDOW_DISPLAY:
        L.append(f"*... and {len(paired) - _MAX_PER_WINDOW_DISPLAY} more windows.*")
    L.append("")

    logger.info(
        "Hybrid vs Static comparison: %d paired windows, t=%.4f, p=%.4f, delta=%.4f",
        len(paired),
        t_stat,
        p_value,
        delta_mean,
    )


def _find_architecture_session(
    results_dir: Path, target_arch: str, exclude_session: str
) -> Path | None:
    """Find the most recent session directory with a given architecture.

    Args:
        results_dir: Directory containing session subdirectories.
        target_arch: Architecture to search for (``"static"`` or ``"hybrid"``).
        exclude_session: Session path to exclude (the current session).

    Returns:
        Path to the most recent matching session, or ``None``.
    """
    if not results_dir.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for session_dir in sorted(results_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_str = str(session_dir)
        if session_str == str(exclude_session):
            continue

        snapshot = session_dir / "config" / "config_snapshot.toml"
        if not snapshot.exists():
            continue

        try:
            with open(snapshot, "rb") as f:
                data = tomllib.load(f)
            arch = data.get("model", {}).get("architecture", "")
            if arch == target_arch:
                # Use directory modification time for recency
                candidates.append((session_dir.stat().st_mtime, session_dir))
        except Exception:
            logger.debug(
                "Skipping session %s during architecture search",
                session_dir.name,
                exc_info=True,
            )
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _pair_windows_by_date(
    current_windows: list[dict],
    sibling_windows: list[dict],
) -> list[tuple[float, float]]:
    """Pair windows by overlapping test date ranges.

    Each window dict is expected to have ``accuracy``, ``test_dates``
    (with ``start``/``end`` keys), and ``window``.

    Args:
        current_windows: Window details from the current session.
        sibling_windows: Window details from the sibling session.

    Returns:
        List of ``(current_accuracy, sibling_accuracy)`` paired by
        best-overlapping test date range.
    """
    paired: list[tuple[float, float]] = []

    for cw in current_windows:
        if "accuracy" not in cw or cw["accuracy"] is None:
            continue
        cd = cw.get("test_dates", {})
        c_start = _parse_date(cd.get("start", ""))
        c_end = _parse_date(cd.get("end", ""))
        if c_start is None or c_end is None:
            continue

        best_sw = None
        best_overlap = timedelta.min
        for sw in sibling_windows:
            if "accuracy" not in sw or sw["accuracy"] is None:
                continue
            sd = sw.get("test_dates", {})
            s_start = _parse_date(sd.get("start", ""))
            s_end = _parse_date(sd.get("end", ""))
            if s_start is None or s_end is None:
                continue

            overlap_start = max(c_start, s_start)
            overlap_end = min(c_end, s_end)
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_sw = sw

        if best_sw is not None and best_overlap > timedelta(0):
            paired.append((cw["accuracy"], best_sw["accuracy"]))

    return paired


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string into a datetime, trying multiple formats.

    Args:
        date_str: Date string in one of the supported formats
            (``"%Y-%m-%d"``, ``"%Y-%m-%d %H:%M:%S"``, or ISO 8601
            variants).

    Returns:
        Parsed ``datetime`` object, or ``None`` if no format matched.
    """
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(
                date_str[:19] if len(date_str) > 19 else date_str, fmt
            )
        except ValueError:
            continue
    return None


def _render_data_quality_section(L: list[str], config: Config) -> None:
    """Render the Data Quality analysis section from the JSON sidecar.

    Loads ``config.paths.data_quality_json`` and produces a markdown table
    with duplicate detection, gap distribution (weekend vs real), estimated
    missing bars, largest gap, and data coverage dates.

    Args:
        L: Output markdown lines list (mutated in-place).
        config: Application configuration.
    """
    dq_path = Path(config.paths.data_quality_json)
    if not dq_path.exists():
        L.append("*Data quality JSON not found — stage 1 may not have run.*")
        L.append("")
        return

    try:
        with open(dq_path) as f:
            dq = json.load(f)
    except Exception:
        logger.warning("Failed to load data quality JSON: %s", dq_path, exc_info=True)
        L.append("*Data quality JSON could not be read.*")
        L.append("")
        return

    L.append("## Data Quality")
    L.append("")
    L.append(
        'This section addresses the thesis question: *"Is that because of data, '
        'the result awful?"*'
    )
    L.append("")
    L.append(_tbl_row("Metric", "Value"))
    L.append(_tbl_row("------", "-----"))
    L.append(_tbl_row("Total Bars", f"{dq.get('total_bars', 0):,}"))
    L.append(_tbl_row("Deduped Timestamps", f"{dq.get('deduped_timestamps', 0):,}"))
    L.append(_tbl_row("Calendar Gaps (all)", f"{dq.get('calendar_gaps', 0):,}"))
    L.append(_tbl_row("  - Weekend / Holiday", f"{dq.get('weekend_gaps', 0):,}"))
    L.append(_tbl_row("  - Real Gaps", f"{dq.get('real_gaps', 0):,}"))
    L.append(
        _tbl_row(
            "Estimated Missing Bars",
            f"{dq.get('estimated_missing_bars', 0):,}",
        )
    )
    L.append(
        _tbl_row(
            "Largest Gap",
            f"{dq.get('largest_gap_bars', 0)} bars",
        )
    )
    L.append(_tbl_row("Data Start", str(dq.get("start_date", "N/A"))))
    L.append(_tbl_row("Data End", str(dq.get("end_date", "N/A"))))
    L.append("")

    # --- Computed data quality from _data_quality module ---
    ohlcv_path = Path(config.paths.ohlcv)
    if ohlcv_path.exists():
        try:
            ohlcv_df = pl.read_parquet(ohlcv_path)
            computed_dq = _data_quality.compute_data_quality_report(ohlcv_df)
            ohlcv_c = computed_dq.get("ohlcv_consistency", {})
            missing = computed_dq.get("missing_bars", {})
            outliers = computed_dq.get("outlier_returns", {})

            L.append("### OHLCV Consistency (computed)")
            L.append("")
            L.append(_tbl_row("Check", "Result"))
            L.append(_tbl_row("-----", "------"))
            L.append(
                _tbl_row(
                    "Total rows",
                    f"{ohlcv_c.get('total_rows', 0):,}",
                )
            )
            L.append(
                _tbl_row(
                    "OHLC violations",
                    f"{ohlcv_c.get('ohlc_violations', 0):,}",
                )
            )
            L.append(
                _tbl_row(
                    "Negative prices",
                    f"{ohlcv_c.get('price_negative_count', 0):,}",
                )
            )
            L.append(
                _tbl_row(
                    "Consistent",
                    "✅ Yes" if ohlcv_c.get("is_consistent") else "❌ No",
                )
            )
            L.append("")

            L.append("### Missing Bar Analysis (computed)")
            L.append("")
            L.append(_tbl_row("Metric", "Value"))
            L.append(_tbl_row("------", "-----"))
            L.append(
                _tbl_row(
                    "Total bars",
                    f"{missing.get('total_bars', 0):,}",
                )
            )
            L.append(
                _tbl_row(
                    "Gaps found",
                    f"{missing.get('gaps_found', 0):,}",
                )
            )
            L.append(
                _tbl_row(
                    "Weekend gaps",
                    f"{missing.get('weekend_gaps', 0):,}",
                )
            )
            L.append(
                _tbl_row(
                    "Missing ratio",
                    f"{missing.get('missing_ratio', 0.0):.6f}",
                )
            )
            L.append("")

            if outliers:
                L.append("### Outlier Returns (computed)")
                L.append("")
                L.append(_tbl_row("Metric", "Value"))
                L.append(_tbl_row("------", "-----"))
                L.append(
                    _tbl_row(
                        "Outlier count",
                        str(outliers.get("outlier_count", 0)),
                    )
                )
                L.append(
                    _tbl_row(
                        "Outlier ratio",
                        f"{outliers.get('outlier_ratio', 0.0):.6f}",
                    )
                )
                L.append(
                    _tbl_row(
                        "Max return",
                        f"{outliers.get('max_return', 0.0):.8f}",
                    )
                )
                L.append(
                    _tbl_row(
                        "Min return",
                        f"{outliers.get('min_return', 0.0):.8f}",
                    )
                )
                L.append("")
        except Exception:
            logger.warning("Failed to compute data quality from OHLCV", exc_info=True)


def _render_label_design_section(L: list[str], config: Config) -> None:
    """Render the Label Design & Methodology explanation section.

    Describes the triple-barrier labeling method, class definitions, and
    ATR-based barrier configuration used to produce the classification
    target (Short / Hold / Long).

    Args:
        L: Output markdown lines list (mutated in-place).
        config: Application configuration.
    """
    L.append("## Label Design & Methodology")
    L.append("")
    labels_cfg = config.labels
    split_cfg = config.splitting
    L.append(
        "Labels are generated using the **triple-barrier method**: for each "
        "bar, ATR-scaled take-profit and stop-loss barriers are placed "
        "symmetrically. The first barrier touched within the forward horizon "
        "determines the class label."
    )
    L.append("")
    L.append(_tbl_row("Parameter", "Value"))
    L.append(_tbl_row("---------", "-----"))
    L.append(_tbl_row("ATR TP multiplier", f"{labels_cfg.atr_tp_multiplier}×"))
    L.append(_tbl_row("ATR SL multiplier", f"{labels_cfg.atr_sl_multiplier}×"))
    L.append(_tbl_row("Horizon", f"{labels_cfg.horizon_bars} bars"))
    L.append(_tbl_row("Classes", str(labels_cfg.num_classes)))
    L.append(
        _tbl_row(
            "Class mapping",
            "Short (-1) / Hold (0) / Long (+1)",
        )
    )
    L.append(
        _tbl_row("Train period", f"{split_cfg.train_start} → {split_cfg.train_end}")
    )
    L.append(
        _tbl_row("Validation period", f"{split_cfg.val_start} → {split_cfg.val_end}")
    )
    L.append(
        _tbl_row("Test (OOS) period", f"{split_cfg.test_start} → {split_cfg.test_end}")
    )
    L.append("")

    # Label distribution
    labels_path = Path(config.paths.labels)
    dist = _load_label_distribution(labels_path)
    if dist:
        L.append("**Class distribution:**")
        L.append("")
        L.append(_tbl_row("Class", "Count", "Share"))
        L.append(_tbl_row("-----", "-----", "-----"))
        for name in ("Short", "Hold", "Long"):
            count, pct = dist[name]
            L.append(_tbl_row(name, f"{count:,}", f"{pct:.1f}%"))
        L.append(_tbl_row("Total", f"{dist['total']:,}", ""))
        L.append("")


def _render_validation_methodology_section(L: list[str], config: Config) -> None:
    """Render the Validation Methodology section (walk-forward, purge/embargo).

    Args:
        L: Output markdown lines list (mutated in-place).
        config: Application configuration.
    """
    L.append("## Validation Methodology")
    L.append("")
    val_cfg = config.validation
    split_cfg = config.splitting

    method_label = (
        "Walk-forward (sliding window)"
        if val_cfg.method == "sliding"
        else "Static train/val/test split"
    )
    L.append(
        "Model evaluation uses a **walk-forward (anchored sliding-window)** "
        "cross-validation scheme to prevent look-ahead bias and simulate "
        "realistic deployment conditions."
    )
    L.append("")
    L.append(_tbl_row("Parameter", "Value"))
    L.append(_tbl_row("---------", "-----"))
    L.append(_tbl_row("Method", method_label))
    L.append(
        _tbl_row(
            "Train window",
            f"{val_cfg.train_window_bars:,} bars (~{val_cfg.train_window_bars // 8760}y)",
        )
    )
    L.append(
        _tbl_row(
            "Test window",
            f"{val_cfg.test_window_bars:,} bars (~{val_cfg.test_window_bars // 730}mo)",
        )
    )
    L.append(_tbl_row("Step", f"{val_cfg.step_bars:,} bars"))
    L.append(_tbl_row("Purge gap", f"{val_cfg.purge_bars} bars at train/test boundary"))
    L.append(_tbl_row("Embargo gap", f"{val_cfg.embargo_bars} bars after purge"))
    L.append(_tbl_row("Min train bars", f"{val_cfg.min_train_bars:,}"))
    L.append(
        _tbl_row(
            "Split purge",
            f"{split_cfg.purge_bars} bars / embargo {split_cfg.embargo_bars} bars",
        )
    )
    L.append("")
    L.append(
        "*The **purge** gap removes bars at the train/test boundary to prevent "
        "label leakage from the forward-looking horizon. The **embargo** gap "
        "adds an additional buffer after the purge to further isolate the test "
        "set. Together they ensure strict temporal separation between training "
        "and evaluation data.*"
    )
    L.append("")


def _render_auxiliary_regression_section(L: list[str], pred_stats: dict | None) -> None:
    """Render auxiliary regression metrics section (MAE/RMSE/R²) if available.

    Args:
        L: Output markdown lines list (mutated in-place).
        pred_stats: Preloaded prediction statistics.
    """
    L.append("## Auxiliary: Regression Metrics")
    L.append("")
    if pred_stats and any(k in pred_stats for k in ("mae", "rmse", "r2")):
        L.append(_tbl_row("Metric", "Value"))
        L.append(_tbl_row("------", "-----"))
        for key, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R²")]:
            val = pred_stats.get(key)
            if val is not None:
                L.append(_tbl_row(label, f"{val:.4f}"))
        L.append("")
    else:
        L.append(
            "*Regression metrics (MAE, RMSE, R²) are not available for the "
            "current multiclass classification objective. When the model is "
            'configured with `objective = "regression"`, this section will '
            "show continuous-return prediction quality.*"
        )
        L.append("")


def _render_oof_vs_oos_section(L: list[str], config: Config) -> None:
    """Render OOF vs OOS comparison section with side-by-side metrics table.

    OOF (Out-Of-Fold): Aggregated from walk-forward CV history across all
    windows. OOS (Out-Of-Sample): Computed from the held-out test period
    (2024-01 to 2026-03) by filtering prediction records to the OOS date
    range.

    Args:
        L: Output markdown lines list (mutated in-place).
        config: Application configuration.
    """
    # ── Load walk-forward history ───────────────────────────────────────
    session_dir = config.paths.session_dir
    if not session_dir:
        L.append("## OOF vs OOS Generalization Check")
        L.append("")
        L.append("*Comparison unavailable — no session directory configured.*")
        L.append("")
        return

    wf_path = Path(session_dir) / "reports" / "walk_forward_history.json"
    if not wf_path.exists():
        L.append("## OOF vs OOS Generalization Check")
        L.append("")
        L.append("*Comparison unavailable — walk-forward history not found.*")
        L.append("")
        return

    try:
        wf = json.loads(wf_path.read_text())
    except Exception:
        logger.warning(
            "Failed to load walk-forward history: %s", wf_path, exc_info=True
        )
        L.append("## OOF vs OOS Generalization Check")
        L.append("")
        L.append("*Comparison unavailable — failed to load walk-forward history.*")
        L.append("")
        return

    window_details = wf.get("window_details", [])
    if not window_details:
        L.append("## OOF vs OOS Generalization Check")
        L.append("")
        L.append(
            "*Comparison unavailable — no window details in walk-forward history.*"
        )
        L.append("")
        return

    # ── Aggregate OOF metrics across windows (weighted by test_rows) ────
    total_test_rows = 0
    weighted_acc = 0.0
    weighted_macro_f1 = 0.0
    class_support: dict[str, int] = {"-1": 0, "0": 0, "1": 0}
    weighted_class_f1: dict[str, float] = {"-1": 0.0, "0": 0.0, "1": 0.0}

    for wd in window_details:
        test_rows = wd.get("test_rows", 0)
        if test_rows <= 0:
            continue
        total_test_rows += test_rows

        acc = wd.get("accuracy")
        if acc is not None:
            weighted_acc += acc * test_rows

        per_class = wd.get("per_class", {})
        window_f1s: list[float] = []
        for cls_key in ("-1", "0", "1"):
            cls_f1 = per_class.get(cls_key, {}).get("f1", 0.0)
            window_f1s.append(cls_f1)
            support = per_class.get(cls_key, {}).get("support", 0)
            class_support[cls_key] += support
            weighted_class_f1[cls_key] += cls_f1 * support
        window_macro_f1 = float(np.mean(window_f1s)) if window_f1s else 0.0
        weighted_macro_f1 += window_macro_f1 * test_rows

    if total_test_rows == 0:
        oof_accuracy: float | None = None
        oof_macro_f1: float | None = None
        oof_class_f1: dict[str, float | None] = {"-1": None, "0": None, "1": None}
    else:
        oof_accuracy = weighted_acc / total_test_rows
        oof_macro_f1 = weighted_macro_f1 / total_test_rows
        oof_class_f1 = {}
        for cls_key in ("-1", "0", "1"):
            sup = class_support.get(cls_key, 0)
            oof_class_f1[cls_key] = (
                weighted_class_f1[cls_key] / sup if sup > 0 else None
            )

    # ── Compute OOS metrics from predictions filtered to test period ────
    oos_accuracy: float | None = None
    oos_macro_f1: float | None = None
    oos_class_f1: dict[str, float | None] = {"-1": None, "0": None, "1": None}

    preds_path = Path(config.paths.predictions)
    if preds_path.exists():
        oos_start = config.backtest.oob_start_date or config.splitting.test_start
        oos_end = config.backtest.oob_end_date or config.splitting.test_end

        if oos_start and oos_end:
            try:
                df = pl.read_parquet(preds_path)
                if "true_label" not in df.columns or "pred_label" not in df.columns:
                    logger.warning(
                        "Predictions parquet missing true_label/pred_label columns"
                    )
                else:
                    ts_col = df["timestamp"]
                    if ts_col.dtype != pl.Datetime:
                        try:
                            ts_col = ts_col.str.strptime(pl.Datetime)
                        except Exception:
                            ts_col = ts_col.cast(pl.Datetime)
                    start_dt = _parse_date(oos_start)
                    end_dt = _parse_date(oos_end)
                    if start_dt is not None and end_dt is not None:
                        end_dt = end_dt.replace(hour=23, minute=59, second=59)
                        oos_df = df.filter((ts_col >= start_dt) & (ts_col <= end_dt))
                        if len(oos_df) > 0:
                            true = oos_df["true_label"].to_numpy()
                            pred = oos_df["pred_label"].to_numpy()
                            oos_accuracy = float((true == pred).mean())

                            per_class_metrics: dict[str, dict] = {}
                            for lv, cls_key in [(-1, "-1"), (0, "0"), (1, "1")]:
                                true_mask = true == lv
                                pred_mask = pred == lv
                                recall = (
                                    float((pred[true_mask] == lv).mean())
                                    if true_mask.sum() > 0
                                    else 0.0
                                )
                                precision = (
                                    float((true[pred_mask] == lv).mean())
                                    if pred_mask.sum() > 0
                                    else 0.0
                                )
                                f1 = (
                                    (2 * precision * recall / (precision + recall))
                                    if (precision + recall) > 0
                                    else 0.0
                                )
                                per_class_metrics[cls_key] = {
                                    "f1": f1,
                                    "support": int(true_mask.sum()),
                                }
                            oos_macro_f1 = float(
                                np.mean(
                                    [
                                        per_class_metrics[k]["f1"]
                                        for k in ("-1", "0", "1")
                                    ]
                                )
                            )
                            oos_class_f1 = {
                                k: per_class_metrics[k]["f1"] for k in ("-1", "0", "1")
                            }
            except Exception:
                logger.warning(
                    "Failed to compute OOS prediction metrics", exc_info=True
                )

    # ── Render table ─────────────────────────────────────────────────────
    L.append("## OOF vs OOS Generalization Check")
    L.append("")
    L.append(
        "*OOF (Out-Of-Fold) metrics are aggregated across all walk-forward "
        "cross-validation windows. OOS (Out-Of-Sample) metrics are computed "
        "from the held-out test period (2024-01 to 2026-03). A meaningful gap "
        "between OOF and OOS suggests overfitting; close alignment suggests "
        "the model generalizes well.*"
    )
    L.append("")
    L.append(_tbl_row("Metric", "OOF (Walk-Forward)", "OOS (2024-2026)", "Delta"))
    L.append(_tbl_row("------", "-------------------", "----------------", "-----"))

    def _metric_row(name: str, oof_val: float | None, oos_val: float | None) -> None:
        oof_str = f"{oof_val * 100:.1f}%" if oof_val is not None else "N/A"
        oos_str = f"{oos_val * 100:.1f}%" if oos_val is not None else "N/A"
        if oof_val is not None and oos_val is not None:
            delta = oos_val - oof_val
            delta_str = f"{delta * 100:+.1f}pp"
        else:
            delta_str = "N/A"
        L.append(_tbl_row(name, oof_str, oos_str, delta_str))

    _metric_row("Accuracy", oof_accuracy, oos_accuracy)
    _metric_row("Macro F1", oof_macro_f1, oos_macro_f1)

    for cls_key, cls_name in [("-1", "Short"), ("0", "Flat"), ("1", "Long")]:
        _metric_row(
            f"F1 ({cls_name})", oof_class_f1.get(cls_key), oos_class_f1.get(cls_key)
        )

    L.append("")

    # ── Interpretive note ────────────────────────────────────────────────
    if oof_accuracy is not None and oos_accuracy is not None:
        gap = abs(oos_accuracy - oof_accuracy)
        if gap < 0.02:
            note = (
                "OOF-OOS alignment is tight (< 2pp) — model generalizes "
                "well to unseen data."
            )
        elif gap < 0.05:
            note = (
                "Moderate OOF-OOS gap (2-5pp) — acceptable but monitor for overfitting."
            )
        else:
            note = (
                "Large OOF-OOS gap (≥5pp) — possible overfitting; review "
                "feature stability and window design."
            )
        L.append(f"**Interpretation:** {note}")
        L.append("")

    logger.info(
        "OOF vs OOS comparison: OOF acc=%.4f, OOS acc=%.4f",
        oof_accuracy or 0.0,
        oos_accuracy or 0.0,
    )


def _compute_avg_win_loss_ratio(trades: list[dict]) -> float | None:
    """Compute average win / average loss ratio from trade records.

    Args:
        trades: List of trade dictionaries with ``pnl`` key.

    Returns:
        Ratio of avg win to abs(avg loss), or ``None`` if insufficient
        winning or losing trades.
    """
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    if not wins or not losses:
        return None
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
        return None
    return avg_win / avg_loss


def _get_zone_info(metric_name: str, value: float | None) -> tuple[str, str, str]:
    """Return (emoji, zone_label, recommended_range) for a backtest metric.

    Uses the simplified 3-level zone scheme from the thesis requirements,
    aligned with https://boringedge.com/backtest-metrics-explained.

    Args:
        metric_name: Internal metric key (e.g. ``sharpe_ratio``).
        value: Metric value; ``None`` or ``NaN`` yields ⚪ N/A.

    Returns:
        Tuple of (emoji, zone_description, recommended_range).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("⚪", "N/A", "N/A")

    zones: dict[str, list[tuple[float | None, float | None, str, str, str]]] = {
        "return_pct": [
            (None, 0, "🔴", "Below 0 — Loss", "> 0%"),
            (0, 10, "🟡", "0–10% — Low", "> 10%"),
            (10, None, "🟢", "> 10% — Good", "> 10%"),
        ],
        "sharpe_ratio": [
            (None, 0, "🔴", "Below 0 — Negative", "> 1.0"),
            (0, 1.0, "🟡", "0–1.0 — Acceptable", "> 1.0"),
            (1.0, None, "🟢", "> 1.0 — Good", "> 1.0"),
        ],
        "max_drawdown_pct": [
            (None, -30, "🔴", "> 30% — Dangerous", "< 15%"),
            (-30, -15, "🟡", "15–30% — Moderate", "< 15%"),
            (-15, None, "🟢", "< 15% — Excellent", "< 15%"),
        ],
        "win_rate_pct": [
            (None, 40, "🔴", "< 40% — Low", "> 55%"),
            (40, 55, "🟡", "40–55% — Acceptable", "> 55%"),
            (55, None, "🟢", "> 55% — Good", "> 55%"),
        ],
        "profit_factor": [
            (None, 1.0, "🔴", "< 1.0 — Losing", "> 1.5"),
            (1.0, 1.5, "🟡", "1.0–1.5 — Marginal", "> 1.5"),
            (1.5, None, "🟢", "> 1.5 — Good", "> 1.5"),
        ],
        "calmar_ratio": [
            (None, 0, "🔴", "Below 0 — Negative", "> 1.0"),
            (0, 1.0, "🟡", "0–1.0 — Acceptable", "> 1.0"),
            (1.0, None, "🟢", "> 1.0 — Good", "> 1.0"),
        ],
        "sortino_ratio": [
            (None, 0, "🔴", "Below 0 — Negative", "> 1.0"),
            (0, 1.0, "🟡", "0–1.0 — Acceptable", "> 1.0"),
            (1.0, None, "🟢", "> 1.0 — Good", "> 1.0"),
        ],
        "avg_win_loss_ratio": [
            (None, 1.0, "🔴", "< 1.0 — Losing", "> 1.5"),
            (1.0, 1.5, "🟡", "1.0–1.5 — Marginal", "> 1.5"),
            (1.5, None, "🟢", "> 1.5 — Good", "> 1.5"),
        ],
        "expectancy_pct": [
            (None, 0, "🔴", "< 0% — Negative", "> 0.5%"),
            (0, 0.5, "🟡", "0–0.5% — Small Edge", "> 0.5%"),
            (0.5, None, "🟢", "> 0.5% — Good", "> 0.5%"),
        ],
    }

    zone_list = zones.get(metric_name)
    if zone_list is None:
        return ("⚪", "Unclassified", "N/A")

    for lo, hi, emoji, label, rec in zone_list:
        if lo is None:
            if value <= (hi or 0):
                return (emoji, label, rec)
        elif hi is None:
            if value > lo:
                return (emoji, label, rec)
        else:
            if lo < value <= hi:
                return (emoji, label, rec)

    return ("⚪", "Unclassified", "N/A")


def _render_metric_zones_section(
    L: list[str],
    metrics: dict,
    trades: list[dict] | None = None,
) -> None:
    """Render backtest metric quality zones with emoji indicators and recommended ranges.

    Each metric is annotated with:
        - Value
        - Zone emoji + description (e.g. ``🔴 Below 0 — Negative``)
        - Recommended range (e.g. ``> 1.0``)

    Zone definitions follow the 3-level scheme:
        🔴 = poor/dangerous  │  🟡 = marginal/moderate  │  🟢 = good

    Arguments:
        L: Output markdown lines list (mutated in-place).
        metrics: Backtest metrics dictionary.
        trades: Optional trade records for computing win/loss ratio.
    """
    L.append("## Metric Quality Zones")
    L.append("")
    L.append(
        "*Each metric is classified into three quality zones based on "
        "industry-standard thresholds (see "
        "[Boring Edge](https://boringedge.com/backtest-metrics-explained)). "
        "🔴 = poor/dangerous, 🟡 = marginal, 🟢 = good.*"
    )
    L.append("")

    L.append(_tbl_row("Metric", "Value", "Zone & Rating", "Recommended"))
    L.append(_tbl_row("------", "-----", "------------", "-----------"))

    # Compute win/loss ratio from trades if available
    avg_wl: float | None = None
    if trades:
        avg_wl = _compute_avg_win_loss_ratio(trades)

    # -- Metric definitions: (key, label, format_fn) --
    metric_defs: list[tuple[str, str, Callable[[float], str], float | None]] = [
        ("return_pct", "Total Return", _fmt_pct, metrics.get("return_pct")),
        ("sharpe_ratio", "Sharpe Ratio", _fmt_f2, metrics.get("sharpe_ratio")),
        (
            "max_drawdown_pct",
            "Max Drawdown",
            lambda v: f"{abs(v):.1f}%",
            metrics.get("max_drawdown_pct"),
        ),
        ("win_rate_pct", "Win Rate", _fmt_pct, metrics.get("win_rate_pct")),
        (
            "profit_factor",
            "Profit Factor",
            _fmt_f2,
            metrics.get("profit_factor"),
        ),
        (
            "calmar_ratio",
            "Calmar Ratio",
            _fmt_f2,
            metrics.get("calmar_ratio"),
        ),
        (
            "sortino_ratio",
            "Sortino Ratio",
            _fmt_f2,
            metrics.get("sortino_ratio"),
        ),
        (
            "avg_win_loss_ratio",
            "Avg Win / Avg Loss",
            _fmt_f2,
            avg_wl,
        ),
        (
            "expectancy_pct",
            "Expectancy",
            _fmt_pct,
            metrics.get("expectancy_pct"),
        ),
    ]

    for key, label, fmt, val in metric_defs:
        if val is None:
            L.append(_tbl_row(label, "N/A", "⚪ N/A", "N/A"))
            continue
        emoji, zone_desc, rec = _get_zone_info(key, val)
        value_str = fmt(val)
        zone_str = f"{emoji} {zone_desc}"
        L.append(_tbl_row(label, value_str, zone_str, rec))
    L.append("")


def _render_baseline_comparison_section(L: list[str], config: Config) -> None:
    """Render baseline strategy comparison using the _baselines module.

    Loads true labels from the predictions parquet and bar returns from
    OHLCV data, then computes naive, majority, always-long/short/hold,
    and random baseline metrics for comparison with the model.

    Args:
        L: Output markdown lines list (mutated in-place).
        config: Application configuration.
    """
    L.append("## Baseline Comparison")
    L.append("")

    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        L.append("*Predictions not available — baseline comparison skipped.*")
        L.append("")
        return

    try:
        df = pl.read_parquet(preds_path)
    except Exception:
        logger.warning("Failed to load predictions for baselines", exc_info=True)
        L.append("*Predictions file could not be read.*")
        L.append("")
        return

    if "true_label" not in df.columns:
        L.append("*true_label column missing — baseline comparison skipped.*")
        L.append("")
        return

    y_true = df["true_label"].to_numpy()

    # Get bar returns for naive_direction baseline
    y_returns: np.ndarray | None = None
    ohlcv_path = Path(config.paths.ohlcv)
    if ohlcv_path.exists():
        try:
            ohlcv = pl.read_parquet(ohlcv_path, columns=["close"])
            close = ohlcv["close"].to_numpy()
            if len(close) > 1:
                bar_returns = np.diff(close) / close[:-1]
                n = min(len(y_true), len(bar_returns))
                y_returns = bar_returns[-n:]
                y_true = y_true[-n:]
        except Exception:
            logger.warning("Failed to load OHLCV for baseline returns", exc_info=True)

    if y_returns is None:
        # Fallback: label-derived synthetic returns (approximate)
        y_returns = y_true.astype(np.float64)

    try:
        baselines = _baselines.run_all_baselines(
            y_true, y_returns, seed=config.workflow.random_seed
        )
    except Exception:
        logger.warning("Failed to compute baselines", exc_info=True)
        L.append("*Baseline computation failed.*")
        L.append("")
        return

    L.append(
        "*Baseline strategies computed on the same prediction labels as reference. "
        "The model should outperform all baselines on directional accuracy and macro F1.*"
    )
    L.append("")
    L.append(_tbl_row("Strategy", "Accuracy", "Macro F1", "Dir. Accuracy"))
    L.append(_tbl_row("--------", "--------", "---------", "-------------"))
    for name, m in baselines.items():
        display = name.replace("_", " ").title()
        L.append(
            _tbl_row(
                display,
                f"{m['accuracy'] * 100:.1f}%",
                f"{m['macro_f1']:.3f}",
                f"{m['directional_accuracy'] * 100:.1f}%",
            )
        )
    L.append("")


def _build_model_comparison_rows(
    config: Config, pred_stats: dict | None
) -> list[dict[str, Any]]:
    """Build thesis-level model comparison rows with available metrics.

    Columns are aligned with the thesis table:
    Directional Acc, Accuracy, Macro F1, Long F1, Short F1, optional regression metrics.
    """
    rows: list[dict[str, Any]] = []

    if pred_stats:
        per_class = pred_stats.get("per_class", {})
        reg_aux = pred_stats.get("regression_auxiliary", {})
        rows.append(
            {
                "model": _model_label(config),
                "directional_accuracy": pred_stats.get("directional_accuracy"),
                "accuracy": pred_stats.get("accuracy"),
                "macro_f1": pred_stats.get("macro_f1"),
                "long_f1": per_class.get("Long", {}).get("f1"),
                "short_f1": per_class.get("Short", {}).get("f1"),
                "mae_return": reg_aux.get("mae"),
                "rmse_return": reg_aux.get("rmse"),
                "r2_return": reg_aux.get("r_squared"),
                "source": "current_session",
            }
        )

    preds_path = Path(config.paths.predictions)
    if preds_path.exists():
        try:
            df = pl.read_parquet(preds_path)
            y_true = df["true_label"].to_numpy()
            close_path = Path(config.paths.ohlcv)
            y_returns = y_true.astype(np.float64)
            if close_path.exists():
                ohlcv = pl.read_parquet(close_path, columns=["close"])
                close = ohlcv["close"].to_numpy()
                if len(close) > 1:
                    bar_returns = np.diff(close) / close[:-1]
                    n = min(len(y_true), len(bar_returns))
                    y_returns = bar_returns[-n:]
                    y_true = y_true[-n:]
            baselines = _baselines.run_all_baselines(
                y_true, y_returns, seed=config.workflow.random_seed
            )
            for baseline_key, label in (
                ("naive_direction", "Naive Direction"),
                ("majority_class", "Majority Baseline"),
                ("random", "Random Baseline"),
            ):
                if baseline_key not in baselines:
                    continue
                m = baselines[baseline_key]
                rows.append(
                    {
                        "model": label,
                        "directional_accuracy": m.get("directional_accuracy"),
                        "accuracy": m.get("accuracy"),
                        "macro_f1": m.get("macro_f1"),
                        "long_f1": None,
                        "short_f1": None,
                        "mae_return": None,
                        "rmse_return": None,
                        "r2_return": None,
                        "source": "derived_baseline",
                    }
                )
        except Exception:
            logger.warning(
                "Failed to build baseline rows for model comparison", exc_info=True
            )

    # Keep planned model slots visible even when not yet available.
    existing = {str(r["model"]).lower() for r in rows}
    for model_name in ("LightGBM Static", "GRU-only", "Hybrid GRU+LightGBM"):
        if model_name.lower() in existing:
            continue
        rows.append(
            {
                "model": model_name,
                "directional_accuracy": None,
                "accuracy": None,
                "macro_f1": None,
                "long_f1": None,
                "short_f1": None,
                "mae_return": None,
                "rmse_return": None,
                "r2_return": None,
                "source": "pending_experiment",
            }
        )
    return rows


def _write_model_comparison_artifacts(
    out_dir: Path, rows: list[dict[str, Any]]
) -> tuple[Path, Path]:
    """Write model comparison table to CSV and Markdown."""
    csv_path = out_dir / "model_comparison.csv"
    md_path = out_dir / "model_comparison.md"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)

    display_cols = [
        "model",
        "directional_accuracy",
        "accuracy",
        "macro_f1",
        "long_f1",
        "short_f1",
        "mae_return",
        "rmse_return",
        "r2_return",
        "source",
    ]
    with md_path.open("w") as f:
        f.write("# Model Comparison\n\n")
        f.write(
            "Primary focus: Directional Accuracy, Accuracy, Macro F1, and per-class F1. "
            "Rows with empty values require additional experiment runs.\n\n"
        )
        f.write("| " + " | ".join(display_cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(display_cols)) + "|\n")
        for row in rows:
            vals = []
            for col in display_cols:
                val = row.get(col)
                vals.append("" if val is None else str(val))
            f.write("| " + " | ".join(vals) + " |\n")
    return csv_path, md_path


def _build_markdown(
    config: Config,
    metrics: dict,
    trades: list[dict],
    feature_importance: dict,
    pred_stats: dict | None,
) -> str:
    """Build concise metrics-first markdown report.

    Args:
        config: Loaded runtime configuration.
        metrics: Backtest metrics dictionary.
        trades: Backtest trades list.
        feature_importance: Feature importance values.
        pred_stats: Preloaded prediction statistics, if available.

    Returns:
        Rendered markdown report content.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = config.paths.session_dir or "N/A"
    L: list[str] = []

    # -- Header --
    L.append(f"# Thesis Report: {_model_label(config)} — XAU/USD")
    L.append("")
    L.append(f"> Generated: {now} | Session: `{session}`")
    L.append("")

    # -- Executive Summary --
    L.append("## Executive Summary")
    L.append("")
    _exec_table(L, metrics, pred_stats)
    _exec_verdict(L, metrics, pred_stats)
    L.append("")

    # -- Configuration --
    L.append("## Configuration")
    L.append("")
    _config_table(L, config)
    L.append("")

    # ── SECTION 1: Data Quality ───────────────────────────────────────────
    _render_data_quality_section(L, config)

    # ── SECTION 2: Label Design & Methodology ─────────────────────────────
    _render_label_design_section(L, config)

    # ── SECTION 3: Validation Methodology ─────────────────────────────────
    _render_validation_methodology_section(L, config)

    # ── SECTION 4: Classification Metrics (Primary) ───────────────────────
    L.append("## Classification Metrics")
    L.append("")
    L.append(
        "*Classification metrics are the primary evaluation criterion for "
        "this thesis. Directional Accuracy and Macro F1 measure the model's "
        "ability to predict market direction (Short / Hold / Long).*"
    )
    L.append("")
    _accuracy_table(L, pred_stats, config)
    L.append("")

    # ── SECTION 5: Model Architecture & Features ──────────────────────────
    L.append("## Model Architecture & Features")
    L.append("")
    if config.model.architecture == "hybrid":
        _gru_summary(L, config)
    _feature_importance_table(L, feature_importance)
    L.append("")

    # ── SECTION 6: Model Comparison ───────────────────────────────────────
    _static_vs_hybrid_comparison(L, config)

    # ── SECTION 6b: Baseline Comparison ───────────────────────────────────
    _render_baseline_comparison_section(L, config)

    # ── SECTION 7: Auxiliary Regression Metrics ───────────────────────────
    _render_auxiliary_regression_section(L, pred_stats)

    # ── SECTION 8: Application Demo — Backtest Results ────────────────────
    L.append("## Application Demo: Backtest Results")
    L.append("")
    L.append(
        "*Backtest results are presented as an application demo to illustrate "
        "how classification signals *could* be translated into trades. "
        "They are **not** the primary evaluation criterion.*"
    )
    L.append("")
    _backtest_params_table(L, config)
    _backtest_metrics_table(L, metrics, config)
    _render_metric_zones_section(L, metrics, trades)
    L.append("")

    # ── SECTION 9: Application Demo — Benchmark Comparison ────────────────
    L.append("## Application Demo: Benchmark Comparison")
    L.append("")
    _benchmark_comparison_table(L, metrics, config)

    # ── SECTION 10: OOF vs OOS Generalization Check ───────────────────────
    _render_oof_vs_oos_section(L, config)

    # ── SECTION 11: Issues & Recommendations ──────────────────────────────
    L.append("## Issues & Recommendations")
    L.append("")
    _issues_list(L, metrics, trades, config, pred_stats)
    L.append("")

    return "\n".join(L)


def _build_model_evaluation_markdown(
    config: Config, pred_stats: dict | None, model_comparison_rows: list[dict[str, Any]]
) -> str:
    """Build compact evaluation-first markdown artifact."""
    lines: list[str] = ["# Model Evaluation", ""]
    lines.append(
        "This file is the primary ML evidence artifact. Backtest metrics are intentionally excluded."
    )
    lines.append("")
    lines.append(f"- Model: {_model_label(config)}")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if not pred_stats:
        lines.append("*Prediction statistics unavailable.*")
        return "\n".join(lines)

    lines.append("## Classification Metrics (Primary)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {pred_stats.get('accuracy', 0.0) * 100:.2f}% |")
    lines.append(
        f"| Directional Accuracy | {pred_stats.get('directional_accuracy', 0.0) * 100:.2f}% |"
    )
    lines.append(f"| Macro F1 | {pred_stats.get('macro_f1', 0.0):.4f} |")
    lines.append(
        f"| Balanced Accuracy | {pred_stats.get('balanced_accuracy', 0.0) * 100:.2f}% |"
    )
    lines.append("")
    lines.append("## Per-Class Metrics")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|")
    for class_name in ("Short", "Hold", "Long"):
        pc = pred_stats.get("per_class", {}).get(class_name, {})
        lines.append(
            f"| {class_name} | {pc.get('precision', 0.0):.4f} | {pc.get('recall', 0.0):.4f} | {pc.get('f1', 0.0):.4f} |"
        )
    lines.append("")

    reg_aux = pred_stats.get("regression_auxiliary")
    if reg_aux:
        lines.append("## Regression Auxiliary Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| MAE Return | {reg_aux.get('mae', float('nan')):.6f} |")
        lines.append(f"| RMSE Return | {reg_aux.get('rmse', float('nan')):.6f} |")
        lines.append(f"| R² Return | {reg_aux.get('r_squared', float('nan')):.6f} |")
        lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append(
        "| Model | Directional Acc | Accuracy | Macro F1 | Long F1 | Short F1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in model_comparison_rows:
        lines.append(
            "| {model} | {da} | {acc} | {mf1} | {lf1} | {sf1} |".format(
                model=row.get("model", ""),
                da=""
                if row.get("directional_accuracy") is None
                else f"{float(row['directional_accuracy']) * 100:.2f}%",
                acc=""
                if row.get("accuracy") is None
                else f"{float(row['accuracy']) * 100:.2f}%",
                mf1=""
                if row.get("macro_f1") is None
                else f"{float(row['macro_f1']):.4f}",
                lf1=""
                if row.get("long_f1") is None
                else f"{float(row['long_f1']):.4f}",
                sf1=""
                if row.get("short_f1") is None
                else f"{float(row['short_f1']):.4f}",
            )
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _exec_table(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    """Key ML-first metrics with application-demo metrics second.

    Args:
        L: Output markdown lines.
        metrics: Backtest metrics dictionary.
        pred_stats: Preloaded prediction statistics.
    """
    if not pred_stats and not metrics:
        L.append("*No metrics available.*")
        return

    L.append(_tbl_row("Metric", "Value", "Zone"))
    L.append(_tbl_row("------", "-----", "----"))

    if pred_stats:
        acc = pred_stats["accuracy"]
        maj_bl = pred_stats["majority_baseline"]
        acc_gap = acc - maj_bl
        dir_acc = pred_stats["directional_accuracy"]
        per_class = pred_stats["per_class"]
        macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))
        L.append(
            _tbl_row("Exact Accuracy", f"{acc * 100:.1f}%", _zone("accuracy", acc))
        )
        L.append(_tbl_row("Majority Baseline", f"{maj_bl * 100:.1f}%", ""))
        L.append(_tbl_row("Acc - Baseline", f"{acc_gap * 100:+.1f}pp", ""))
        L.append(
            _tbl_row(
                "Directional Acc.",
                f"{dir_acc * 100:.1f}%",
                _zone("directional_accuracy", dir_acc),
            )
        )
        L.append(_tbl_row("Macro F1", f"{macro_f1:.3f}", ""))

    if metrics:
        rows = [
            ("Demo Return", "return_pct", metrics.get("return_pct", 0), _fmt_pct),
            (
                "Demo Max DD",
                "max_drawdown_pct",
                metrics.get("max_drawdown_pct", 0),
                _fmt_pct,
            ),
            (
                "Demo Trades",
                "num_trades",
                float(metrics.get("num_trades", 0)),
                lambda v: f"{int(v):,}",
            ),
        ]
        for label, key, val, fmt in rows:
            L.append(_tbl_row(label, fmt(val), _zone(key, val)))


def _assess_model_quality(pred_stats: dict) -> tuple[str, str]:
    """Classify ML quality into POOR / FAIR / GOOD with a short reason.

    Args:
        pred_stats: Preloaded prediction statistics.

    Returns:
        (quality_label, reason_phrase) — e.g. ("POOR", "acc below baseline").
    """
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
    """Classify trading edge into NEGATIVE / MARGINAL / POSITIVE.

    Args:
        metrics: Backtest metrics dictionary.

    Returns:
        (edge_label, reason_phrase).
    """
    pf = metrics.get("profit_factor", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    ret = metrics.get("return_pct", 0)

    if pf < _EDGE_PF_NEGATIVE or sharpe < 0 or ret < 0:
        return ("NEGATIVE", f"PF={pf:.2f}" if pf > 0 else f"PF<{_EDGE_PF_NEGATIVE:.1f}")
    if sharpe < _EDGE_SHARPE_MARGINAL or pf < _EDGE_PF_MARGINAL:
        return ("MARGINAL", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")
    return ("POSITIVE", f"PF={pf:.2f}, Sharpe={sharpe:.2f}")


def _derive_recommendation(ml_quality: str, trading_edge: str, metrics: dict) -> str:
    """Produce a deployment recommendation from model quality + trading edge.

    Args:
        ml_quality: "POOR", "FAIR", or "GOOD".
        trading_edge: "NEGATIVE", "MARGINAL", or "POSITIVE".
        metrics: Backtest metrics dictionary.

    Returns:
        Recommendation string, e.g. "NOT DEPLOYABLE without fixes".
    """
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
    """Return the single most critical issue description, or None.

    Issues are ranked by severity (critical > warning > info), then by
    impact (e.g. zero trades beats low win rate).

    Checks use lazy ``(condition_fn, message_fn)`` tuples so that
    conditions and messages are only evaluated until the first match.

    Args:
        metrics: Backtest metrics dictionary.
        pred_stats: Preloaded prediction statistics.

    Returns:
        Most severe issue description, or ``None`` if none found.
    """
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
                f"Max drawdown {dd:.1f}% > {_ISSUE_DD_CATASTROPHIC:.0f}% — catastrophic capital erosion"
            ),
        ),
        (
            lambda: pf < _EDGE_PF_NEGATIVE,
            lambda: (
                f"Profit factor {pf:.2f} < {_EDGE_PF_NEGATIVE:.1f} — strategy loses money on average"
            ),
        ),
        (
            lambda: da > 0 and da < _QUALITY_DIR_ACC_FAIR,
            lambda: (
                f"Directional accuracy {da:.1%} < {_QUALITY_DIR_ACC_FAIR:.0%} — predicts worse than random"
            ),
        ),
        (
            lambda: ret < _ISSUE_RET_SEVERE_LOSS,
            lambda: f"Return {ret:.0f}% — severe capital loss",
        ),
        (
            lambda: pf < _ISSUE_PF_MARGINAL_EDGE and pf >= _EDGE_PF_NEGATIVE,
            lambda: (
                f"Profit factor {pf:.2f} < {_ISSUE_PF_MARGINAL_EDGE:.1f} — barely covers transaction costs"
            ),
        ),
        (
            lambda: sh < _ISSUE_SHARPE_POOR and sh >= 0,
            lambda: (
                f"Sharpe {sh:.2f} < {_ISSUE_SHARPE_POOR:.1f} — poor risk-adjusted returns"
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
                f"Sharpe {sh:.2f} < {_EDGE_SHARPE_MARGINAL:.1f} — below professional threshold"
            ),
        ),
        (
            lambda: ret > _ISSUE_RET_SUSPICIOUS,
            lambda: f"Return {ret:.0f}% suspiciously high — verify for overfitting",
        ),
        (
            lambda: dd > _ISSUE_DD_CFD_ELEVATED and dd <= _ISSUE_DD_ELEVATED,
            lambda: (
                f"Max drawdown {dd:.1f}% > {_ISSUE_DD_CFD_ELEVATED:.0f}% — elevated for CFD trading"
            ),
        ),
        (
            lambda: wr < _ISSUE_WIN_RATE_VIABILITY and wr >= 0,
            lambda: (
                f"Win rate {wr:.1f}% < {_ISSUE_WIN_RATE_VIABILITY:.0f}% — below trading viability"
            ),
        ),
        (
            lambda: da > 0 and da < _QUALITY_DIR_ACC_GOOD,
            lambda: (
                f"Directional accuracy {da:.1%} < {_QUALITY_DIR_ACC_GOOD:.0%} — unreliable"
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


def _render_ml_quality_paragraph(L: list[str], pred_stats: dict) -> None:
    """Append one-paragraph ML quality assessment to markdown lines.

    Args:
        L: Output markdown lines.
        pred_stats: Preloaded prediction statistics.
    """
    acc = pred_stats["accuracy"]
    baseline = pred_stats["majority_baseline"]
    dir_acc = pred_stats["directional_accuracy"]
    per_class = pred_stats["per_class"]
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))

    gap = acc - baseline
    if gap < 0:
        ml_quality = "weak"
        gate_msg = "Model is below majority baseline; predictive edge is not validated."
    elif (
        acc > baseline + _QUALITY_ACC_DELTA
        and dir_acc > _QUALITY_DIR_ACC_GOOD
        and macro_f1 >= _QUALITY_MACRO_F1_GOOD
    ):
        ml_quality = "strong"
        gate_msg = "Model is above baseline with directional edge."
    elif dir_acc >= _QUALITY_DIR_ACC_FAIR:
        ml_quality = "acceptable"
        gate_msg = "Model is slightly above baseline with marginal directional edge."
    else:
        ml_quality = "weak"
        gate_msg = "Model has no reliable directional edge."
    L.append(
        f"ML quality is **{ml_quality}**: exact accuracy {acc:.1%} vs "
        f"majority baseline {baseline:.1%}, directional accuracy {dir_acc:.1%}, "
        f"macro F1 {macro_f1:.3f}. {gate_msg} Backtest figures below are treated as an "
        "application demo, not the primary proof of model quality."
    )


def _render_synthesized_verdict(L: list[str], pred_stats: dict, metrics: dict) -> None:
    """Append synthesized verdict line (model quality + trading edge + recommendation).

    Args:
        L: Output markdown lines.
        pred_stats: Preloaded prediction statistics.
        metrics: Backtest metrics dictionary.
    """
    model_quality, ml_reason = _assess_model_quality(pred_stats)
    if metrics:
        trading_edge, trade_reason = _assess_trading_edge(metrics)
        recommendation = _derive_recommendation(model_quality, trading_edge, metrics)
        L.append(
            f"**Verdict:** Model quality **{model_quality}** ({ml_reason}), "
            f"Trading edge **{trading_edge}** ({trade_reason}), "
            f"Recommendation: **{recommendation}**."
        )
    else:
        L.append(
            f"**Verdict:** Model quality **{model_quality}** ({ml_reason}). "
            "No backtest metrics available for trading assessment."
        )


def _render_primary_issue(L: list[str], metrics: dict, pred_stats: dict) -> None:
    """Append primary issue identification and application demo summary.

    Args:
        L: Output markdown lines.
        metrics: Backtest metrics dictionary.
        pred_stats: Preloaded prediction statistics.
    """
    if metrics:
        primary = _identify_primary_issue(metrics, pred_stats)
        if primary:
            L.append(f"**Primary issue:** {primary}.")
    else:
        L.append("**Primary issue:** No backtest metrics — pipeline may have failed.")

    if not metrics:
        return
    ret = metrics.get("return_pct", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    n_trades = int(metrics.get("num_trades", 0))
    wr = metrics.get("win_rate_pct", 0)
    dd = abs(metrics.get("max_drawdown_pct", 0))
    L.append(
        f"Application demo returned {ret:.1f}% over {n_trades} trades "
        f"with Sharpe {sharpe:.2f}, win rate {wr:.1f}%, "
        f"max drawdown {dd:.1f}%."
    )


def _exec_verdict(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    """One-paragraph ML-first overall assessment with synthesized verdict.

    Delegates to three rendering helpers that each handle one aspect:
    ML quality paragraph, synthesized verdict, and primary issue.

    Args:
        L: Output markdown lines.
        metrics: Backtest metrics dictionary.
        pred_stats: Preloaded prediction statistics.
    """
    if not pred_stats:
        if not metrics:
            return
        L.append("Prediction metrics are unavailable; only the application demo ran.")
        return

    _render_ml_quality_paragraph(L, pred_stats)
    _render_synthesized_verdict(L, pred_stats, metrics)
    _render_primary_issue(L, metrics, pred_stats)


def _config_table(L: list[str], config: Config) -> None:
    """Key hyperparameters in one table.

    Args:
        L: Output markdown lines list (mutated in-place).
        config: Application configuration.
    """
    rows = [
        ("Data", "symbol", str(config.data.symbol)),
        ("Data", "timeframe", config.data.timeframe),
        ("Validation", "method", config.validation.method),
    ]

    if config.validation.method == "sliding":
        rows.extend(
            [
                ("Validation", "window type", "bar-based walk-forward"),
                (
                    "Validation",
                    "train/test/step bars",
                    f"{config.validation.train_window_bars}/"
                    f"{config.validation.test_window_bars}/"
                    f"{config.validation.step_bars}",
                ),
                (
                    "Validation",
                    "purge/embargo bars",
                    f"{config.validation.purge_bars}/{config.validation.embargo_bars}",
                ),
                (
                    "Validation",
                    "min_train_bars",
                    str(config.validation.min_train_bars),
                ),
            ]
        )
    else:
        rows.extend(
            [
                (
                    "Split",
                    "train",
                    f"{config.splitting.train_start} → {config.splitting.train_end}",
                ),
                (
                    "Split",
                    "val",
                    f"{config.splitting.val_start} → {config.splitting.val_end}",
                ),
                (
                    "Split",
                    "test",
                    f"{config.splitting.test_start} → {config.splitting.test_end}",
                ),
                (
                    "Split",
                    "purge/embargo",
                    f"{config.splitting.purge_bars}/{config.splitting.embargo_bars}",
                ),
            ]
        )

    rows.extend(
        [
            (
                "Labels",
                "atr_mult / horizon",
                f"{config.labels.atr_tp_multiplier}/{config.labels.atr_sl_multiplier} / {config.labels.horizon_bars}",
            ),
            (
                "GRU",
                "hidden/layers/seq",
                f"{config.gru.hidden_size}/{config.gru.num_layers}/{config.gru.sequence_length}",
            ),
            (
                "GRU",
                "lr/dropout/epochs",
                f"{config.gru.learning_rate}/{config.gru.dropout}/{config.gru.epochs}",
            ),
            (
                "LGBM",
                "leaves/depth/lr",
                f"{config.model.num_leaves}/{config.model.max_depth}/{config.model.learning_rate}",
            ),
            (
                "LGBM",
                "estimators/subsample",
                f"{config.model.n_estimators}/{config.model.subsample}",
            ),
            ("LGBM", "feature_fraction", str(config.model.feature_fraction)),
            (
                "Backtest",
                "capital/leverage",
                f"${config.backtest.initial_capital:,.0f}/{config.backtest.leverage}:1",
            ),
            (
                "Backtest",
                "lots/conf_thr",
                f"{config.backtest.lots_per_trade}/{config.backtest.confidence_threshold}",
            ),
            (
                "Backtest",
                "stop/tp (ATR)",
                f"{config.backtest.atr_stop_multiplier}/{config.backtest.atr_tp_multiplier}",
            ),
            ("Seed", "random_seed", str(config.workflow.random_seed)),
        ]
    )

    L.append(_tbl_row("Section", "Parameter", "Value"))
    L.append(_tbl_row("-------", "---------", "-----"))
    for section, param, val in rows:
        L.append(_tbl_row(section, param, val))


def _compute_ece_numpy(
    proba: np.ndarray, labels: np.ndarray, n_bins: int = _ECE_N_BINS
) -> float:
    """Compute Expected Calibration Error (ECE) from NumPy arrays.

    Delegates to :func:`thesis.stage_6_reporting._calibration.expected_calibration_error`.

    Args:
        proba: Softmax probabilities with shape ``(N, C)``.
        labels: Ground-truth class indices with shape ``(N,)``.
        n_bins: Number of confidence bins (default matches ``_ECE_N_BINS``).

    Returns:
        ECE value (0.0 = perfectly calibrated).
    """
    n_classes = proba.shape[1]
    y_onehot = np.zeros((len(labels), n_classes), dtype=np.float64)
    for i, lbl in enumerate(labels):
        y_onehot[i, lbl] = 1.0
    return _calibration.expected_calibration_error(y_onehot, proba, n_bins=n_bins)


def _calibration_summary_text(config: Config) -> str | None:
    """Compute a one-paragraph calibration reliability note.

    Reads predicted probabilities and true labels from the predictions
    parquet file, computes ECE, and returns a human-readable summary
    of whether confidence scores appear calibrated.

    Args:
        config: Loaded runtime configuration.

    Returns:
        Calibration summary string, or ``None`` if the predictions file
        is unavailable or missing probability columns.
    """
    preds_path = Path(config.paths.predictions)
    if not preds_path.exists():
        return None

    proba_cols = [
        "pred_proba_class_minus1",
        "pred_proba_class_0",
        "pred_proba_class_1",
    ]
    try:
        df = pl.read_parquet(preds_path)
    except Exception:
        logger.warning(
            "Failed to load predictions for calibration check: %s",
            preds_path,
            exc_info=True,
        )
        return None

    if not all(c in df.columns for c in proba_cols):
        return None
    if "true_label" not in df.columns:
        return None

    proba = df.select(proba_cols).to_numpy()
    true_labels = df["true_label"].to_numpy()
    pred_labels = df["pred_label"].to_numpy() if "pred_label" in df.columns else None

    # Map label values (-1, 0, 1) → class indices (0, 1, 2)
    class_indices = np.zeros(len(true_labels), dtype=np.int64)
    class_indices[true_labels == 0] = 1
    class_indices[true_labels == 1] = 2

    ece = _compute_ece_numpy(proba, class_indices)

    # Full calibration suite from _calibration module
    calib_metrics = _calibration.compute_all_calibration_metrics(
        true_labels,
        pred_labels if pred_labels is not None else np.argmax(proba, axis=1) - 1,
        proba,
        classes=[-1, 0, 1],
    )
    brier = calib_metrics.get("brier_score", float("nan"))
    logloss = calib_metrics.get("log_loss", float("nan"))

    if ece < _ECE_WELL_CALIBRATED:
        quality = "well-calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            f"(ECE < {_ECE_WELL_CALIBRATED:.2f}). Predicted probabilities closely match observed frequencies."
        )
    elif ece < _ECE_MODERATELY_CALIBRATED:
        quality = "moderately calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            f"({_ECE_WELL_CALIBRATED:.2f} ≤ ECE < {_ECE_MODERATELY_CALIBRATED:.2f}). Probabilities are somewhat aligned with outcomes; "
            "the model may be slightly over- or under-confident in some bins."
        )
    else:
        quality = "poorly calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            f"(ECE ≥ {_ECE_MODERATELY_CALIBRATED:.2f}). Predicted probabilities do not reliably reflect true "
            "likelihoods. Consider temperature scaling or isotonic regression."
        )

    note += f" Brier score = {brier:.4f}, Log-loss = {logloss:.4f}."

    logger.info(
        "Calibration summary: ECE=%.4f, Brier=%.4f, LogLoss=%.4f (%s)",
        ece,
        brier,
        logloss,
        quality,
    )
    return note


def _accuracy_table(
    L: list[str], pred_stats: dict | None, config: Config | None = None
) -> None:
    """Model accuracy: exact + directional + per-class + calibration.

    Args:
        L: Output markdown lines.
        pred_stats: Preloaded prediction statistics.
        config: Optional application configuration for calibration check.
    """
    if not pred_stats:
        L.append("*Prediction data not found.*")
        return

    total = pred_stats["total"]
    acc = pred_stats["accuracy"]
    dir_acc = pred_stats.get("directional_accuracy", acc)
    dir_bl = pred_stats.get("directional_baseline", _DIRECTIONAL_BASELINE)
    maj_bl = pred_stats.get("majority_baseline", 0)
    acc_gap = acc - maj_bl

    L.append(_tbl_row("Metric", "Value", "Zone"))
    L.append(_tbl_row("------", "-----", "----"))
    L.append(_tbl_row("Samples", f"{total:,}", ""))
    L.append(_tbl_row("Exact Accuracy", f"{acc * 100:.1f}%", _zone("accuracy", acc)))
    L.append(
        _tbl_row(
            "Directional Acc.",
            f"{dir_acc * 100:.1f}%",
            _zone("directional_accuracy", dir_acc),
        )
    )
    L.append(_tbl_row("Dir. Baseline", f"{dir_bl * 100:.1f}%", ""))
    L.append(_tbl_row("Majority Baseline", f"{maj_bl * 100:.1f}%", ""))
    L.append(_tbl_row("Acc - Baseline", f"{acc_gap * 100:+.1f}pp", ""))
    L.append("")

    # Per-class
    per_class = pred_stats["per_class"]
    L.append(_tbl_row("Class", "Actual", "Pred", "Recall", "F1"))
    L.append(_tbl_row("-----", "------", "----", "------", "--"))
    for name in ("Short", "Hold", "Long"):
        pc = per_class[name]
        L.append(
            _tbl_row(
                name,
                f"{pc['true_count']:,}",
                f"{pc['pred_count']:,}",
                f"{pc['recall'] * 100:.1f}%",
                f"{pc['f1']:.3f}",
            )
        )
    L.append("")

    # Confidence filtering
    hc = pred_stats.get("high_confidence")
    if hc:
        hc_ratio = hc["count"] / total if total else 0.0
        L.append(
            f"High-confidence (≥{hc['threshold']:.0%}): "
            f"{hc['count']:,} samples ({hc_ratio * 100:.2f}%), "
            f"accuracy {hc['accuracy'] * 100:.1f}%, "
            f"dir. acc. {hc['directional_accuracy'] * 100:.1f}%"
        )
        L.append("")

    # Calibration reliability note (after confidence section)
    if config is not None:
        calib_note = _calibration_summary_text(config)
        if calib_note:
            L.append(calib_note)
            L.append("")


def _gru_summary(L: list[str], config: Config) -> None:
    """GRU architecture summary line (hybrid only — caller guards architecture)."""
    gru = config.gru
    L.append(
        f"GRU: input={gru.input_size}, hidden={gru.hidden_size}, "
        f"layers={gru.num_layers}, seq={gru.sequence_length}, "
        f"dropout={gru.dropout}, epochs≤{gru.epochs}, patience={gru.patience}"
    )
    L.append("")


def _feature_importance_table(L: list[str], feature_importance: dict) -> None:
    """Top-10 feature importance."""
    if not feature_importance:
        return
    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    gru_count = sum(1 for n, _ in items if n.startswith("gru_"))
    L.append(_tbl_row("Rank", "Feature", "Source", "Score"))
    L.append(_tbl_row("----", "-------", "------", "-----"))
    for i, (name, imp) in enumerate(items, 1):
        src = "GRU" if name.startswith("gru_") else "Technical"
        L.append(_tbl_row(str(i), f"`{name}`", src, f"{imp:.0f}"))
    L.append(
        f"Top-10: {gru_count}/{len(items)} GRU features ({gru_count / len(items) * 100:.0f}%)"
    )
    L.append("")


def _backtest_params_table(L: list[str], config: Config) -> None:
    """Backtest simulation parameters."""
    bc = config.backtest
    L.append(_tbl_row("Parameter", "Value"))
    L.append(_tbl_row("---------", "-----"))
    L.append(_tbl_row("Initial Capital", _fmt_dollar(bc.initial_capital)))
    L.append(_tbl_row("Leverage", f"{bc.leverage}:1"))
    L.append(_tbl_row("Lots/Trade", str(bc.lots_per_trade)))
    L.append(_tbl_row("ATR Stop", f"{bc.atr_stop_multiplier}x"))
    if bc.atr_tp_multiplier > 0:
        L.append(_tbl_row("ATR TP", f"{bc.atr_tp_multiplier}x"))
    L.append(_tbl_row("Confidence Thr.", str(bc.confidence_threshold)))
    L.append(
        _tbl_row(
            "Spread",
            f"${bc.spread_ticks * config.data.tick_size:.2f}",
        )
    )
    L.append(_tbl_row("Commission/lot", _fmt_dollar(bc.commission_per_lot)))
    L.append("")


def _backtest_metrics_table(L: list[str], metrics: dict, config: Config) -> None:
    """Core backtest metrics with zone indicators.

    Args:
        L: Output markdown lines list (mutated in-place).
        metrics: Backtest metrics dictionary.
        config: Application configuration (for initial capital display).
    """
    if not metrics:
        L.append("*No backtest results available.*")
        return

    rows = [
        ("Return", "return_pct", _fmt_pct),
        ("Sharpe", "sharpe_ratio", _fmt_f2),
        ("Max DD", "max_drawdown_pct", _fmt_pct),
        ("Win Rate", "win_rate_pct", _fmt_pct),
        ("Profit Factor", "profit_factor", _fmt_f2),
        ("Trades", "num_trades", lambda v: f"{int(v):,}"),
    ]
    L.append(_tbl_row("Metric", "Value", "Zone"))
    L.append(_tbl_row("------", "-----", "----"))
    for label, key, fmt in rows:
        val = metrics.get(key)
        if val is None:
            continue
        L.append(_tbl_row(label, fmt(val), _zone(key, val)))
    L.append("")

    initial = config.backtest.initial_capital
    eq_final = metrics.get("equity_final", 0)
    L.append(
        f"Initial balance: {_fmt_dollar(initial)} | Final equity: ${eq_final:,.0f}"
    )
    L.append("")


def _benchmark_comparison_table(L: list[str], metrics: dict, config: Config) -> None:
    """Compare benchmarks against the configured model architecture."""
    test_path = Path(config.paths.test_data)
    benchmarks = compute_benchmark_comparison(test_path, metrics, config)
    if not benchmarks:
        L.append("*Test data unavailable — benchmark comparison skipped.*")
        L.append("")
        return

    L.append(
        "*Benchmarks are rough directional references and are not "
        "trading-cost-equivalent to the CFD backtest strategy.*"
    )
    L.append(
        "*Note: Benchmarks exclude transaction costs (spread, slippage, "
        f"commission); not directly comparable to the {_model_label(config)} model "
        "which incurs all three.*"
    )
    L.append("")
    L.append(_tbl_row("Strategy", "Return", "Sharpe", "Max DD", "Win Rate", "Trades"))
    L.append(_tbl_row("--------", "------", "------", "-------", "--------", "------"))
    for b in benchmarks:
        ret = _fmt_pct(b["return_pct"])
        sharpe = _fmt_f2(b["sharpe"])
        dd = _fmt_pct(b["max_dd_pct"])
        wr = (
            "—"
            if np.isnan(b.get("win_rate_pct", float("nan")))
            else _fmt_pct(b["win_rate_pct"])
        )
        trades = str(b.get("num_trades", "—"))
        L.append(_tbl_row(b["strategy"], ret, sharpe, dd, wr, trades))

    valid_returns = [
        b for b in benchmarks if not np.isnan(b.get("return_pct", float("nan")))
    ]
    if valid_returns:
        best_ret = max(valid_returns, key=lambda x: x["return_pct"])
        best_sharpe = max(benchmarks, key=lambda x: x.get("sharpe", -999))
        best_dd = min(benchmarks, key=lambda x: x.get("max_dd_pct", 999))
        L.append("")
        L.append(
            f"Best return: **{best_ret['strategy']}** | "
            f"Best Sharpe: **{best_sharpe['strategy']}** | "
            f"Lowest DD: **{best_dd['strategy']}**"
        )
    L.append("")


_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
_SEVERITY_ICON = {"critical": "🔴", "warning": "🟡", "info": "✅"}
_PRIORITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "✅"}


def _render_issues(
    L: list[str],
    issues: list[tuple[str, str]],
    recs: list[tuple[str, str]],
) -> None:
    """Render sorted issues and recommendations into markdown lines.

    Args:
        L: Output markdown lines list (mutated in-place).
        issues: List of ``(severity, description)`` tuples,
            sorted by severity before rendering.
        recs: List of ``(priority, description)`` tuples,
            sorted by priority before rendering.
    """
    L.append("### Issues")
    L.append("")
    if not issues:
        L.append("*No issues detected.*")
    else:
        sorted_issues = sorted(issues, key=lambda x: _SEVERITY_ORDER.get(x[0], 9))
        for i, (severity, desc) in enumerate(sorted_issues, 1):
            icon = _SEVERITY_ICON.get(severity, "⚪")
            L.append(f"{i}. {icon} {desc}")
    L.append("")

    L.append("### Recommendations")
    L.append("")
    if not recs:
        L.append("*No specific recommendations.*")
    else:
        sorted_recs = sorted(recs, key=lambda x: _PRIORITY_ORDER.get(x[0], 9))
        for i, (priority, desc) in enumerate(sorted_recs, 1):
            icon = _PRIORITY_ICON.get(priority, "⚪")
            L.append(f"{i}. {icon} {desc}")


def _issues_list(
    L: list[str],
    metrics: dict,
    trades: list[dict],
    config: Config,
    pred_stats: dict | None,
) -> None:
    """High-signal issues and recommendations from report metrics.

    Only the most critical checks are included to keep the report focused.

    Args:
        L: Output markdown lines.
        metrics: Backtest metrics dictionary.
        trades: Backtest trades list.
        config: Loaded runtime configuration.
        pred_stats: Preloaded prediction statistics.
    """
    issues: list[tuple[str, str]] = []
    recs: list[tuple[str, str]] = []

    if not metrics:
        issues.append(("critical", "No backtest metrics — pipeline may have failed."))
        _render_issues(L, issues, recs)
        return

    sharpe = metrics.get("sharpe_ratio", 0)
    dd = abs(metrics.get("max_drawdown_pct", 0))
    pf = metrics.get("profit_factor", 0)
    n_trades = int(metrics.get("num_trades", 0))
    dir_acc = pred_stats.get("directional_accuracy", 0) if pred_stats else 0

    # — Core high-signal checks —

    if n_trades == 0:
        issues.append(
            (
                "critical",
                "Zero trades executed — model produces no actionable signals in test period.",
            )
        )

    if sharpe < 0:
        issues.append(
            (
                "critical",
                f"Sharpe {sharpe:.2f} is negative — strategy underperforms risk-free rate.",
            )
        )

    if dd > _ISSUE_DD_CATASTROPHIC:
        issues.append(
            (
                "critical",
                f"Max drawdown {dd:.1f}% > {_ISSUE_DD_CATASTROPHIC:.0f}% — catastrophic capital erosion.",
            )
        )

    if pf < _EDGE_PF_NEGATIVE:
        issues.append(
            (
                "critical",
                f"Profit factor {pf:.2f} < {_EDGE_PF_NEGATIVE:.1f} — strategy loses money on average.",
            )
        )

    if dir_acc > 0 and dir_acc < _QUALITY_DIR_ACC_FAIR:
        issues.append(
            (
                "critical",
                f"Directional accuracy {dir_acc:.1%} < {_QUALITY_DIR_ACC_FAIR:.0%} — model predicts worse than random.",
            )
        )

    if not issues:
        issues.append(("info", "No critical issues identified."))

    # — Single actionable recommendation —
    if not recs:
        recs.append(
            (
                "info",
                "Consider walk-forward validation for production readiness and robustness testing.",
            )
        )

    _render_issues(L, issues, recs)


# ---------------------------------------------------------------------------
# Chart helpers (formerly report/main.py)
# ---------------------------------------------------------------------------


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


def _build_equity_series(
    trades: list[dict], initial_capital: float
) -> tuple[list, list]:
    """Build timestamp and cumulative equity series from trades.

    Note:
        The equity curve is trade-by-trade (closed-trade PnL), not
        mark-to-market. Intra-trade drawdowns are not visible.
    """
    times = [pd.to_datetime(trades[0]["entry_time"])]
    equity = [initial_capital]
    for t in trades:
        times.append(pd.to_datetime(t["exit_time"]))
        equity.append(equity[-1] + t["pnl"])
    return times, equity


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


# ---------------------------------------------------------------------------
# Public entry point (formerly report/main.py → generate_report)
# ---------------------------------------------------------------------------


def generate_report(config: Config) -> None:
    """**Pipeline Stage 6 (of 6):** Generate thesis report with static charts and markdown.

    Args:
        config: Loaded application configuration.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )

    if config.paths.session_dir:
        out_dir = Path(config.paths.session_dir) / "reports"
    else:
        out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load backtest results
    bt_path = Path(config.paths.backtest_results)
    metrics: dict = {}
    trades: list[dict] = []
    if bt_path.exists():
        with console.status(f"[cyan]Loading backtest results[/] {bt_path}"):
            with open(bt_path) as f:
                bt = json.load(f)
            metrics = bt.get("metrics", {})
            trades = bt.get("trades", [])

    with console.status("[cyan]Rendering report charts[/]"):
        _plot_equity_curve(trades, config, out_dir)
        feature_importance = _load_feature_importance(config, out_dir)
        _plot_feature_importance(feature_importance, out_dir)
    # Markdown Report
    with console.status("[cyan]Building thesis markdown[/]"):
        pred_stats = _load_prediction_stats(Path(config.paths.predictions))
        model_comparison_rows = _build_model_comparison_rows(config, pred_stats)
        md = _build_markdown(
            config,
            metrics,
            trades,
            feature_importance,
            pred_stats,
        )
        model_eval_md = _build_model_evaluation_markdown(
            config, pred_stats, model_comparison_rows
        )
    report_path = Path(config.paths.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)
    logger.info("Report saved: %s", report_path)

    model_eval_path = out_dir / "model_evaluation.md"
    with model_eval_path.open("w") as f:
        f.write(model_eval_md)
    logger.info("Model evaluation saved: %s", model_eval_path)

    model_metrics_path = out_dir / "model_metrics.json"
    with model_metrics_path.open("w") as f:
        json.dump(pred_stats or {}, f, indent=2)
    logger.info("Model metrics saved: %s", model_metrics_path)

    model_cmp_csv, model_cmp_md = _write_model_comparison_artifacts(
        out_dir, model_comparison_rows
    )
    logger.info("Model comparison saved: %s, %s", model_cmp_csv, model_cmp_md)
