"""Markdown table builders and verdict logic for the thesis report.

Each function appends markdown lines to a caller-provided list ``L``.
Used by ``generation.py``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.zones import _get_metric_zone
from thesis.stage_6_reporting import calibration
from thesis.stage_6_reporting.benchmarks import (
    _model_label,
    compute_benchmark_comparison,
)
from thesis.stage_6_reporting.md_format import _fmt_dollar, _fmt_f2, _fmt_pct, _tbl_row
from thesis.stage_6_reporting.sections import (
    _assess_model_quality,
    _assess_trading_edge,
    _derive_recommendation,
    _identify_primary_issue,
    _render_issues,
    _render_ml_quality_paragraph,
    _render_primary_issue,
    _render_synthesized_verdict,
)

logger = logging.getLogger("thesis.report")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Confidence & baseline
_DIRECTIONAL_BASELINE: float = 0.5

# Expected Calibration Error (ECE)
_ECE_N_BINS: int = 10
_ECE_WELL_CALIBRATED: float = 0.05
_ECE_MODERATELY_CALIBRATED: float = 0.15

# Zone emoji mapping
_ZONE_EMOJI = {
    "excellent": "✅",
    "good": "🟢",
    "moderate": "🟡",
    "poor": "🟠",
    "dangerous": "🔴",
}

# ---------------------------------------------------------------------------
# Zone helper
# ---------------------------------------------------------------------------


def _zone(key: str, value: float) -> str:
    """Zone emoji for a metric value."""
    if value is None or (
        isinstance(value, float)
        and (math.isnan(value) if isinstance(value, float) else False)
    ):
        return "⚪"
    color, _, _ = _get_metric_zone(key, value)
    return _ZONE_EMOJI.get(color, "⚪")


# ---------------------------------------------------------------------------
# Calibration helpers (used by _accuracy_table)
# ---------------------------------------------------------------------------


def _compute_ece_numpy(
    proba: np.ndarray, labels: np.ndarray, n_bins: int = _ECE_N_BINS
) -> float:
    """Compute Expected Calibration Error (ECE) from NumPy arrays.

    Delegates to
    :func:`thesis.stage_6_reporting.calibration.expected_calibration_error`.

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
    return calibration.expected_calibration_error(y_onehot, proba, n_bins=n_bins)


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
        df = pl.read_csv(preds_path)
    except (pl.exceptions.ComputeError, OSError):
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
    calib_metrics = calibration.compute_all_calibration_metrics(
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
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are"
            f" **{quality}** (ECE < {_ECE_WELL_CALIBRATED:.2f})."
            " Predicted probabilities closely match observed frequencies."
        )
    elif ece < _ECE_MODERATELY_CALIBRATED:
        quality = "moderately calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            f"({_ECE_WELL_CALIBRATED:.2f} ≤ ECE"
            f" < {_ECE_MODERATELY_CALIBRATED:.2f}). Probabilities are"
            " somewhat aligned with outcomes; the model may be slightly"
            " over- or under-confident in some bins."
        )
    else:
        quality = "poorly calibrated"
        note = (
            f"**Calibration**: ECE = {ece:.4f} — confidence scores are **{quality}** "
            f"(ECE ≥ {_ECE_MODERATELY_CALIBRATED:.2f}). Predicted"
            " probabilities do not reliably reflect true likelihoods."
            " Consider temperature scaling or isotonic regression."
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


# ---------------------------------------------------------------------------
# Table builders
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


def _exec_verdict(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    """One-paragraph ML-first overall assessment with synthesized verdict.

    Delegates to rendering helpers for ML quality, synthesized verdict, and
    primary issue text.

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
                f"{config.labels.barrier_atr_multiplier}"
                f" / {config.labels.horizon_bars}",
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
        f"Top-10: {gru_count}/{len(items)} GRU features"
        f" ({gru_count / len(items) * 100:.0f}%)"
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


def _issues_list(
    L: list[str],
    metrics: dict,
    trades: list[dict],
    config: Config,
    pred_stats: dict | None,
) -> None:
    """High-signal issues and recommendations from report metrics.

    Delegates to assess module functions for consistent logic with verdict
    section. Only the most critical checks are included to keep the report
    focused.

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

    primary = _identify_primary_issue(metrics, pred_stats)
    if primary:
        issues.append(("critical", primary))

    model_quality, _ = _assess_model_quality(pred_stats) if pred_stats else ("POOR", "")
    trading_edge, _ = _assess_trading_edge(metrics)
    recommendation = _derive_recommendation(model_quality, trading_edge, metrics)

    if recommendation.startswith("NOT DEPLOYABLE"):
        recs.append(("high", f"Fix root causes before deployment: {recommendation}."))
    elif recommendation.startswith("DEPLOYABLE with caution"):
        recs.append(("medium", f"Edge is marginal — {recommendation}."))
    else:
        recs.append(("info", recommendation))

    if not issues:
        issues.append(("info", "No critical issues identified."))

    _render_issues(L, issues, recs)
