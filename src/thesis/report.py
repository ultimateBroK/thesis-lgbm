"""Report generation: markdown builder, statistics, charts, and orchestrator.

Merged from the former ``thesis.report`` package (``__init__``,
``main``, ``builder``, ``stats``).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from thesis.config import Config
from thesis.zones import _get_metric_zone

logger = logging.getLogger("thesis.report")

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
        dist: dict[str, tuple[int, float]] = {}
        for label_val, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            count = (df["label"] == label_val).sum()
            dist[name] = (count, count / total * 100 if total > 0 else 0)
        dist["total"] = total
        return dist
    except Exception:
        return None


def _load_prediction_stats(preds_path: Path) -> dict | None:
    """Compute prediction quality statistics from a predictions parquet file.

    Args:
        preds_path: Path to predictions parquet containing ``true_label``,
            ``pred_label``, and optional class-probability columns.

    Returns:
        A dictionary with overall accuracy, directional accuracy, baselines,
        per-class metrics, confusion matrix, and optional high-confidence stats;
        returns ``None`` if the file is unavailable or unreadable.
    """
    if not preds_path.exists():
        return None
    try:
        cols = ["true_label", "pred_label"]
        proba_cols = [
            "pred_proba_class_minus1",
            "pred_proba_class_0",
            "pred_proba_class_1",
        ]
        # Try loading with probability columns
        try:
            df = pl.read_parquet(preds_path)
        except Exception:
            df = pl.read_parquet(preds_path, columns=cols)

        true = df["true_label"].to_numpy()
        pred = df["pred_label"].to_numpy()
        total = len(true)

        # Overall accuracy: fraction of predictions matching true labels
        accuracy = float((true == pred).mean())
        # Majority baseline: accuracy if we always predict the most common class
        majority_baseline = float(max((true == lv).sum() for lv in [-1, 0, 1]) / total)

        # Directional accuracy: evaluate only on non-Hold predictions
        non_hold_mask = (true != 0) & (pred != 0)
        if non_hold_mask.sum() > 0:
            directional_correct = true[non_hold_mask] == pred[non_hold_mask]
            directional_accuracy = float(directional_correct.mean())
            directional_baseline = 0.5
        else:
            directional_accuracy = 0.0
            directional_baseline = 0.5

        # Per-class metrics
        per_class: dict = {}
        for lv, ln in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            true_mask = true == lv
            pred_mask = pred == lv
            recall = float((pred[true_mask] == lv).mean()) if true_mask.sum() > 0 else 0
            precision = (
                float((true[pred_mask] == lv).mean()) if pred_mask.sum() > 0 else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            per_class[ln] = {
                "true_count": int(true_mask.sum()),
                "pred_count": int(pred_mask.sum()),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }

        # Confusion matrix
        cm: dict = {}
        for true_lv, true_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            row: dict = {}
            for pred_lv, pred_name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
                row[pred_name] = int(((true == true_lv) & (pred == pred_lv)).sum())
            cm[true_name] = row

        result: dict = {
            "total": total,
            "accuracy": accuracy,
            "directional_accuracy": directional_accuracy,
            "directional_baseline": directional_baseline,
            "majority_baseline": majority_baseline,
            "per_class": per_class,
            "confusion_matrix": cm,
        }

        # Confidence-filtered accuracy
        has_proba = all(c in df.columns for c in proba_cols)
        if has_proba:
            proba = df.select(proba_cols).to_numpy()
            max_proba = proba.max(axis=1)
            threshold = 0.70
            hc_mask = max_proba >= threshold
            if hc_mask.sum() > 0:
                hc_acc = float((true[hc_mask] == pred[hc_mask]).mean())
                hc_total = int(hc_mask.sum())
                non_hold = pred[hc_mask] != 0
                if non_hold.sum() > 0:
                    dir_acc = float(
                        (true[hc_mask][non_hold] == pred[hc_mask][non_hold]).mean()
                    )
                else:
                    dir_acc = 0
                result["high_confidence"] = {
                    "threshold": threshold,
                    "count": hc_total,
                    "pct_of_total": hc_total / total * 100,
                    "accuracy": hc_acc,
                    "directional_accuracy": dir_acc,
                }

        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Benchmark comparison helpers (formerly report/stats.py)
# ---------------------------------------------------------------------------

_BARS_PER_YEAR = 252 * 24


def _annualized_sharpe(
    returns: np.ndarray, bars_per_year: int = _BARS_PER_YEAR
) -> float:
    """Compute annualized Sharpe ratio from bar returns."""
    std = float(np.std(returns, ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(bars_per_year))


def _max_drawdown_pct(equity: np.ndarray) -> float:
    """Compute maximum drawdown as a percentage from an equity curve."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    return float(abs(dd.min()))


def _build_equity_curve(
    returns: np.ndarray,
    initial_capital: float,
) -> np.ndarray:
    """Build equity curve from bar returns and initial capital."""
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
    """Simulate a random long/short signal strategy."""
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
    # 1. Try static test split first
    if test_data_path.exists():
        try:
            df = pl.read_parquet(test_data_path, columns=["close"])
            return df["close"].to_numpy()
        except Exception:
            logger.warning("Failed to load test data for benchmarks")

    # 2. Walk-forward fallback: load OHLCV and filter to backtest period
    ohlcv_path = Path(config.paths.ohlcv)
    if not ohlcv_path.exists():
        logger.warning("No OHLCV data available for benchmark fallback: %s", ohlcv_path)
        return None

    try:
        df = pl.read_parquet(ohlcv_path)
    except Exception:
        logger.warning("Failed to load OHLCV for benchmarks")
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
    if config.model.architecture == "stacking":
        return "True Stacking (GRU + LGBM -> Meta LGBM)"
    return "Hybrid GRU + LightGBM"


def _build_markdown(
    config: Config,
    metrics: dict,
    trades: list[dict],
    feature_importance: dict,
    ablation: dict,
    pred_stats: dict | None,
) -> str:
    """Build concise metrics-first markdown report.

    Args:
        config: Loaded runtime configuration.
        metrics: Backtest metrics dictionary.
        trades: Backtest trades list.
        feature_importance: Feature importance values.
        ablation: Ablation-study summary.
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

    # -- Model Performance --
    L.append("## Model Performance")
    L.append("")
    _accuracy_table(L, pred_stats)
    _gru_summary(L, config)
    _stacking_summary(L, config)
    _feature_importance_table(L, feature_importance)
    L.append("")

    # -- Backtest Results --
    L.append("## Backtest Results")
    L.append("")
    _backtest_params_table(L, config)
    _backtest_metrics_table(L, metrics)
    _trade_stats(L, trades, metrics)
    L.append("")

    # -- Benchmark Comparison --
    L.append("## Benchmark Comparison")
    L.append("")
    _benchmark_comparison_table(L, metrics, config)

    # -- Issues & Recommendations --
    L.append("## Issues & Recommendations")
    L.append("")
    _issues_list(L, metrics, trades, config, pred_stats)
    L.append("")

    return "\n".join(L)


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


def _exec_verdict(L: list[str], metrics: dict, pred_stats: dict | None) -> None:
    """One-paragraph ML-first overall assessment.

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

    acc = pred_stats["accuracy"]
    baseline = pred_stats["majority_baseline"]
    dir_acc = pred_stats["directional_accuracy"]
    per_class = pred_stats["per_class"]
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in per_class]))

    gap = acc - baseline
    if gap < 0:
        ml_quality = "weak"
        gate_msg = "Model is below majority baseline; predictive edge is not validated."
    elif acc > baseline + 0.05 and dir_acc > 0.55 and macro_f1 >= 0.45:
        ml_quality = "strong"
        gate_msg = "Model is above baseline with directional edge."
    elif dir_acc >= 0.50:
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


def _config_table(L: list[str], config: Config) -> None:
    """Key hyperparameters in one table."""
    rows = [
        ("Data", "symbol", str(config.data.symbol)),
        ("Data", "timeframe", config.data.timeframe),
        (
            "Split",
            "train",
            f"{config.splitting.train_start} → {config.splitting.train_end}",
        ),
        ("Split", "val", f"{config.splitting.val_start} → {config.splitting.val_end}"),
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
        (
            "Labels",
            "atr_mult / horizon",
            f"{config.labels.atr_multiplier} / {config.labels.horizon_bars}",
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
    L.append(_tbl_row("Section", "Parameter", "Value"))
    L.append(_tbl_row("-------", "---------", "-----"))
    for section, param, val in rows:
        L.append(_tbl_row(section, param, val))


def _accuracy_table(L: list[str], pred_stats: dict | None) -> None:
    """Model accuracy: exact + directional + per-class.

    Args:
        L: Output markdown lines.
        pred_stats: Preloaded prediction statistics.
    """
    if not pred_stats:
        L.append("*Prediction data not found.*")
        return

    total = pred_stats["total"]
    acc = pred_stats["accuracy"]
    dir_acc = pred_stats.get("directional_accuracy", acc)
    dir_bl = pred_stats.get("directional_baseline", 0.5)
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


def _gru_summary(L: list[str], config: Config) -> None:
    """GRU architecture summary line."""
    gru = config.gru
    L.append(
        f"GRU: input={gru.input_size}, hidden={gru.hidden_size}, "
        f"layers={gru.num_layers}, seq={gru.sequence_length}, "
        f"dropout={gru.dropout}, epochs≤{gru.epochs}, patience={gru.patience}"
    )
    L.append("")


def _stacking_summary(L: list[str], config: Config) -> None:
    """Add a compact stacking summary when the session used true stacking."""
    if config.model.architecture != "stacking":
        return

    wf_path = Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
    if not wf_path.exists():
        L.append("Stacking: base models = GRU + LightGBM, meta learner = LightGBM.")
        L.append("")
        return

    try:
        history = json.loads(wf_path.read_text())
    except Exception:
        L.append("Stacking: base models = GRU + LightGBM, meta learner = LightGBM.")
        L.append("")
        return

    L.append(
        "Stacking: base models = GRU + LightGBM, "
        f"meta learner = LightGBM, "
        f"base OOF rows = {history.get('base_oof_rows', 0):,}, "
        f"meta OOF rows = {history.get('meta_oof_rows', 0):,}, "
        f"meta warmup skips = {history.get('skipped_meta_folds', 0)}."
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


def _backtest_metrics_table(L: list[str], metrics: dict) -> None:
    """Full backtest metrics with zone indicators."""
    if not metrics:
        L.append("*No backtest results available.*")
        return

    rows = [
        ("Return", "return_pct", _fmt_pct),
        ("Ann. Return", "return_ann_pct", _fmt_pct),
        ("Sharpe", "sharpe_ratio", _fmt_f2),
        ("Sortino", "sortino_ratio", _fmt_f2),
        ("Max DD", "max_drawdown_pct", _fmt_pct),
        ("Win Rate", "win_rate_pct", _fmt_pct),
        ("Profit Factor", "profit_factor", _fmt_f2),
        ("Recovery Factor", "recovery_factor", _fmt_f2),
        ("SQN", "sqn", _fmt_f2),
        ("Avg Trade %", "avg_trade_pct", _fmt_pct),
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

    eq_final = metrics.get("equity_final", 0)
    eq_peak = metrics.get("equity_peak", 0)
    L.append(f"Equity: ${eq_final:,.0f} (peak ${eq_peak:,.0f})")
    L.append("")


def _trade_stats(L: list[str], trades: list[dict], metrics: dict) -> None:
    """Trade-level statistics."""
    if not trades:
        return

    pnls = [t["pnl"] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    longs = [t for t in trades if t.get("direction") == "long"]
    shorts = [t for t in trades if t.get("direction") == "short"]

    L.append(_tbl_row("Stat", "Value"))
    L.append(_tbl_row("----", "-----"))
    L.append(_tbl_row("Total Trades", str(len(trades))))
    L.append(_tbl_row("Long / Short", f"{len(longs)} / {len(shorts)}"))
    L.append(
        _tbl_row(
            "Winners / Losers",
            f"{len(winners)} / {len(losers)}",
        )
    )
    if winners:
        L.append(_tbl_row("Avg Win", _fmt_dollar(np.mean(winners))))
        L.append(_tbl_row("Best Trade", _fmt_dollar(max(winners))))
    if losers:
        L.append(_tbl_row("Avg Loss", _fmt_dollar(np.mean(losers))))
        L.append(_tbl_row("Worst Trade", _fmt_dollar(min(losers))))
    if winners and losers:
        rr = abs(np.mean(winners) / np.mean(losers))
        L.append(_tbl_row("Win/Loss Ratio", f"{rr:.1f}:1"))

    consec = _max_consecutive_losses(pnls)
    L.append(_tbl_row("Max Consec. Losses", str(consec)))
    L.append("")


def _max_consecutive_losses(pnls: list[float]) -> int:
    """Compute max consecutive losing trades."""
    max_streak = 0
    streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _benchmark_comparison_table(L: list[str], metrics: dict, config: Config) -> None:
    """Compare 4 strategies: Buy & Hold, Always Long, Random, Hybrid."""
    test_path = Path(config.paths.test_data)
    benchmarks = compute_benchmark_comparison(test_path, metrics, config)
    if not benchmarks:
        L.append("*Test data unavailable — benchmark comparison skipped.*")
        L.append("")
        return

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
    """Render sorted issues and recommendations into markdown lines."""
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


def _count_features(config: Config) -> int:
    """Count total features from the features parquet or GRU config."""
    features_path = Path(config.paths.features)
    if features_path.exists():
        try:
            import polars as pl  # noqa: F811

            df = pl.read_parquet(features_path)
            exclude = {"timestamp", "open", "high", "low", "close", "volume", "label"}
            return sum(1 for c in df.columns if c not in exclude)
        except Exception:
            pass
    return config.gru.hidden_size + len(config.features.static_feature_cols)


def _issues_list(
    L: list[str],
    metrics: dict,
    trades: list[dict],
    config: Config,
    pred_stats: dict | None,
) -> None:
    """Identify issues and recommendations from report metrics.

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
    ret = metrics.get("return_pct", 0)
    wr = metrics.get("win_rate_pct", 0)
    sqn = metrics.get("sqn", 0)
    recovery = metrics.get("recovery_factor", 0)
    exposure = metrics.get("exposure_time_pct", 0)
    avg_trade = metrics.get("avg_trade_pct", 0)

    dir_acc = pred_stats.get("directional_accuracy", 0) if pred_stats else 0

    label_dist = _load_label_distribution(Path(config.paths.labels))
    hold_pct = 0.0
    if label_dist and "Hold" in label_dist:
        _, hold_pct = label_dist["Hold"]

    feature_count = _count_features(config)

    sortino = metrics.get("sortino_ratio", 0)

    # =========================================================================
    # Issues — threshold-based
    # =========================================================================

    if sharpe < 0:
        issues.append(
            (
                "critical",
                f"Sharpe {sharpe:.2f} is negative — strategy underperforms risk-free rate.",
            )
        )
    elif sharpe < 0.5:
        issues.append(
            (
                "critical",
                f"Sharpe {sharpe:.2f} < 0.5 — model does not generate risk-adjusted returns. "
                "Consider increasing feature set or adjusting labeling parameters.",
            )
        )
    elif sharpe < 1.0:
        issues.append(
            (
                "warning",
                f"Sharpe {sharpe:.2f} < 1.0 — risk-adjusted returns below professional threshold.",
            )
        )

    if dd > 50:
        issues.append(
            (
                "critical",
                f"Max drawdown {dd:.1f}% > 50% — catastrophic capital erosion. "
                "Strategy viability is questionable.",
            )
        )
    elif dd > 30:
        issues.append(
            (
                "critical",
                f"Max drawdown {dd:.1f}% exceeds 30% threshold. "
                "Add position sizing rules or circuit breaker.",
            )
        )
    elif dd > 20:
        issues.append(
            (
                "warning",
                f"Max drawdown {dd:.1f}% > 20% — elevated drawdown for CFD trading.",
            )
        )

    if dir_acc > 0 and dir_acc < 0.50:
        issues.append(
            (
                "critical",
                f"Directional accuracy {dir_acc:.1%} < 50% — model predicts worse than random. "
                "Check label distribution and GRU training convergence.",
            )
        )
    elif dir_acc > 0 and dir_acc < 0.55:
        issues.append(
            (
                "warning",
                f"Directional accuracy {dir_acc:.1%} < 55% — model does not reliably predict direction. "
                "Check label distribution and GRU training convergence.",
            )
        )

    if np.isnan(wr):
        issues.append(
            (
                "critical",
                "Win rate is NaN — zero trades or calculation error.",
            )
        )
    elif wr < 30:
        issues.append(
            (
                "critical",
                f"Win rate {wr:.1f}% < 30% — requires exceptional risk/reward ratio to be viable.",
            )
        )
    elif wr < 40:
        issues.append(
            (
                "warning",
                f"Win rate {wr:.1f}% < 40% — below trading viability. "
                "Review stop-loss/take-profit ratio.",
            )
        )

    if pf < 1.0:
        issues.append(
            (
                "critical",
                f"Profit factor {pf:.2f} < 1.0 — strategy loses money on average.",
            )
        )
    elif pf < 1.2:
        issues.append(
            (
                "warning",
                f"Profit factor {pf:.2f} < 1.2 — barely covers transaction costs.",
            )
        )
    elif pf < 1.5:
        issues.append(
            (
                "warning",
                f"Profit factor {pf:.2f} < 1.5 — indicates marginal edge. "
                "Tighten confidence threshold to filter low-conviction trades.",
            )
        )

    if n_trades == 0:
        issues.append(
            (
                "critical",
                "Zero trades executed — model produces no actionable signals in test period.",
            )
        )
    elif n_trades < 30:
        issues.append(
            (
                "critical",
                f"Only {n_trades} trades — statistically unreliable results.",
            )
        )
    elif n_trades < 100:
        issues.append(
            (
                "warning",
                f"{n_trades} trades — marginal sample size for statistical significance.",
            )
        )

    if ret > 500:
        issues.append(
            (
                "warning",
                f"Return {ret:.0f}% suspiciously high — verify for overfitting or data leakage.",
            )
        )
    elif ret < -50:
        issues.append(
            (
                "critical",
                f"Return {ret:.0f}% — severe capital loss. Strategy is destroying value.",
            )
        )

    if recovery < 1.0:
        issues.append(
            (
                "warning",
                f"Recovery factor {recovery:.2f} < 1.0 — strategy never recovered from worst drawdown.",
            )
        )

    if sqn < 1.0 and n_trades > 0:
        issues.append(
            (
                "warning",
                f"SQN {sqn:.2f} < 1.0 — system quality suggests no reliable edge.",
            )
        )

    if avg_trade < 0:
        issues.append(
            (
                "warning",
                f"Average trade {avg_trade:.2f}% is negative — expected value per trade is a loss.",
            )
        )

    if config.backtest.leverage > 20:
        issues.append(
            (
                "warning",
                f"Leverage {config.backtest.leverage}:1 amplifies both returns and drawdowns.",
            )
        )

    if exposure > 0 and exposure < 10:
        issues.append(
            (
                "warning",
                f"Market exposure {exposure:.1f}% < 10% — model is overly selective, may miss opportunities.",
            )
        )

    if not issues:
        issues.append(("info", "No critical issues identified."))

    # =========================================================================
    # Recommendations — prioritized
    # =========================================================================

    if feature_count < 20:
        recs.append(
            (
                "high",
                f"Feature set has {feature_count} features (< 20). "
                "Add more features: order flow, cross-asset correlations, microstructure metrics.",
            )
        )
    elif feature_count < 30:
        recs.append(
            (
                "medium",
                f"Feature set has {feature_count} features. "
                "Consider adding order flow imbalance or cross-asset features for additional signal.",
            )
        )

    if hold_pct > 50:
        recs.append(
            (
                "high",
                f"Hold labels are {hold_pct:.1f}% (> 50%). "
                f"Reduce ATR multiplier (current: {config.labels.atr_multiplier}) "
                "for tighter barriers to generate more directional signals.",
            )
        )
    elif hold_pct > 40:
        recs.append(
            (
                "medium",
                f"Hold labels are {hold_pct:.1f}% (> 40%). "
                f"Consider reducing ATR multiplier (current: {config.labels.atr_multiplier}) "
                "for tighter barriers.",
            )
        )

    if sharpe < 2.0:
        recs.append(
            (
                "medium",
                "Tune confidence threshold to filter low-conviction trades and improve risk-adjusted returns.",
            )
        )

    if dd > 20:
        recs.append(
            (
                "high",
                "Add position sizing or circuit breaker to limit drawdowns. "
                "Consider Kelly criterion for optimal sizing.",
            )
        )

    if n_trades < 100:
        recs.append(
            (
                "medium",
                "Lower confidence threshold or expand test period for more trades "
                "to improve statistical reliability.",
            )
        )

    if pf < 1.5:
        recs.append(
            (
                "high",
                "Optimize take-profit/stop-loss ratio. "
                f"Current ATR stop={config.backtest.atr_stop_multiplier}x, "
                f"ATR TP={config.backtest.atr_tp_multiplier}x.",
            )
        )

    if config.gru.epochs < 20:
        recs.append(
            (
                "medium",
                f"GRU max epochs set to {config.gru.epochs} (< 20). "
                "Model may not converge. Increase epochs or review learning rate.",
            )
        )

    if sortino > 0 and sharpe > 0 and (sortino / sharpe) < 1.2:
        recs.append(
            (
                "low",
                "Sortino/Sharpe ratio close to 1.0 — upside and downside volatility similar. "
                "Strategy does not effectively limit downside risk.",
            )
        )

    if wr < 45 and pf < 1.5:
        recs.append(
            (
                "high",
                f"Low win rate ({wr:.1f}%) combined with low profit factor ({pf:.2f}). "
                "Fundamental strategy review recommended — check signal quality and trade execution.",
            )
        )

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
    """Render and save an equity curve image from trade history."""
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
    """Build timestamp and cumulative equity series from trades."""
    times = [pd.to_datetime(trades[0]["entry_time"])]
    equity = [initial_capital]
    for t in trades:
        times.append(pd.to_datetime(t["exit_time"]))
        equity.append(equity[-1] + t["pnl"])
    return times, equity


def _plot_feature_importance(feature_importance: dict, out_dir: Path) -> None:
    """Render and save a top-20 feature-importance chart."""
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
    """Load feature-importance JSON from session report outputs."""
    fi_path = (
        Path(config.paths.session_dir) / "reports" / "feature_importance.json"
        if config.paths.session_dir
        else out_dir.parent / "feature_importance.json"
    )
    if not fi_path.exists():
        return {}
    with open(fi_path) as f:
        return json.load(f)


def _load_ablation_results(config: Config) -> dict:
    """Load ablation-study results for the current session."""
    if not config.paths.session_dir:
        return {}
    abl_path = Path(config.paths.session_dir) / "reports" / "ablation_results.json"
    if not abl_path.exists():
        return {}
    with open(abl_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public entry point (formerly report/main.py → generate_report)
# ---------------------------------------------------------------------------


def generate_report(config: Config) -> None:
    """Generate thesis report with static charts and markdown.

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
        with open(bt_path) as f:
            bt = json.load(f)
        metrics = bt.get("metrics", {})
        trades = bt.get("trades", [])

    _plot_equity_curve(trades, config, out_dir)
    feature_importance = _load_feature_importance(config, out_dir)
    _plot_feature_importance(feature_importance, out_dir)
    ablation = _load_ablation_results(config)

    # Markdown Report
    pred_stats = _load_prediction_stats(Path(config.paths.predictions))
    md = _build_markdown(
        config,
        metrics,
        trades,
        feature_importance,
        ablation,
        pred_stats,
    )
    report_path = Path(config.paths.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(md)
    logger.info("Report saved: %s", report_path)
