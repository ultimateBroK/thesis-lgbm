"""OOF vs OOS generalization check section renderer.

Renders the out-of-fold vs out-of-sample comparison table, aggregating
walk-forward cross-validation results against held-out test-period metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.stage_6_reporting.comparison import _parse_date
from thesis.stage_6_reporting.sections.data import _tbl_row

logger = logging.getLogger("thesis.report")


def _render_oof_vs_oos_section(L: list[str], config: Config) -> None:
    """Render OOF vs OOS comparison section with side-by-side metrics table."""
    # ── Load walk-forward history ──
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
    except (OSError, json.JSONDecodeError):
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

    # ── Aggregate OOF metrics across windows (weighted by test_rows) ──
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
        oof_class_f1: dict[str, float | None] = {}
        for cls_key in ("-1", "0", "1"):
            sup = class_support.get(cls_key, 0)
            oof_class_f1[cls_key] = (
                weighted_class_f1[cls_key] / sup if sup > 0 else None
            )

    # ── Compute OOS metrics from predictions filtered to test period ──
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
                        except (pl.ComputeError, ValueError):
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
            except (pl.ColumnNotFoundError, ValueError, pl.ComputeError):
                logger.warning(
                    "Failed to compute OOS prediction metrics", exc_info=True
                )

    # ── Render table ──
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

    # ── Interpretive note ──
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
