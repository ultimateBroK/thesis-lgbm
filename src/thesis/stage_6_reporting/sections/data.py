"""Data-quality and methodology section renderers.

Renderers for data quality, label design, validation methodology, and
auxiliary regression metrics.  Each function appends markdown lines to a
caller-provided list ``L``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

from thesis.shared.config import Config
from thesis.stage_6_reporting import data_quality
from thesis.stage_6_reporting.md_format import (
    _fmt_f2,  # noqa: F401
    _fmt_pct,  # noqa: F401
    _tbl_row,
)

logger = logging.getLogger("thesis.report")

# ---------------------------------------------------------------------------
# Support functions
# ---------------------------------------------------------------------------


def _load_label_distribution(labels_path: Path) -> dict | None:
    """Compute class distribution from the labels parquet file."""
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
    except (pl.ComputeError, OSError):
        logger.warning(
            "Failed to load label distribution: %s", labels_path, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_data_quality_section(L: list[str], config: Config) -> None:
    """Render the Data Quality analysis section from the JSON sidecar."""
    dq_path = Path(config.paths.data_quality_json)
    if not dq_path.exists():
        L.append("*Data quality JSON not found — stage 1 may not have run.*")
        L.append("")
        return

    try:
        with open(dq_path) as f:
            dq = json.load(f)
    except (OSError, json.JSONDecodeError):
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
            computed_dq = data_quality.compute_data_quality_report(ohlcv_df)
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
        except (pl.ComputeError, pl.ColumnNotFoundError, ValueError):
            logger.warning("Failed to compute data quality from OHLCV", exc_info=True)


def _render_label_design_section(L: list[str], config: Config) -> None:
    """Render the Label Design & Methodology explanation section."""
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
    """Render the Validation Methodology section (walk-forward, purge/embargo)."""
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
            f"{val_cfg.train_window_bars:,} bars"
            f" (~{val_cfg.train_window_bars // 8760}y)",
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
    """Render auxiliary regression metrics section (MAE/RMSE/R²) if available."""
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
