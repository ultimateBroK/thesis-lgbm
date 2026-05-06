"""Hybrid-vs-static statistical comparison and model-comparison helpers.

Provides paired t-test comparison between hybrid and static architecture
sessions by matching walk-forward windows on overlapping test date ranges,
plus thesis-level model comparison row builders and artifact writers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from thesis.shared.config import Config
from thesis.stage_4_training import baselines as baselines_mod
from thesis.stage_6_reporting.benchmarks import _model_label

logger = logging.getLogger("thesis.report")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MIN_WINDOWS_COMPARISON: int = 3
_SIGNIFICANCE_ALPHA: float = 0.05
_MAX_PER_WINDOW_DISPLAY: int = 10

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _tbl_row(*cells: str) -> str:
    """Format cells as a markdown table row."""
    return "| " + " | ".join(cells) + " |"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Window pairing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


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
        except (OSError, ValueError):
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


# ---------------------------------------------------------------------------
# Public comparison renderer
# ---------------------------------------------------------------------------


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
    except (OSError, json.JSONDecodeError):
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
    except (ValueError, TypeError):
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


# ---------------------------------------------------------------------------
# Thesis-level model comparison rows & artifacts
# ---------------------------------------------------------------------------


def _build_model_comparison_rows(
    config: Config, pred_stats: dict | None
) -> list[dict[str, Any]]:
    """Build thesis-level model comparison rows with available metrics.

    Rows include directional accuracy, accuracy, macro F1, long/short F1, and
    optional regression metrics.
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
            baselines = baselines_mod.run_all_baselines(
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
        except (pl.ColumnNotFoundError, ValueError):
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
            "Primary focus: Directional Accuracy, Accuracy, Macro F1,"
            " and per-class F1.  Rows with empty values require"
            " additional experiment runs.\n\n"
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
