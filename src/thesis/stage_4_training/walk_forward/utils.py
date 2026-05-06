"""Shared walk-forward utility functions used by both hybrid and static paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config

logger = logging.getLogger("thesis.pipeline")

# --- Constants ---

_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)

# --- Confidence & Signal Quality Thresholds ---
_HIGH_CONFIDENCE_THRESHOLD = 0.70  # High-confidence prediction floor
_SHORT_BIAS_RATIO_THRESHOLD = 0.5  # LONG/SHORT ratio warning trigger
_GRU_SIGNAL_F_SCORE_THRESHOLD = 0.5  # Mean F-score below → no detectable signal

# --- Minimum Sample Thresholds ---
_ANOVA_MIN_SAMPLES_PER_CLASS = 2  # Minimum samples per class for ANOVA F-statistic

# --- Display / Logging ---
_SIGNAL_QUALITY_TOP_N = 5  # Top-N for GRU signal quality logging


# --- Feature selection ---


def _select_static_feature_cols(
    config: Config,
    df: pl.DataFrame,
    candidate_cols: list[str],
) -> list[str]:
    """Return compact, interpretable static features for LightGBM.

    Args:
        config: Runtime configuration containing the static feature whitelist.
        df: DataFrame slice used for model training or inference.
        candidate_cols: Fallback feature columns discovered from the dataset.

    Returns:
        Ordered feature names present in ``df``. Uses the centralized whitelist
        first and falls back to discovered candidates for tests or partial data.
    """
    available = [c for c in config.features.static_feature_cols if c in df.columns]
    if available:
        return available
    # Fallback keeps tests and partial feature sets usable.
    return [c for c in candidate_cols if c in df.columns]


# --- Label counting ---


def _counts_dict(values: np.ndarray) -> dict[str, int]:
    """Return compact class/count diagnostics with string keys for JSON."""
    if values.size == 0:
        return {}
    labels, counts = np.unique(values.astype(np.int32), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _pct_dict(counts: dict[str, int]) -> dict[str, float]:
    """Convert count dict to rounded percentages."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: round(count / total * 100.0, 2) for label, count in counts.items()}


# --- Date range extraction ---


def _window_dates(df: pl.DataFrame) -> dict[str, str]:
    """Return start/end timestamps for a window slice."""
    if df.is_empty() or "timestamp" not in df.columns:
        return {"start": "", "end": ""}
    return {"start": str(df["timestamp"][0]), "end": str(df["timestamp"][-1])}


# --- OOF validation ---


def _validate_predictions(df: pl.DataFrame, path: Path) -> None:
    """Validate final OOF predictions before writing the parquet artifact."""
    required = {"timestamp", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing columns {sorted(missing)}: file={path}")
    if df.is_empty():
        raise ValueError(f"Predictions are empty: file={path}")

    ts_col = df["timestamp"]
    if ts_col.null_count() > 0:
        raise ValueError(
            f"Predictions timestamp has nulls:"
            f" actual={ts_col.null_count()}, file={path}"
        )
    if ts_col.n_unique() < len(ts_col):
        dup_count = len(ts_col) - ts_col.n_unique()
        raise ValueError(
            f"OOF predictions contain {dup_count} duplicate timestamps — "
            "walk-forward test windows should be non-overlapping. "
            f"Check step_bars vs test_window_bars. file={path}"
        )
    if ts_col.to_list() != sorted(ts_col.to_list()):
        raise ValueError(f"OOF predictions must be sorted by timestamp: file={path}")

    pred_col = df["pred_label"]
    if pred_col.null_count() > 0:
        raise ValueError(
            f"pred_label has nulls: actual={pred_col.null_count()}, file={path}"
        )
    invalid = sorted(set(pred_col.unique().to_list()) - {-1, 0, 1})
    if invalid:
        raise ValueError(
            f"Invalid pred_label values: expected={{-1,0,1}},"
            f" actual={invalid}, file={path}"
        )

    null_cols = {
        col: df[col].null_count() for col in df.columns if df[col].null_count()
    }
    if null_cols:
        raise ValueError(f"Predictions contain nulls: actual={null_cols}, file={path}")


# --- Manifest I/O ---


def _write_prediction_manifest(
    df: pl.DataFrame,
    path: Path,
    *,
    windows_count: int,
) -> None:
    """Write compact diagnostics beside final_predictions.parquet."""
    mean_confidence = (
        float(df["max_confidence"].mean()) if "max_confidence" in df.columns else None
    )
    manifest = {
        "row_count": len(df),
        "start": str(df["timestamp"][0]),
        "end": str(df["timestamp"][-1]),
        "label_distribution": _counts_dict(df["true_label"].to_numpy())
        if "true_label" in df.columns
        else {},
        "prediction_distribution": _counts_dict(df["pred_label"].to_numpy()),
        "mean_confidence": mean_confidence,
        "windows_count": windows_count,
    }
    manifest_path = path.with_name("prediction_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Prediction manifest saved: %s", manifest_path)


# --- Per-window diagnostics ---


def _window_diagnostics(
    window_idx: int,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Build per-window label diagnostics for logs and JSON artifacts.

    Args:
        window_idx: Zero-based window index.
        train_df: Training split Polars DataFrame.
        test_df: Test split Polars DataFrame.
        y_train: Training label array.
        y_test: Test label array.

    Returns:
        Dictionary with window index, row counts, date ranges, label
        counts (raw and percentage) for both train and test splits.
    """
    train_counts = _counts_dict(y_train)
    test_counts = _counts_dict(y_test)
    diag: dict[str, Any] = {
        "window": window_idx,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_dates": _window_dates(train_df),
        "test_dates": _window_dates(test_df),
        "train_label_counts": train_counts,
        "train_label_pct": _pct_dict(train_counts),
        "test_label_counts": test_counts,
        "test_label_pct": _pct_dict(test_counts),
    }
    logger.info(
        "Window %d labels | train=%s test=%s",
        window_idx,
        diag["train_label_pct"],
        diag["test_label_pct"],
    )
    return diag


# --- Per-class metrics ---


def _compute_per_class_metrics(
    preds: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute per-class precision, recall, F1, and support from predictions.

    Uses ``sklearn.metrics.precision_recall_fscore_support`` with
    ``zero_division=0`` so missing classes return 0.0 rather than raising.

    Args:
        preds: Predicted class labels as a NumPy array.
        y_test: Ground-truth class labels.

    Returns:
        Mapping ``{class_label_str: {"precision", "recall", "f1", "support"}}``
        with string keys (``"-1"``, ``"0"``, ``"1"``) for JSON serialization.
    """
    from sklearn.metrics import precision_recall_fscore_support

    classes = np.array([-1, 0, 1], dtype=np.int32)
    p, r, f1, s = precision_recall_fscore_support(
        y_test, preds, labels=classes, zero_division=0
    )
    return {
        str(int(cls)): {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        }
        for i, cls in enumerate(classes)
    }


# --- Diagnostic enrichment ---


def _add_prediction_diagnostics(
    diag: dict[str, Any],
    preds: np.ndarray,
    y_test: np.ndarray,
    proba: np.ndarray,
) -> None:
    """Attach prediction distribution, confidence, and per-class metrics.

    Mutates ``diag`` by adding prediction counts, accuracy, mean
    confidence, high-confidence fraction, long/short ratio, and per-class
    precision / recall / F1.

    Args:
        diag: Per-window diagnostics dict (mutated in-place).
        preds: Predicted class labels as a NumPy array.
        y_test: Ground-truth class labels.
        proba: Probability matrix (N x 3).
    """
    pred_counts = _counts_dict(preds)
    confidence = np.max(proba, axis=1) if len(proba) else np.array([], dtype=float)

    # Compute LONG/SHORT prediction ratio
    long_count = pred_counts.get("1", 0)
    short_count = pred_counts.get("-1", 0)
    ls_ratio = long_count / short_count if short_count > 0 else float("inf")

    per_class = _compute_per_class_metrics(preds, y_test) if len(y_test) else {}
    diag.update(
        {
            "prediction_counts": pred_counts,
            "prediction_pct": _pct_dict(pred_counts),
            "accuracy": float((preds == y_test).mean()) if len(y_test) else None,
            "mean_confidence": float(confidence.mean()) if len(confidence) else None,
            "high_conf_70_pct": float(
                (confidence >= _HIGH_CONFIDENCE_THRESHOLD).mean() * 100.0
            )
            if len(confidence)
            else None,
            "ls_ratio": round(ls_ratio, 4) if short_count > 0 else None,
            "per_class": per_class,
        }
    )
    logger.info(
        "Window %d preds | pred=%s acc=%.4f mean_conf=%.3f L/S=%.3f",
        diag["window"],
        diag["prediction_pct"],
        diag["accuracy"] or 0.0,
        diag["mean_confidence"] or 0.0,
        ls_ratio if short_count > 0 else float("nan"),
    )
    if per_class:
        logger.info(
            "Window %d per-class | SHORT: P=%.3f R=%.3f F1=%.3f | "
            "HOLD: P=%.3f R=%.3f F1=%.3f | "
            "LONG: P=%.3f R=%.3f F1=%.3f",
            diag["window"],
            per_class["-1"]["precision"],
            per_class["-1"]["recall"],
            per_class["-1"]["f1"],
            per_class["0"]["precision"],
            per_class["0"]["recall"],
            per_class["0"]["f1"],
            per_class["1"]["precision"],
            per_class["1"]["recall"],
            per_class["1"]["f1"],
        )
    if short_count > 0 and long_count / short_count < _SHORT_BIAS_RATIO_THRESHOLD:
        logger.warning(
            "Window %d: SHORT bias — LONG/SHORT ratio = %.2f",
            diag["window"],
            long_count / short_count,
        )
    elif long_count > 0 and short_count / long_count < _SHORT_BIAS_RATIO_THRESHOLD:
        logger.warning(
            "Window %d: LONG bias — SHORT/LONG ratio = %.2f",
            diag["window"],
            short_count / long_count,
        )
    else:
        logger.info(
            "Window %d: L/S balanced — ratio %.2f",
            diag["window"],
            ls_ratio if short_count > 0 else float("inf"),
        )


# --- GRU signal quality ---


def _log_gru_signal_quality(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    config: Config,
) -> None:
    """Log GRU hidden-state signal-to-noise diagnostic using ANOVA F-statistic.

    For each walk-forward window's GRU hidden states, compute the ANOVA
    F-statistic between each hidden dimension and the label via
    ``sklearn.feature_selection.f_classif``.  Logs the top-5 and bottom-5
    dimensions with their F-scores.  If all dimensions have near-zero
    predictive power, logs a warning that the GRU is contributing noise.

    Args:
        hidden_states: (n_samples, n_features) GRU hidden-state matrix.
        labels: (n_samples,) multiclass integer labels.
        config: Runtime configuration (unused; kept for interface consistency).
    """
    try:
        from sklearn.feature_selection import f_classif  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("sklearn not available — skipping GRU signal quality check")
        return

    if hidden_states is None or hidden_states.size == 0:
        logger.warning("GRU signal quality: empty hidden states, skipping")
        return

    if labels is None or labels.size == 0:
        logger.warning("GRU signal quality: empty labels, skipping")
        return

    if len(hidden_states) != len(labels):
        logger.warning(
            "GRU signal quality: shape mismatch hidden=%s vs labels=%s, skipping",
            hidden_states.shape,
            labels.shape,
        )
        return

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(
            "GRU signal quality: only %d class(es) present, "
            "cannot compute F-statistic (need ≥2)",
            len(unique_labels),
        )
        return

    min_samples_per_class = _ANOVA_MIN_SAMPLES_PER_CLASS
    for cls in unique_labels:
        if np.sum(labels == cls) < min_samples_per_class:
            logger.warning(
                "GRU signal quality: class %s has < %d samples, skipping",
                cls,
                min_samples_per_class,
            )
            return

    try:
        f_scores, _p_values = f_classif(hidden_states, labels)
    except (ValueError, TypeError) as exc:
        logger.warning("GRU signal quality: f_classif failed — %s", exc)
        return

    n_features = len(f_scores)
    sorted_indices = np.argsort(f_scores)[::-1]  # descending

    top_n = min(_SIGNAL_QUALITY_TOP_N, n_features)
    bottom_n = min(_SIGNAL_QUALITY_TOP_N, n_features)

    top_indices = sorted_indices[:top_n]
    bottom_indices = sorted_indices[-bottom_n:][::-1]  # ascending for bottom display

    mean_f = float(np.mean(f_scores))

    logger.info(
        "GRU hidden signal quality: mean F=%.4f | top-5: %s | bottom-5: %s",
        mean_f,
        ", ".join(f"dim{i}={f_scores[i]:.3f}" for i in top_indices),
        ", ".join(f"dim{i}={f_scores[i]:.3f}" for i in bottom_indices),
    )

    if mean_f < _GRU_SIGNAL_F_SCORE_THRESHOLD:
        logger.warning(
            "GRU hidden states show no detectable signal — GRU contributes noise "
            "(mean F=%.4f across %d dimensions)",
            mean_f,
            n_features,
        )


# --- Probability / label helpers ---


def _label_suffix(class_label: int) -> str:
    """Return the canonical probability-column suffix for a class label.

    Args:
        class_label: An integer from ``{-1, 0, 1}``.

    Returns:
        String suffix such as ``"minus1"`` or ``"0"``.
    """
    return f"minus{abs(class_label)}" if class_label < 0 else str(class_label)


def _one_hot_proba_columns(
    preds: np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """Build one-hot probability columns from predicted class labels.

    Used for regression mode where the model outputs a scalar rather than
    class probabilities.  Each class column is 1.0 where the prediction
    matches, 0.0 otherwise.

    Args:
        preds: Array of predicted class labels (``-1``, ``0``, or ``1``).
        prefix: Column name prefix (default ``"pred_proba_class_"``).

    Returns:
        Dictionary mapping canonical column names to 1-D one-hot arrays.
    """
    preds = np.asarray(preds, dtype=np.int32)
    return {
        f"{prefix}{_label_suffix(int(cls))}": (preds == cls).astype(np.float64)
        for cls in _CLASS_ORDER
    }


def _align_probability_matrix(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
) -> np.ndarray:
    """Align class probabilities to the canonical ``[-1, 0, 1]`` order.

    Some LightGBM models produce probabilities in a different class order
    (e.g. ``[0, 1]`` for binary).  This function maps them to the fixed
    ``[-1, 0, 1]`` column order expected by downstream stages.

    Args:
        proba: Raw probability matrix ``(N, C)`` from the model.
        class_order: Model's ``classes_`` attribute (list or array).

    Returns:
        Probability matrix aligned to ``_CLASS_ORDER`` (``[-1, 0, 1]``).
    """
    aligned = np.zeros((len(proba), len(_CLASS_ORDER)), dtype=np.float64)
    index_by_class = {int(cls): idx for idx, cls in enumerate(class_order)}
    for target_idx, cls in enumerate(_CLASS_ORDER):
        source_idx = index_by_class.get(int(cls))
        if source_idx is not None:
            aligned[:, target_idx] = proba[:, source_idx]
    return aligned


def _probability_columns(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """Build canonical probability columns for ``{-1, 0, 1}``.

    Args:
        proba: Raw probability matrix from the model.
        class_order: Model's ``classes_`` attribute.
        prefix: Column name prefix (default ``"pred_proba_class_"``).

    Returns:
        Dictionary mapping canonical column names to 1-D probability
        arrays.
    """
    aligned = _align_probability_matrix(proba, class_order)
    return {
        f"{prefix}{_label_suffix(int(cls))}": aligned[:, idx]
        for idx, cls in enumerate(_CLASS_ORDER)
    }


_PROBA_COLS = ("pred_proba_class_minus1", "pred_proba_class_0", "pred_proba_class_1")
"""Canonical probability column names in ``[-1, 0, 1]`` order."""


def _add_confidence_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``max_confidence`` and ``confidence_bin`` to an OOF DataFrame.

    Requires the three canonical probability columns to exist.  If they
    are absent (e.g. legacy outputs), returns the DataFrame unchanged.

    Args:
        df: OOF predictions DataFrame with ``pred_proba_class_*`` columns.

    Returns:
        DataFrame augmented with ``max_confidence`` (float64) and
        ``confidence_bin`` (string: ``"high"`` / ``"medium"`` / ``"low"``).
    """
    if not all(c in df.columns for c in _PROBA_COLS):
        return df
    return df.with_columns(
        pl.max_horizontal([pl.col(c) for c in _PROBA_COLS]).alias("max_confidence"),
    ).with_columns(
        pl.when(pl.col("max_confidence") >= 0.6)
        .then(pl.lit("high"))
        .when(pl.col("max_confidence") >= 0.4)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
        .alias("confidence_bin"),
    )
