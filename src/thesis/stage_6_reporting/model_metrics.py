"""Classification and regression auxiliary metric functions.

Primary metrics measure directional accuracy for the 3-class
(Short / Hold / Long) classification model.  Secondary *regression auxiliary*
metrics (MAE, RMSE, R²) measure error in predicted **return magnitude** — they
are computed on continuous arrays, not classification labels.

For multiclass models that only emit class probabilities, a proxy return is
derived via ``compute_proxy_return`` (class-weighted scoring).  For true
regression models, use the raw predicted return directly.

All functions are stateless, side-effect free, and accept only numpy arrays.
This module is the canonical source for metric computation; reporting and
dashboard code should import from here rather than re-implementing inline.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Individual metric helpers


def accuracy(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Overall accuracy: fraction of correct predictions."""
    return float((y_true == y_pred).mean())


def balanced_accuracy(
    y_true: npt.NDArray, y_pred: npt.NDArray, classes: list[int] | None = None
) -> float:
    """Average recall across classes."""
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    recalls: list[float] = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def directional_accuracy(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Accuracy on bars where *both* true and predicted labels are non-zero.

    Hold-vs-direction mismatches are excluded rather than counted as wrong.
    """
    mask = (y_true != 0) & (y_pred != 0)
    if mask.sum() == 0:
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def mda_no_hold(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """MDA excluding Hold — only evaluate rows where true label is Short or Long."""
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def mda_including_hold(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """MDA including Hold — exact match across all three classes."""
    return accuracy(y_true, y_pred)


def mda_binary(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """MDA for Long vs Short only.

    Hold predictions on directional bars count as wrong.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    correct = (y_true[mask] == y_pred[mask]) & (y_pred[mask] != 0)
    return float(correct.mean())


def _precision_recall_f1_for_class(
    y_true: npt.NDArray, y_pred: npt.NDArray, cls: int
) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for a single class."""
    true_mask = y_true == cls
    pred_mask = y_pred == cls
    rec = float((y_pred[true_mask] == cls).mean()) if true_mask.sum() > 0 else 0.0
    prec = float((y_true[pred_mask] == cls).mean()) if pred_mask.sum() > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def macro_f1(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: list[int] | None = None,
) -> float:
    """Macro-averaged F1 score."""
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = [_precision_recall_f1_for_class(y_true, y_pred, c)[2] for c in classes]
    return float(np.mean(f1s))


def weighted_f1(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: list[int] | None = None,
) -> float:
    """Support-weighted F1 score."""
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    total_f1 = 0.0
    total_support = 0
    for c in classes:
        support = int((y_true == c).sum())
        _, _, f1 = _precision_recall_f1_for_class(y_true, y_pred, c)
        total_f1 += f1 * support
        total_support += support
    return total_f1 / total_support if total_support > 0 else 0.0


def precision_recall_f1_per_class(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1 keyed by human-readable name."""
    if classes is None:
        classes = [-1, 0, 1]
    if class_names is None:
        class_names = {-1: "Short", 0: "Hold", 1: "Long"}
    result: dict[str, dict[str, float]] = {}
    for c in classes:
        prec, rec, f1 = _precision_recall_f1_for_class(y_true, y_pred, c)
        name = class_names.get(c, str(c))
        result[name] = {"precision": prec, "recall": rec, "f1": f1}
    return result


def confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, dict[str, int]]:
    """3×3 confusion matrix as nested dict  {true_name: {pred_name: count}}."""
    if classes is None:
        classes = [-1, 0, 1]
    if class_names is None:
        class_names = {-1: "Short", 0: "Hold", 1: "Long"}
    cm: dict[str, dict[str, int]] = {}
    for tc in classes:
        row: dict[str, int] = {}
        for pc in classes:
            row[class_names.get(pc, str(pc))] = int(
                ((y_true == tc) & (y_pred == pc)).sum()
            )
        cm[class_names.get(tc, str(tc))] = row
    return cm


def direction_confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
) -> dict[str, dict[str, int]]:
    """2×2 confusion matrix for Short vs Long only (Hold rows excluded)."""
    mask = y_true != 0
    yt = y_true[mask]
    yp = y_pred[mask]
    names = {-1: "Short", 1: "Long"}
    cm: dict[str, dict[str, int]] = {}
    for tc in [-1, 1]:
        row: dict[str, int] = {}
        for pc in [-1, 1]:
            row[names[pc]] = int(((yt == tc) & (yp == pc)).sum())
        cm[names[tc]] = row
    return cm


def majority_baseline_accuracy(
    y_true: npt.NDArray,
    classes: list[int] | None = None,
) -> float:
    """Accuracy if we always predict the most common class."""
    if classes is None:
        classes = [-1, 0, 1]
    n = len(y_true)
    if n == 0:
        return 0.0
    return float(max((y_true == c).sum() for c in classes) / n)


def high_confidence_accuracy(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    y_proba: npt.NDArray,
    threshold: float = 0.6,
) -> dict[str, float | int]:
    """Accuracy when max predicted probability exceeds *threshold*.

    Returns dict with keys: accuracy, count, pct_of_total.
    """
    max_proba = y_proba.max(axis=1)
    mask = max_proba >= threshold
    count = int(mask.sum())
    total = len(y_true)
    if count == 0:
        return {"accuracy": 0.0, "count": 0, "pct_of_total": 0.0}
    acc = float((y_true[mask] == y_pred[mask]).mean())
    return {"accuracy": acc, "count": count, "pct_of_total": count / total * 100}


# Regression auxiliary metrics (continuous arrays)


def mae(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Mean Absolute Error on continuous return arrays."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """Root Mean Squared Error on continuous return arrays."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
    """R² (coefficient of determination) on continuous return arrays."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_proxy_return(
    y_proba: npt.NDArray, classes: list[int] | None = None
) -> npt.NDArray:
    """Convert class probabilities to a pseudo-continuous predicted return.

    Uses class-weighted scoring:  proxy = Σ class_label × P(class).
    For the default classes [-1, 0, 1] this simplifies to
    ``P(Long) - P(Short)``, producing values in [-1, 1].

    This is a **proxy** — it does not represent an actual predicted return
    magnitude.  It is useful as a secondary diagnostic for regression-style
    evaluation of classification models.

    Args:
        y_proba: (N, C) probability matrix.
        classes: label values corresponding to columns (default [-1, 0, 1]).

    Returns:
        1-D array of shape (N,) with pseudo-return scores.
    """
    if classes is None:
        classes = _DEFAULT_CLASSES
    labels = np.array(classes, dtype=np.float64)
    return y_proba @ labels  # (N, C) @ (C,) -> (N,)


def compute_regression_auxiliary(
    y_true_returns: npt.NDArray, y_pred_returns: npt.NDArray
) -> dict[str, float]:
    """Return dict with MAE, RMSE, R² for continuous return arrays.

    Args:
        y_true_returns: 1-D array of actual (ground-truth) returns.
        y_pred_returns: 1-D array of predicted returns — either raw model
            output (regression) or proxy from ``compute_proxy_return``.

    Returns:
        ``{"mae": float, "rmse": float, "r_squared": float}``
    """
    return {
        "mae": mae(y_true_returns, y_pred_returns),
        "rmse": rmse(y_true_returns, y_pred_returns),
        "r_squared": r_squared(y_true_returns, y_pred_returns),
    }


# Main entry point

# Default class ordering and names for the 3-class directional model.
_DEFAULT_CLASSES: list[int] = [-1, 0, 1]
_DEFAULT_CLASS_NAMES: dict[int, str] = {-1: "Short", 0: "Hold", 1: "Long"}


def compute_all_classification_metrics(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    y_proba: npt.NDArray | None = None,
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
    y_true_returns: npt.NDArray | None = None,
    y_pred_returns: npt.NDArray | None = None,
) -> dict:
    """Compute the full suite of classification metrics.

    Optionally includes regression auxiliary metrics when continuous return
    arrays are supplied.

    Args:
        y_true: 1-D array of true labels.
        y_pred: 1-D array of predicted labels.
        y_proba: (N, C) probability matrix — optional, needed only for
            high-confidence accuracy and proxy return computation.
        classes: ordered list of label values (default ``[-1, 0, 1]``).
        class_names: mapping from label value to display name.
        y_true_returns: 1-D array of actual returns.  If provided together
            with *y_pred_returns* (or *y_proba*), regression auxiliary metrics
            are appended under key ``"regression_auxiliary"``.
        y_pred_returns: 1-D array of predicted returns for regression models.
            For classification models, omit this and pass *y_proba* instead —
            a proxy return will be computed automatically.

    Returns:
        Dict with every metric listed in the module docstring.
    """
    if classes is None:
        classes = _DEFAULT_CLASSES
    if class_names is None:
        class_names = _DEFAULT_CLASS_NAMES

    result: dict = {
        "total": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, classes),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "mda_no_hold": mda_no_hold(y_true, y_pred),
        "mda_including_hold": mda_including_hold(y_true, y_pred),
        "mda_binary": mda_binary(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, classes),
        "weighted_f1": weighted_f1(y_true, y_pred, classes),
        "precision_recall_f1_per_class": precision_recall_f1_per_class(
            y_true, y_pred, classes, class_names
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, classes, class_names),
        "direction_confusion_matrix": direction_confusion_matrix(y_true, y_pred),
        "majority_baseline_accuracy": majority_baseline_accuracy(y_true, classes),
    }

    if y_proba is not None:
        result["high_confidence_accuracy"] = high_confidence_accuracy(
            y_true, y_pred, y_proba
        )

    # Regression auxiliary: compute when ground-truth returns are available.
    if y_true_returns is not None:
        if y_pred_returns is not None:
            pred_returns = y_pred_returns
        elif y_proba is not None:
            pred_returns = compute_proxy_return(y_proba, classes)
        else:
            pred_returns = None

        if pred_returns is not None:
            result["regression_auxiliary"] = compute_regression_auxiliary(
                y_true_returns, pred_returns
            )

    return result
