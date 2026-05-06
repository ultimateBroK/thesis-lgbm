"""Baseline prediction strategies for walk-forward comparison.

All baselines operate on the same walk-forward windows as the hybrid model
to prevent information leakage. Pure numpy, no side effects.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Individual baseline strategies


def naive_direction(y_returns: npt.NDArray) -> npt.NDArray:
    """Predict the direction of the previous bar's return (persistence).

    Maps: return > 0 → 1, return < 0 → -1, else 0.
    First element has no predecessor → prediction is 0.
    """
    preds = np.zeros(len(y_returns), dtype=np.int8)
    preds[1:] = np.sign(y_returns[:-1]).astype(np.int8)
    return preds


def always_predict_class(y_true: npt.NDArray, class_label: int) -> npt.NDArray:
    """Return predictions that always equal *class_label*."""
    return np.full(len(y_true), class_label, dtype=np.int8)


def majority_class_baseline(
    y_true: npt.NDArray,
) -> tuple[npt.NDArray, int]:
    """Find the most common class and return uniform predictions + class chosen."""
    values, counts = np.unique(y_true, return_counts=True)
    majority = int(values[np.argmax(counts)])
    return np.full(len(y_true), majority, dtype=np.int8), majority


def random_baseline(
    n_samples: int,
    classes: list[int] | None = None,
    seed: int = 42,
) -> npt.NDArray:
    """Random predictions drawn uniformly from *classes* with a fixed seed."""
    if classes is None:
        classes = [-1, 0, 1]
    rng = np.random.default_rng(seed)
    return rng.choice(classes, size=n_samples).astype(np.int8)


# Metric computation (reuses stage_6_reporting where possible)


def compute_baseline_metrics(y_true: npt.NDArray, y_pred: npt.NDArray) -> dict:
    """Compute accuracy, macro_f1, directional_accuracy for a baseline."""
    from thesis.stage_6_reporting.model_metrics import (
        accuracy,
        directional_accuracy,
        macro_f1,
    )

    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


# Run all baselines


def run_all_baselines(
    y_true: npt.NDArray,
    y_returns: npt.NDArray,
    seed: int = 42,
) -> dict[str, dict]:
    """Run every baseline and return ``{name: metrics_dict}``."""
    results: dict[str, dict] = {}

    # 1. Naive direction (persistence)
    naive_pred = naive_direction(y_returns)
    results["naive_direction"] = compute_baseline_metrics(y_true, naive_pred)

    # 2. Always long
    always_long = always_predict_class(y_true, 1)
    results["always_long"] = compute_baseline_metrics(y_true, always_long)

    # 3. Always short
    always_short = always_predict_class(y_true, -1)
    results["always_short"] = compute_baseline_metrics(y_true, always_short)

    # 4. Always hold
    always_hold = always_predict_class(y_true, 0)
    results["always_hold"] = compute_baseline_metrics(y_true, always_hold)

    # 5. Majority class
    majority_pred, majority_cls = majority_class_baseline(y_true)
    results["majority_class"] = compute_baseline_metrics(y_true, majority_pred)
    results["majority_class"]["majority_class_label"] = majority_cls

    # 6. Random
    random_pred = random_baseline(len(y_true), seed=seed)
    results["random"] = compute_baseline_metrics(y_true, random_pred)

    return results
