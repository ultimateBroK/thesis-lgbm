"""Unit tests for _model_metrics — pure classification metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.stage_6_reporting.model_metrics import (
    accuracy,
    balanced_accuracy,
    compute_all_classification_metrics,
    confusion_matrix,
    direction_confusion_matrix,
    directional_accuracy,
    high_confidence_accuracy,
    macro_f1,
    majority_baseline_accuracy,
    mda_binary,
    mda_including_hold,
    mda_no_hold,
    precision_recall_f1_per_class,
    weighted_f1,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Labels: -1 = Short, 0 = Hold, 1 = Long

_PERFECT_TRUE = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
_PERFECT_PRED = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

_ALL_WRONG_TRUE = np.array([-1, 0, 1, -1, 0, 1])
_ALL_WRONG_PRED = np.array([1, -1, -1, 1, 1, -1])

# 90 % Hold, 5 % Short, 5 % Long
_IMBALANCE_TRUE = np.array([0] * 90 + [-1] * 5 + [1] * 5)
_IMBALANCE_PRED_HOLD = np.array([0] * 100)  # predict Hold for everything


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAccuracy:
    def test_perfect(self) -> None:
        assert accuracy(_PERFECT_TRUE, _PERFECT_PRED) == 1.0

    def test_all_wrong(self) -> None:
        assert accuracy(_ALL_WRONG_TRUE, _ALL_WRONG_PRED) == 0.0

    def test_partial(self) -> None:
        y_true = np.array([0, 1, -1, 0])
        y_pred = np.array([0, 0, -1, 1])
        assert accuracy(y_true, y_pred) == 0.5

    def test_single_element_correct(self) -> None:
        assert accuracy(np.array([1]), np.array([1])) == 1.0

    def test_single_element_wrong(self) -> None:
        assert accuracy(np.array([1]), np.array([0])) == 0.0


# ---------------------------------------------------------------------------
# balanced_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBalancedAccuracy:
    def test_perfect(self) -> None:
        assert balanced_accuracy(_PERFECT_TRUE, _PERFECT_PRED) == 1.0

    def test_all_wrong_zero(self) -> None:
        # Every class has recall 0
        assert balanced_accuracy(_ALL_WRONG_TRUE, _ALL_WRONG_PRED) == 0.0

    def test_imbalance_predict_all_hold(self) -> None:
        # Hold recall=1.0, Short recall=0.0, Long recall=0.0 → mean=1/3
        ba = balanced_accuracy(_IMBALANCE_TRUE, _IMBALANCE_PRED_HOLD)
        assert abs(ba - 1 / 3) < 1e-9


# ---------------------------------------------------------------------------
# directional_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDirectionalAccuracy:
    def test_perfect_directional(self) -> None:
        y_true = np.array([-1, 1, -1, 1])
        y_pred = np.array([-1, 1, -1, 1])
        assert directional_accuracy(y_true, y_pred) == 1.0

    def test_hold_predictions_excluded(self) -> None:
        y_true = np.array([-1, 1, -1])
        y_pred = np.array([0, 0, 0])
        # All predicted Hold → mask excludes everything
        assert directional_accuracy(y_true, y_pred) == 0.0

    def test_mixed(self) -> None:
        y_true = np.array([-1, 1, -1, 1])
        y_pred = np.array([-1, 1, 0, 0])
        # Only first two qualify (both non-zero on both sides), both correct
        assert directional_accuracy(y_true, y_pred) == 1.0


# ---------------------------------------------------------------------------
# MDA variants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMdaNoHold:
    def test_perfect_directional(self) -> None:
        y_true = np.array([-1, 1, -1, 1])
        y_pred = np.array([-1, 1, -1, 1])
        assert mda_no_hold(y_true, y_pred) == 1.0

    def test_hold_true_excluded(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert mda_no_hold(y_true, y_pred) == 0.0

    def test_partial(self) -> None:
        y_true = np.array([-1, 1, -1, 1])
        y_pred = np.array([-1, -1, -1, 1])
        # 3 correct out of 4 = 0.75
        assert mda_no_hold(y_true, y_pred) == 0.75


@pytest.mark.unit
class TestMdaIncludingHold:
    def test_same_as_accuracy(self) -> None:
        y_true = np.array([-1, 0, 1, 0])
        y_pred = np.array([-1, 0, 1, 1])
        assert mda_including_hold(y_true, y_pred) == accuracy(y_true, y_pred)


@pytest.mark.unit
class TestMdaBinary:
    def test_correct_long_short(self) -> None:
        y_true = np.array([-1, 1, -1, 1])
        y_pred = np.array([-1, 1, -1, 1])
        assert mda_binary(y_true, y_pred) == 1.0

    def test_hold_prediction_counts_wrong(self) -> None:
        y_true = np.array([-1, 1])
        y_pred = np.array([0, 0])
        # Hold predictions on directional bars count as wrong
        assert mda_binary(y_true, y_pred) == 0.0

    def test_no_directional_bars(self) -> None:
        y_true = np.array([0, 0])
        y_pred = np.array([0, 0])
        assert mda_binary(y_true, y_pred) == 0.0


# ---------------------------------------------------------------------------
# F1 scores
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMacroF1:
    def test_perfect(self) -> None:
        assert macro_f1(_PERFECT_TRUE, _PERFECT_PRED) == 1.0

    def test_all_wrong_zero(self) -> None:
        assert macro_f1(_ALL_WRONG_TRUE, _ALL_WRONG_PRED) == 0.0


@pytest.mark.unit
class TestWeightedF1:
    def test_perfect(self) -> None:
        assert weighted_f1(_PERFECT_TRUE, _PERFECT_PRED) == 1.0


@pytest.mark.unit
class TestPrecisionRecallF1PerClass:
    def test_perfect(self) -> None:
        result = precision_recall_f1_per_class(_PERFECT_TRUE, _PERFECT_PRED)
        for cls_name in ("Short", "Hold", "Long"):
            assert result[cls_name]["precision"] == 1.0
            assert result[cls_name]["recall"] == 1.0
            assert result[cls_name]["f1"] == 1.0

    def test_all_same_class(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        result = precision_recall_f1_per_class(y_true, y_pred)
        assert result["Hold"]["f1"] == 1.0


# ---------------------------------------------------------------------------
# confusion_matrix & direction_confusion_matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfusionMatrix:
    def test_perfect_diagonal(self) -> None:
        cm = confusion_matrix(_PERFECT_TRUE, _PERFECT_PRED)
        for true_name in ("Short", "Hold", "Long"):
            for pred_name in ("Short", "Hold", "Long"):
                expected = 3 if true_name == pred_name else 0
                assert cm[true_name][pred_name] == expected

    def test_all_wrong(self) -> None:
        cm = confusion_matrix(np.array([-1, 0, 1]), np.array([1, 1, -1]))
        assert cm["Short"]["Long"] == 1
        assert cm["Hold"]["Long"] == 1
        assert cm["Long"]["Short"] == 1


@pytest.mark.unit
class TestDirectionConfusionMatrix:
    def test_only_short_long(self) -> None:
        y_true = np.array([-1, 1, -1, 1])
        y_pred = np.array([-1, 1, 1, -1])
        cm = direction_confusion_matrix(y_true, y_pred)
        # Only Short and Long keys
        assert set(cm.keys()) == {"Short", "Long"}
        for row in cm.values():
            assert set(row.keys()) == {"Short", "Long"}
        assert cm["Short"]["Short"] == 1
        assert cm["Short"]["Long"] == 1
        assert cm["Long"]["Long"] == 1
        assert cm["Long"]["Short"] == 1

    def test_hold_rows_excluded(self) -> None:
        y_true = np.array([0, 0, -1])
        y_pred = np.array([0, 0, -1])
        cm = direction_confusion_matrix(y_true, y_pred)
        assert cm["Short"]["Short"] == 1
        assert cm["Long"]["Short"] == 0


# ---------------------------------------------------------------------------
# majority_baseline_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMajorityBaseline:
    def test_imbalanced(self) -> None:
        ba = majority_baseline_accuracy(_IMBALANCE_TRUE)
        assert abs(ba - 0.90) < 1e-9

    def test_empty(self) -> None:
        assert majority_baseline_accuracy(np.array([], dtype=int)) == 0.0

    def test_uniform(self) -> None:
        y = np.array([-1, 0, 1, -1, 0, 1])
        assert abs(majority_baseline_accuracy(y) - 1 / 3) < 1e-9


# ---------------------------------------------------------------------------
# high_confidence_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHighConfidenceAccuracy:
    def test_threshold_filters(self) -> None:
        y_true = np.array([0, 1, -1])
        y_pred = np.array([0, 1, -1])
        y_proba = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        result = high_confidence_accuracy(y_true, y_pred, y_proba, threshold=0.7)
        # Only last 2 pass threshold
        assert result["count"] == 2
        assert result["accuracy"] == 1.0
        assert abs(result["pct_of_total"] - 2 / 3 * 100) < 1e-6

    def test_none_above_threshold(self) -> None:
        y_true = np.array([0])
        y_pred = np.array([0])
        y_proba = np.array([[0.4, 0.35, 0.25]])
        result = high_confidence_accuracy(y_true, y_pred, y_proba, threshold=0.9)
        assert result["accuracy"] == 0.0
        assert result["count"] == 0

    def test_all_above_threshold(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_proba = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
        result = high_confidence_accuracy(y_true, y_pred, y_proba, threshold=0.5)
        assert result["count"] == 2
        assert result["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# compute_all_classification_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeAll:
    def test_returns_all_keys(self) -> None:
        result = compute_all_classification_metrics(_PERFECT_TRUE, _PERFECT_PRED)
        expected_keys = {
            "total",
            "accuracy",
            "balanced_accuracy",
            "directional_accuracy",
            "mda_no_hold",
            "mda_including_hold",
            "mda_binary",
            "macro_f1",
            "weighted_f1",
            "precision_recall_f1_per_class",
            "confusion_matrix",
            "direction_confusion_matrix",
            "majority_baseline_accuracy",
        }
        assert expected_keys <= set(result.keys())

    def test_with_proba_adds_high_conf(self) -> None:
        y_proba = np.full((len(_PERFECT_TRUE), 3), 1 / 3)
        result = compute_all_classification_metrics(
            _PERFECT_TRUE, _PERFECT_PRED, y_proba
        )
        assert "high_confidence_accuracy" in result

    def test_perfect_metrics(self) -> None:
        result = compute_all_classification_metrics(_PERFECT_TRUE, _PERFECT_PRED)
        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0

    def test_with_regression_auxiliary(self) -> None:
        """When y_true_returns and y_proba are provided, regression aux is computed."""
        from thesis.stage_6_reporting.model_metrics import compute_proxy_return

        y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.1, 0.8, 0.1]] * 3)
        y_true_returns = np.random.randn(9)
        result = compute_all_classification_metrics(
            _PERFECT_TRUE, _PERFECT_PRED, y_proba, y_true_returns=y_true_returns
        )
        assert "regression_auxiliary" in result
        assert "mae" in result["regression_auxiliary"]
        assert "rmse" in result["regression_auxiliary"]
        assert "r_squared" in result["regression_auxiliary"]

    def test_with_pred_returns(self) -> None:
        """When y_true_returns and y_pred_returns are provided."""
        y_true_returns = np.array(
            [0.1, -0.2, 0.05, -0.1, 0.15, -0.05, 0.08, -0.03, 0.12]
        )
        y_pred_returns = np.array(
            [0.08, -0.15, 0.06, -0.08, 0.12, -0.04, 0.07, -0.02, 0.10]
        )
        result = compute_all_classification_metrics(
            _PERFECT_TRUE,
            _PERFECT_PRED,
            y_true_returns=y_true_returns,
            y_pred_returns=y_pred_returns,
        )
        assert "regression_auxiliary" in result
        assert result["regression_auxiliary"]["mae"] > 0

    def test_no_returns_no_regression(self) -> None:
        """When no returns are provided, regression_auxiliary is absent."""
        result = compute_all_classification_metrics(_PERFECT_TRUE, _PERFECT_PRED)
        assert "regression_auxiliary" not in result


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

from thesis.stage_6_reporting.model_metrics import (
    compute_proxy_return,
    compute_regression_auxiliary,
    mae,
    rmse,
    r_squared,
)


@pytest.mark.unit
class TestRegressionMetrics:
    def test_mae(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        assert abs(mae(y_true, y_pred) - 0.133) < 0.01

    def test_rmse(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert rmse(y_true, y_pred) == 0.0

    def test_r_squared_perfect(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert r_squared(y_true, y_pred) == 1.0

    def test_r_squared_zero_variance(self) -> None:
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        assert r_squared(y_true, y_pred) == 0.0

    def test_compute_proxy_return(self) -> None:
        proba = np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]])
        result = compute_proxy_return(proba)
        # P(Long) - P(Short) for each row
        assert result[0] == pytest.approx(0.5)  # 0.6 - 0.1
        assert result[1] == pytest.approx(-0.6)  # 0.1 - 0.7

    def test_compute_regression_auxiliary(self) -> None:
        y_true = np.array([0.1, -0.2, 0.05])
        y_pred = np.array([0.08, -0.15, 0.06])
        result = compute_regression_auxiliary(y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result
        assert "r_squared" in result
