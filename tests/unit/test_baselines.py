"""Unit tests for _baselines — baseline prediction strategies."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.stage_4_training.baselines import (
    always_predict_class,
    compute_baseline_metrics,
    majority_class_baseline,
    naive_direction,
    random_baseline,
    run_all_baselines,
)


# ---------------------------------------------------------------------------
# naive_direction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNaiveDirection:
    def test_known_returns(self) -> None:
        y_returns = np.array([0.5, -0.3, 0.0, 0.1, -0.2])
        result = naive_direction(y_returns)
        # First has no predecessor → 0, then sign of previous bar
        np.testing.assert_array_equal(result, [0, 1, -1, 0, 1])

    def test_all_positive(self) -> None:
        y_returns = np.array([1.0, 2.0, 3.0])
        result = naive_direction(y_returns)
        np.testing.assert_array_equal(result, [0, 1, 1])

    def test_single_element(self) -> None:
        result = naive_direction(np.array([0.5]))
        np.testing.assert_array_equal(result, [0])


# ---------------------------------------------------------------------------
# always_predict_class
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAlwaysPredictClass:
    def test_returns_correct_class(self) -> None:
        y_true = np.array([-1, 0, 1, -1, 0])
        result = always_predict_class(y_true, 1)
        assert len(result) == 5
        assert (result == 1).all()

    def test_different_class(self) -> None:
        y_true = np.zeros(10)
        result = always_predict_class(y_true, -1)
        assert (result == -1).all()


# ---------------------------------------------------------------------------
# majority_class_baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMajorityClassBaseline:
    def test_finds_most_common(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, -1])
        preds, cls = majority_class_baseline(y_true)
        assert cls == 0
        assert (preds == 0).all()

    def test_tie_picks_first_sorted(self) -> None:
        y_true = np.array([-1, 1])
        preds, cls = majority_class_baseline(y_true)
        # np.unique sorts → first in sorted order is -1
        assert cls == -1

    def test_single_class(self) -> None:
        y_true = np.array([1, 1, 1])
        _, cls = majority_class_baseline(y_true)
        assert cls == 1


# ---------------------------------------------------------------------------
# random_baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRandomBaseline:
    def test_deterministic_with_seed(self) -> None:
        r1 = random_baseline(20, seed=7)
        r2 = random_baseline(20, seed=7)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self) -> None:
        r1 = random_baseline(100, seed=1)
        r2 = random_baseline(100, seed=2)
        assert not np.array_equal(r1, r2)

    def test_custom_classes(self) -> None:
        result = random_baseline(50, classes=[0, 1], seed=42)
        assert set(result.tolist()) <= {0, 1}

    def test_correct_length(self) -> None:
        assert len(random_baseline(10)) == 10


# ---------------------------------------------------------------------------
# compute_baseline_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeBaselineMetrics:
    def test_perfect_prediction(self) -> None:
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = np.array([-1, 0, 1, -1, 0, 1])
        metrics = compute_baseline_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["directional_accuracy"] == 1.0

    def test_returns_expected_keys(self) -> None:
        y = np.array([0, 1, -1])
        metrics = compute_baseline_metrics(y, y)
        assert set(metrics.keys()) == {"accuracy", "macro_f1", "directional_accuracy"}


# ---------------------------------------------------------------------------
# run_all_baselines
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunAllBaselines:
    @pytest.fixture()
    def sample_data(self) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.array([1, -1, 0, 1, -1, 0, 1, 0])
        y_returns = np.array([0.5, -0.3, 0.0, 0.2, -0.1, 0.0, 0.4, -0.2])
        return y_true, y_returns

    def test_returns_expected_keys(self, sample_data: tuple) -> None:
        y_true, y_returns = sample_data
        results = run_all_baselines(y_true, y_returns)
        expected = {
            "naive_direction",
            "always_long",
            "always_short",
            "always_hold",
            "majority_class",
            "random",
        }
        assert set(results.keys()) == expected

    def test_each_baseline_has_metrics(self, sample_data: tuple) -> None:
        y_true, y_returns = sample_data
        results = run_all_baselines(y_true, y_returns)
        for name, metrics in results.items():
            assert "accuracy" in metrics, f"{name} missing accuracy"
            assert "macro_f1" in metrics, f"{name} missing macro_f1"
            assert "directional_accuracy" in metrics, (
                f"{name} missing directional_accuracy"
            )

    def test_majority_class_includes_label(self, sample_data: tuple) -> None:
        y_true, y_returns = sample_data
        results = run_all_baselines(y_true, y_returns)
        assert "majority_class_label" in results["majority_class"]

    def test_deterministic(self, sample_data: tuple) -> None:
        y_true, y_returns = sample_data
        r1 = run_all_baselines(y_true, y_returns, seed=42)
        r2 = run_all_baselines(y_true, y_returns, seed=42)
        for key in r1:
            for metric in ("accuracy", "macro_f1", "directional_accuracy"):
                assert r1[key][metric] == r2[key][metric]
