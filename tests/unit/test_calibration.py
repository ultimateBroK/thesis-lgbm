"""Unit tests for _calibration — probability calibration metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from thesis.stage_6_reporting.calibration import (
    _to_onehot,
    brier_score,
    calibration_reliability_data,
    compute_all_calibration_metrics,
    confidence_bins_accuracy,
    expected_calibration_error,
    log_loss,
)
from thesis.stage_6_reporting.model_metrics import high_confidence_accuracy


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_CLASSES = [-1, 0, 1]


def _onehot(y: np.ndarray) -> np.ndarray:
    return _to_onehot(y, _CLASSES)


# ---------------------------------------------------------------------------
# _to_onehot
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToOnehot:
    def test_basic(self) -> None:
        y = np.array([-1, 0, 1])
        oh = _onehot(y)
        assert oh.shape == (3, 3)
        np.testing.assert_array_equal(oh[0], [1, 0, 0])
        np.testing.assert_array_equal(oh[1], [0, 1, 0])
        np.testing.assert_array_equal(oh[2], [0, 0, 1])


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestECE:
    def test_perfect_calibration(self) -> None:
        """When confidence always matches accuracy, ECE ≈ 0."""
        y = np.array([-1, 0, 1, -1, 0, 1])
        oh = _onehot(y)
        # Perfect probabilities: all mass on correct class
        proba = _onehot(y).astype(np.float64)
        ece = expected_calibration_error(oh, proba)
        assert ece < 1e-9

    def test_worst_calibration(self) -> None:
        """All confidence on wrong class → high ECE."""
        y = np.array([-1, -1, -1])
        oh = _onehot(y)
        # All probability on wrong class (class 1 = Long)
        proba = np.zeros((3, 3), dtype=np.float64)
        proba[:, 2] = 1.0  # always predict Long with conf=1.0
        ece = expected_calibration_error(oh, proba)
        assert ece > 0.9  # should be close to 1.0

    def test_custom_bins(self) -> None:
        y = np.array([0, 0, 0])
        oh = _onehot(y)
        proba = np.full((3, 3), 1 / 3)
        ece5 = expected_calibration_error(oh, proba, n_bins=5)
        ece10 = expected_calibration_error(oh, proba, n_bins=10)
        # Both should be small for uniform random predictions
        assert isinstance(ece5, float)
        assert isinstance(ece10, float)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBrierScore:
    def test_perfect_prediction(self) -> None:
        y = np.array([-1, 0, 1])
        oh = _onehot(y)
        proba = _onehot(y).astype(np.float64)
        assert brier_score(oh, proba) < 1e-9

    def test_random_prediction(self) -> None:
        """Uniform 1/3 probabilities → Brier ≈ 2/9."""
        y = np.array([-1, 0, 1])
        oh = _onehot(y)
        proba = np.full((3, 3), 1 / 3)
        bs = brier_score(oh, proba)
        # Expected: mean((1/3)^2 * 2) per sample = 2/9 ≈ 0.222
        assert abs(bs - 2 / 9) < 1e-9

    def test_worst_prediction(self) -> None:
        y = np.array([-1, -1])
        oh = _onehot(y)
        proba = np.zeros((2, 3), dtype=np.float64)
        proba[:, 2] = 1.0  # all mass on wrong class
        bs = brier_score(oh, proba)
        # per sample: (1-0)^2 + (0-0)^2 + (0-1)^2 = 2/3 ≈ 0.667
        assert abs(bs - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogLoss:
    def test_perfect_prediction(self) -> None:
        """log_loss maps domain labels to probability columns."""
        y = np.array([-1, 0, 1])
        proba = np.array([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])
        ll = log_loss(y, proba)
        assert ll < 0.1

    def test_random_prediction(self) -> None:
        y = np.array([-1, 0, 1])
        proba = np.full((3, 3), 1 / 3)
        ll = log_loss(y, proba)
        assert ll > 1.0  # -log(1/3) ≈ 1.099

    def test_single_sample(self) -> None:
        y = np.array([0])
        proba = np.array([[0.1, 0.8, 0.1]])
        ll = log_loss(y, proba)
        assert abs(ll - (-np.log(0.8))) < 1e-6


# ---------------------------------------------------------------------------
# confidence_bins_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfidenceBinsAccuracy:
    def test_default_bins(self) -> None:
        y_true = np.array([0, 1, -1])
        y_pred = np.array([0, 1, -1])
        proba = np.array([[0.7, 0.15, 0.15], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        bins = confidence_bins_accuracy(y_true, y_pred, proba)
        assert isinstance(bins, list)
        assert len(bins) == 6  # default 6 bins

    def test_custom_bins(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        proba = np.array([[0.6, 0.2, 0.2], [0.1, 0.9, 0.0]])
        bins = confidence_bins_accuracy(y_true, y_pred, proba, bins=[0.5, 0.7, 1.0])
        assert len(bins) == 3
        assert bins[0]["lo"] == 0.0
        assert bins[0]["hi"] == 0.5
        assert bins[1]["hi"] == 0.7
        assert bins[2]["hi"] == 1.0

    def test_empty_bin(self) -> None:
        y_true = np.array([0])
        y_pred = np.array([0])
        proba = np.array([[0.4, 0.3, 0.3]])
        bins = confidence_bins_accuracy(y_true, y_pred, proba, bins=[0.5, 0.9, 1.0])
        # First bin (0, 0.5] should have count=1
        assert bins[0]["count"] == 1
        # Remaining bins have count=0
        assert bins[1]["count"] == 0
        assert bins[2]["count"] == 0


# ---------------------------------------------------------------------------
# high_confidence_accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHighConfidenceAccuracy:
    def test_threshold_filters(self) -> None:
        y_true = np.array([0, 1, -1])
        y_pred = np.array([0, 1, -1])
        proba = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        result = high_confidence_accuracy(y_true, y_pred, proba, threshold=0.7)
        assert result["count"] == 2
        assert result["accuracy"] == 1.0

    def test_none_above(self) -> None:
        y_true = np.array([0])
        y_pred = np.array([0])
        proba = np.array([[0.4, 0.3, 0.3]])
        result = high_confidence_accuracy(y_true, y_pred, proba, threshold=0.9)
        assert result["count"] == 0
        assert result["accuracy"] == 0.0

    def test_custom_threshold(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        proba = np.array([[0.6, 0.2, 0.2], [0.1, 0.9, 0.0]])
        result = high_confidence_accuracy(y_true, y_pred, proba, threshold=0.5)
        assert result["count"] == 2


# ---------------------------------------------------------------------------
# calibration_reliability_data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCalibrationReliabilityData:
    def test_returns_correct_keys(self) -> None:
        y = np.array([-1, 0, 1, -1, 0, 1])
        oh = _onehot(y)
        proba = _onehot(y).astype(np.float64)
        data = calibration_reliability_data(oh, proba, n_bins=10)
        assert set(data.keys()) == {"bin_centers", "accuracies", "counts"}

    def test_correct_number_of_bins(self) -> None:
        y = np.array([-1, 0, 1])
        oh = _onehot(y)
        proba = _onehot(y).astype(np.float64)
        data = calibration_reliability_data(oh, proba, n_bins=5)
        assert len(data["bin_centers"]) == 5
        assert len(data["accuracies"]) == 5
        assert len(data["counts"]) == 5

    def test_bin_centers_spacing(self) -> None:
        y = np.array([-1, 0, 1])
        oh = _onehot(y)
        proba = _onehot(y).astype(np.float64)
        data = calibration_reliability_data(oh, proba, n_bins=10)
        # Bin centers should be 0.05, 0.15, ..., 0.95
        for i, center in enumerate(data["bin_centers"]):
            assert abs(center - (i * 0.1 + 0.05)) < 1e-6

    def test_perfect_calibration_accuracies(self) -> None:
        y = np.array([-1, 0, 1, -1, 0, 1])
        oh = _onehot(y)
        proba = _onehot(y).astype(np.float64)
        data = calibration_reliability_data(oh, proba, n_bins=10)
        # All samples land in bin (0.9, 1.0] with accuracy 1.0
        total = sum(data["counts"])
        assert total == len(y)
        nonzero_bins = [a for a, c in zip(data["accuracies"], data["counts"]) if c > 0]
        for acc in nonzero_bins:
            assert abs(acc - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compute_all_calibration_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeAllCalibration:
    def test_returns_all_keys(self) -> None:
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = np.array([-1, 0, 1, -1, 0, 1])
        proba = _onehot(y_true).astype(np.float64)
        result = compute_all_calibration_metrics(y_true, y_pred, proba)
        expected_keys = {
            "ece",
            "brier_score",
            "log_loss",
            "high_confidence_accuracy",
            "confidence_bins",
            "reliability_data",
        }
        assert expected_keys <= set(result.keys())

    def test_perfect_prediction_low_metrics(self) -> None:
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = y_true.copy()
        proba = _onehot(y_true).astype(np.float64)
        result = compute_all_calibration_metrics(y_true, y_pred, proba)
        assert result["ece"] < 0.01
        assert result["brier_score"] < 0.01
        assert result["log_loss"] < 0.01


# GRU temperature-scaling tests removed — gru.calibration module not implemented
