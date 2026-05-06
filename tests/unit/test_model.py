"""Tests for model module.

Tests LightGBM training helpers, class weight computation,
and deployment model metadata.
Meta-learner tests removed — pipeline now uses GRU + LightGBM directly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.shared.config import Config
from thesis.stage_4_training.lgbm.utils import (
    _build_interaction_constraints,
    _compute_class_weights,
    _compute_distribution_shift_weights,
)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.model.num_leaves = 4
    config.model.max_depth = 3
    config.model.learning_rate = 0.1
    config.model.n_estimators = 5
    config.model.min_child_samples = 10
    config.model.subsample = 0.8
    config.model.subsample_freq = 1
    config.model.feature_fraction = 0.8
    config.model.reg_alpha = 0.01
    config.model.reg_lambda = 0.01
    config.model.early_stopping_rounds = 5
    config.workflow.random_seed = 42
    config.workflow.n_jobs = 1
    config.splitting.purge_bars = 5
    return config


@pytest.fixture
def synthetic_classification_data():
    """Create synthetic classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Create simple decision boundary
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, np.where(X[:, 0] + X[:, 1] < -0.5, -1, 0))

    return X, y


@pytest.mark.unit
@pytest.mark.models
def test_compute_class_weights_returns_dict(synthetic_classification_data) -> None:
    """Test _compute_class_weights returns dict with classes as keys."""
    X, y = synthetic_classification_data

    weights = _compute_class_weights(y)

    assert isinstance(weights, dict)
    unique_classes = np.unique(y)
    for cls in unique_classes:
        assert int(cls) in weights
        assert isinstance(weights[int(cls)], float)
        assert weights[int(cls)] > 0


@pytest.mark.unit
@pytest.mark.models
def test_compute_class_weights_balanced(synthetic_classification_data) -> None:
    """Test that class weights are approximately balanced."""
    X, y = synthetic_classification_data

    weights = _compute_class_weights(y)

    class_counts = {cls: np.sum(y == cls) for cls in np.unique(y)}
    max_count = max(class_counts.values())

    for cls, count in class_counts.items():
        max_count / count
        actual_weight = weights[int(cls)]
        assert actual_weight > 0


@pytest.mark.unit
@pytest.mark.models
def test_compute_class_weights_single_class() -> None:
    """Test class weights with single class."""
    y = np.ones(100)

    weights = _compute_class_weights(y)

    assert isinstance(weights, dict)
    assert 1 in weights
    assert weights[1] > 0


@pytest.mark.unit
@pytest.mark.models
def test_class_weights_with_imbalanced_data() -> None:
    """Test class weights with highly imbalanced data."""
    y = np.array([1] * 90 + [0] * 5 + [-1] * 5)

    weights = _compute_class_weights(y)

    # Minority classes should have higher weights
    assert weights[0] > weights[1]
    assert weights[-1] > weights[1]


@pytest.mark.unit
@pytest.mark.models
def test_interaction_constraints_skip_empty_groups() -> None:
    """Pure-static or pure-GRU inputs should not emit empty constraint groups."""
    static_only = _build_interaction_constraints(["rsi_14", "atr_14"])
    gru_only = _build_interaction_constraints(["gru_h0", "gru_h1"])

    # Interaction constraints are currently disabled — returns empty list
    # to allow full cross-group interaction in LightGBM.
    assert static_only == []
    assert gru_only == []


# ──────────────────────────────────────────────────────────────────────────────
# _build_lgbm_info deployment metadata tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.models
class TestDeploymentModelMetadata:
    """Tests for _build_lgbm_info deployment model metadata."""

    @staticmethod
    def _make_mock_model(
        best_iteration: int = 50, classes: tuple = (0, 1, 2)
    ) -> MagicMock:
        """Create a mock LightGBM model with required attrs.

        Uses MagicMock so ``best_iteration_`` and ``classes_`` are directly
        settable as attributes without real LightGBM dependency.
        """
        model = MagicMock()
        model.best_iteration_ = best_iteration
        model.classes_ = classes
        return model

    def test_window_provenance_keys_present_with_kwargs(self) -> None:
        """When window_index is provided, provenance keys are in the dict."""
        from thesis.stage_4_training.walk_forward.artifacts import _build_lgbm_info

        model = self._make_mock_model()
        train_dates = {"start": "2023-01-01", "end": "2023-06-01"}
        test_dates = {"start": "2023-06-02", "end": "2023-07-01"}

        info = _build_lgbm_info(
            model,
            ["f1", "f2", "f3"],
            last_window_accuracy=0.85,
            window_index=5,
            total_windows=10,
            window_train_dates=train_dates,
            window_test_dates=test_dates,
        )

        assert info["window_index"] == 5
        assert info["total_windows"] == 10
        assert info["window_oof_accuracy"] == 0.85
        assert info["window_train_date_range"] == train_dates
        assert info["window_test_date_range"] == test_dates

    def test_backward_compatible_no_window_provenance_keys(self) -> None:
        """Missing kwargs → no crash and no window-provenance keys in result."""
        from thesis.stage_4_training.walk_forward.artifacts import _build_lgbm_info

        model = self._make_mock_model()
        info = _build_lgbm_info(model, ["f1", "f2"], last_window_accuracy=0.85)

        # Core keys always present
        for key in (
            "artifact_strategy",
            "validation_protocol",
            "last_window_accuracy",
            "best_iteration",
            "n_features",
            "n_classes",
        ):
            assert key in info, f"Expected key {key} not found in result"

        # Window provenance keys absent
        for key in (
            "window_index",
            "total_windows",
            "window_oof_accuracy",
            "window_train_date_range",
            "window_test_date_range",
        ):
            assert key not in info, f"Provenance key {key} should be absent"

    def test_metadata_includes_per_window_provenance(self) -> None:
        """Result includes per-window provenance when kwargs are supplied."""
        from thesis.stage_4_training.walk_forward.artifacts import _build_lgbm_info

        model = self._make_mock_model(best_iteration=75, classes=(0, 1, 2))
        train_dates = {"start": "2024-01-01", "end": "2024-06-01"}
        test_dates = {"start": "2024-06-02", "end": "2024-07-01"}

        info = _build_lgbm_info(
            model,
            ["a", "b", "c", "d"],
            last_window_accuracy=0.92,
            window_index=3,
            total_windows=6,
            window_train_dates=train_dates,
            window_test_dates=test_dates,
        )

        # Provenance keys present
        assert info["window_index"] == 3
        assert info["total_windows"] == 6
        assert info["window_oof_accuracy"] == 0.92
        assert info["window_train_date_range"] == train_dates
        assert info["window_test_date_range"] == test_dates

        # Core metadata correct
        assert info["n_features"] == 4
        assert info["n_classes"] == 3
        assert info["best_iteration"] == 75
        assert info["artifact_strategy"] == "last_walk_forward_window"

    def test_backward_compatible_with_none_accuracy(self) -> None:
        """None accuracy is handled without crash — key present with None."""
        from thesis.stage_4_training.walk_forward.artifacts import _build_lgbm_info

        model = self._make_mock_model()
        info = _build_lgbm_info(model, ["f1"], last_window_accuracy=None)

        assert info["last_window_accuracy"] is None
        assert "window_index" not in info

    def test_window_oof_accuracy_equals_last_window_accuracy(self) -> None:
        """window_oof_accuracy mirrors last_window_accuracy when set."""
        from thesis.stage_4_training.walk_forward.artifacts import _build_lgbm_info

        model = self._make_mock_model()

        for acc in (0.0, 0.5, 1.0, None):
            info = _build_lgbm_info(
                model,
                ["f"],
                last_window_accuracy=acc,
                window_index=1,
                total_windows=1,
            )
            assert info["window_oof_accuracy"] == acc
            assert info["last_window_accuracy"] == acc


# ──────────────────────────────────────────────────────────────────────────────
# _compute_distribution_shift_weights tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.models
class TestDistributionShiftWeights:
    """Tests for _compute_distribution_shift_weights time-safe weighting."""

    def test_matched_distributions_return_uniformish_weights(self) -> None:
        """When train and val have similar class distributions, weights ≈ 1.0."""
        rng = np.random.default_rng(42)
        # Matched distributions: both ~33% each class
        y_train = rng.choice([-1, 0, 1], size=900, p=[0.33, 0.34, 0.33])
        y_val = rng.choice([-1, 0, 1], size=100, p=[0.33, 0.34, 0.33])

        weights, _ = _compute_distribution_shift_weights(y_train, y_val)

        assert len(weights) == len(y_train)
        # All weights should be close to 1.0
        assert 0.8 < float(np.min(weights)) < 1.2, (
            f"Expected uniform-ish min weight, got {np.min(weights):.3f}"
        )
        assert 0.9 < float(np.mean(weights)) < 1.1, (
            f"Expected mean ≈ 1.0, got {np.mean(weights):.3f}"
        )

    def test_shifted_distributions_return_non_uniform_weights(self) -> None:
        """When val has a different class distribution, weights diverge from 1.0."""
        rng = np.random.default_rng(42)
        # Train: balanced. Val: heavily biased toward LONG (class 1)
        y_train = rng.choice([-1, 0, 1], size=900, p=[0.33, 0.34, 0.33])
        y_val = rng.choice([-1, 0, 1], size=100, p=[0.05, 0.05, 0.90])

        weights, _ = _compute_distribution_shift_weights(y_train, y_val)

        assert len(weights) == len(y_train)
        # LONG samples should be up-weighted (LONG more common in val)
        long_mask = y_train == 1
        short_mask = y_train == -1
        mean_long = float(weights[long_mask].mean())
        mean_short = float(weights[short_mask].mean())
        assert mean_long > mean_short, (
            f"LONG weights ({mean_long:.3f}) should exceed SHORT weights "
            f"({mean_short:.3f}) when val is LONG-heavy"
        )
        # There should be variance in weights (not all uniform)
        assert np.std(weights) > 0.01, (
            f"Expected non-uniform weights, got std={np.std(weights):.5f}"
        )

    def test_weights_aligned_to_y_train_not_y_val(self) -> None:
        """Weights array length matches y_train, not y_val."""
        rng = np.random.default_rng(42)
        y_train = rng.choice([-1, 0, 1], size=750)
        y_val = rng.choice([-1, 0, 1], size=250)

        weights, _ = _compute_distribution_shift_weights(y_train, y_val)

        assert len(weights) == len(y_train), (
            f"Weights length {len(weights)} should match y_train length {len(y_train)}"
        )

    def test_all_weights_within_clip_bounds(self) -> None:
        """No individual weight exceeds the default clip range [0.5, 3.0]."""
        rng = np.random.default_rng(42)
        # Extreme shift: val is entirely one class
        y_train = rng.choice([-1, 0, 1], size=900, p=[0.50, 0.25, 0.25])
        y_val = np.full(100, 1, dtype=np.int32)  # All LONG

        weights, _ = _compute_distribution_shift_weights(y_train, y_val)

        assert float(np.min(weights)) >= 0.5, (
            f"Min weight {np.min(weights):.3f} below clip floor 0.5"
        )
        assert float(np.max(weights)) <= 3.0, (
            f"Max weight {np.max(weights):.3f} above clip ceiling 3.0"
        )

    def test_uniform_val_distribution_yields_uniform_weights(self) -> None:
        """A perfectly uniform val distribution produces uniform weights."""
        rng = np.random.default_rng(42)
        y_train = rng.choice([-1, 0, 1], size=900, p=[0.33, 0.34, 0.33])
        y_val = np.array([-1, 0, 1] * 34, dtype=np.int32)[:100]

        weights, _ = _compute_distribution_shift_weights(y_train, y_val)

        # All classes have the same weights
        unique_weights = set(round(w, 6) for w in weights)
        assert len(unique_weights) <= 3, (
            f"Expected at most 3 distinct weight values, got {len(unique_weights)}"
        )

    def test_zero_train_class_handled_gracefully(self) -> None:
        """Absent train classes should not crash — handled by denominator guard."""
        y_train = np.array([1] * 800 + [0] * 200, dtype=np.int32)  # No SHORT
        y_val = np.array([-1] * 30 + [1] * 50 + [0] * 20, dtype=np.int32)

        weights, _ = _compute_distribution_shift_weights(y_train, y_val)

        assert len(weights) == len(y_train)
        assert np.all(np.isfinite(weights)), "All weights must be finite"

    def test_single_class_train(self) -> None:
        """Degenerate case: single-class train should still return valid weights."""
        y_train = np.full(500, 1, dtype=np.int32)
        y_val = np.array([-1, 0, 1] * 33 + [1], dtype=np.int32)[:100]

        weights, _ = _compute_distribution_shift_weights(y_train, y_val)

        assert len(weights) == len(y_train)
        assert np.all(np.isfinite(weights)), "All weights must be finite"


# ──────────────────────────────────────────────────────────────────────────────
# _lgbm: _normalize_label and _save_predictions
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.models
class TestNormalizeLabel:
    def test_positive(self) -> None:
        from thesis.stage_4_training.lgbm.training import _normalize_label

        assert _normalize_label(0) == "0"
        assert _normalize_label(1) == "1"

    def test_negative(self) -> None:
        from thesis.stage_4_training.lgbm.training import _normalize_label

        assert _normalize_label(-1) == "minus1"
        assert _normalize_label(-2) == "minus2"


@pytest.mark.unit
@pytest.mark.models
class TestSavePredictions:
    def test_saves_parquet_and_csv(self, tmp_path) -> None:
        from thesis.stage_4_training.lgbm.training import _save_predictions

        n = 10
        timestamps = pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1) + pl.duration(hours=n - 1),
            interval="1h",
            eager=True,
        )
        test_aligned = pl.DataFrame({"timestamp": timestamps})
        y_test = np.array([1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
        preds = np.array([1, -1, 0, 0, -1, 0, 1, 0, 0, 1])
        proba = np.random.RandomState(42).rand(n, 3)
        proba = proba / proba.sum(axis=1, keepdims=True)
        class_order = [-1, 0, 1]

        preds_path = tmp_path / "predictions.parquet"
        _save_predictions(test_aligned, y_test, preds, proba, class_order, preds_path)

        assert preds_path.exists()
        csv_path = preds_path.with_suffix(".csv")
        assert csv_path.exists()

        df = pl.read_parquet(preds_path)
        assert "timestamp" in df.columns
        assert "true_label" in df.columns
        assert "pred_label" in df.columns
        assert "pred_proba_class_minus1" in df.columns
        assert "pred_proba_class_0" in df.columns
        assert "pred_proba_class_1" in df.columns
        assert len(df) == n


# ──────────────────────────────────────────────────────────────────────────────
# _lgbm_utils: _wrap_np, _align_splits_with_sequences, _build_hybrid_matrix,
#              _filter_validation_to_seen_classes, _save_feature_importance
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.models
class TestWrapNp:
    def test_wraps_as_dataframe(self) -> None:
        from thesis.stage_4_training.lgbm.utils import _wrap_np
        import pandas as pd

        X = np.array([[1, 2], [3, 4]])
        result = _wrap_np(X, ["a", "b"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert result.shape == (2, 2)


@pytest.mark.unit
@pytest.mark.models
class TestAlignSplitsWithSequences:
    def test_slices_correctly(self) -> None:
        from thesis.stage_4_training.lgbm.utils import _align_splits_with_sequences

        df = pl.DataFrame({"a": range(20), "b": range(20, 40)})
        seq_len = 5
        hidden = np.zeros((16, 3))  # 20 - seq_len + 1 = 16

        train_a, val_a, test_a = _align_splits_with_sequences(
            df, df, df, hidden, hidden, hidden, seq_len
        )
        assert len(train_a) == 16
        assert len(val_a) == 16
        assert len(test_a) == 16

    def test_alignment_values(self) -> None:
        from thesis.stage_4_training.lgbm.utils import _align_splits_with_sequences

        df = pl.DataFrame({"val": np.arange(10.0)})
        seq_len = 3
        hidden = np.zeros((8, 2))

        aligned, _, _ = _align_splits_with_sequences(
            df, df, df, hidden, hidden, hidden, seq_len
        )
        # Should start from index seq_len-1 = 2
        assert aligned["val"][0] == 2.0
        assert len(aligned) == 8


@pytest.mark.unit
@pytest.mark.models
class TestBuildHybridMatrix:
    def test_concatenates_hidden_and_static(self) -> None:
        from thesis.stage_4_training.lgbm.utils import _build_hybrid_matrix

        n = 10
        hidden_size = 4
        hidden = np.ones((n, hidden_size))
        static_cols = ["s1", "s2"]

        df = pl.DataFrame(
            {"s1": np.arange(n, dtype=float), "s2": np.arange(n, 10 + n, dtype=float)}
        )

        X_train, X_val, X_test, feat_cols = _build_hybrid_matrix(
            hidden, hidden, hidden, df, df, df, static_cols, hidden_size
        )

        assert X_train.shape == (n, hidden_size + len(static_cols))
        assert len(feat_cols) == hidden_size + len(static_cols)
        assert feat_cols[0] == "gru_h0"
        assert feat_cols[-1] == "s2"
        # First 4 cols are ones (hidden), next 2 are arange
        np.testing.assert_array_equal(X_train[:, 0], np.ones(n))


@pytest.mark.unit
@pytest.mark.models
class TestFilterValidationToSeenClasses:
    def test_all_seen(self) -> None:
        from thesis.stage_4_training.lgbm.utils import (
            _filter_validation_to_seen_classes,
        )

        X_train = np.random.randn(20, 5)
        X_val = np.random.randn(10, 5)
        y_val = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1, 0])
        y_train = np.array([-1, 0, 1] * 6 + [-1, 0])

        result = _filter_validation_to_seen_classes(
            X_train, X_val, y_val, y_train, ["f" + str(i) for i in range(5)]
        )
        assert result is not None
        X_filt, y_filt = result
        assert len(y_filt) == len(y_val)

    def test_unseen_class_dropped(self) -> None:
        from thesis.stage_4_training.lgbm.utils import (
            _filter_validation_to_seen_classes,
        )

        X_train = np.random.randn(10, 3)
        X_val = np.random.randn(5, 3)
        y_train = np.array([-1, 0] * 5)  # No class 1
        y_val = np.array([-1, 0, 1, 1, 0])

        result = _filter_validation_to_seen_classes(
            X_train, X_val, y_val, y_train, ["a", "b", "c"]
        )
        assert result is not None
        _, y_filt = result
        assert 1 not in y_filt
        assert len(y_filt) == 3

    def test_no_overlap_returns_none(self) -> None:
        from thesis.stage_4_training.lgbm.utils import (
            _filter_validation_to_seen_classes,
        )

        X_train = np.random.randn(10, 3)
        X_val = np.random.randn(5, 3)
        y_train = np.array([-1] * 10)
        y_val = np.array([1] * 5)

        result = _filter_validation_to_seen_classes(
            X_train, X_val, y_val, y_train, ["a", "b", "c"]
        )
        assert result is None


@pytest.mark.unit
@pytest.mark.models
class TestSaveFeatureImportance:
    def test_saves_json(self, tmp_path) -> None:
        from thesis.stage_4_training.lgbm.utils import _save_feature_importance

        model = MagicMock()
        model.feature_importances_ = np.array([10, 30, 20])
        feat_cols = ["a", "b", "c"]

        config = Config()
        config.paths.session_dir = str(tmp_path)

        _save_feature_importance(model, feat_cols, config)

        json_path = tmp_path / "reports" / "feature_importance.json"
        assert json_path.exists()
        import json

        with open(json_path) as f:
            data = json.load(f)
        # Sorted descending
        assert list(data.keys())[0] == "b"
        assert data["b"] == 30.0

    def test_handles_missing_session_dir(self, tmp_path) -> None:
        from thesis.stage_4_training.lgbm.utils import _save_feature_importance

        model = MagicMock()
        model.feature_importances_ = np.array([5, 10])
        feat_cols = ["x", "y"]

        config = Config()
        config.paths.session_dir = ""

        _save_feature_importance(model, feat_cols, config)

        # Falls back to results/feature_importance.json
        import json

        fallback = Path("results/feature_importance.json")
        assert fallback.exists()
        with open(fallback) as f:
            data = json.load(f)
        assert "x" in data
        # Cleanup
        fallback.unlink()
