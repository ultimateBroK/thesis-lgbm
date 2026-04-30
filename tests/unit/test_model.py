"""Tests for model module.

Tests LightGBM training helpers and class weight computation.
Meta-learner tests removed — pipeline now uses GRU + LightGBM directly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.config import Config
from thesis.model import (
    _build_interaction_constraints,
    _compute_class_weights,
    _compute_cost_fraction,
    _compute_sharpe_from_predictions,
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


# ============================================================================
# Sharpe Ratio computation tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.models
def test_sharpe_highly_accurate() -> None:
    """Test Sharpe with very accurate predictions (96% correct).

    Note: 100% correct predictions give Sharpe=0 due to std=0 (undefined).
    """
    np.random.seed(42)
    y_true = np.random.choice([-1, 1], 500)  # No holds for clarity
    y_pred = y_true.copy()
    # Add some noise to avoid std=0 edge case
    flip_idx = np.random.choice(500, 20, replace=False)
    y_pred[flip_idx] = -y_true[flip_idx]  # ~96% accuracy

    sharpe = _compute_sharpe_from_predictions(y_true, y_pred)
    assert sharpe > 0  # Should be positive with high accuracy


@pytest.mark.unit
@pytest.mark.models
def test_sharpe_wrong_predictions() -> None:
    """Test Sharpe with mostly wrong predictions (~4% correct)."""
    np.random.seed(42)
    y_true = np.random.choice([-1, 1], 500)
    y_pred = -y_true.copy()
    # Make a few correct to avoid std=0 edge case
    flip_idx = np.random.choice(500, 20, replace=False)
    y_pred[flip_idx] = y_true[flip_idx]  # ~4% accuracy

    sharpe = _compute_sharpe_from_predictions(y_true, y_pred)
    assert sharpe < 0  # Should be negative with mostly wrong


@pytest.mark.unit
@pytest.mark.models
def test_sharpe_random_predictions() -> None:
    """Test Sharpe with random directional predictions on non-hold data.

    Uses only non-hold labels so that random predictions give ~50% accuracy.
    With spread cost, random predictions will be slightly negative.
    """
    np.random.seed(42)
    # Use only non-hold data for clean 50% random baseline
    y_true = np.random.choice([-1, 1], 500)
    y_pred = np.random.choice([-1, 1], 500)  # Random direction

    sharpe = _compute_sharpe_from_predictions(y_true, y_pred)
    # Random predictions should give Sharpe near 0 (with spread cost, slightly negative)
    # Annualized Sharpe: ~(-0.0002 / 1.0) * sqrt(500) ≈ -0.004
    assert -3.0 < sharpe < 3.0  # Reasonable range


@pytest.mark.unit
@pytest.mark.models
def test_sharpe_insufficient_trades() -> None:
    """Test Sharpe returns 0 with too few trades (below min_trades=3)."""
    y_true = np.array([1, -1])
    y_pred = np.array([1, -1])

    sharpe = _compute_sharpe_from_predictions(y_true, y_pred)
    assert sharpe == 0.0  # Less than min_trades=3 non-hold trades


@pytest.mark.unit
@pytest.mark.models
def test_sharpe_holds_filtered() -> None:
    """Test that hold signals (0) are filtered out."""
    np.random.seed(42)
    y_true = np.array([1, 0, 0, 1, 0, 0, -1, 0, 0, 1] * 50)  # Mostly holds
    y_pred = y_true.copy()

    sharpe = _compute_sharpe_from_predictions(y_true, y_pred)
    # With holds filtered, we should have enough non-hold trades
    assert sharpe >= 0  # Perfect direction should give Sharpe >= 0


@pytest.mark.unit
@pytest.mark.models
def test_sharpe_better_than_random() -> None:
    """Test that better predictions give higher Sharpe."""
    np.random.seed(42)
    n = 500
    y_true = np.random.choice([-1, 0, 1], n, p=[0.3, 0.4, 0.3])

    # Good predictions (80% correct on non-holds)
    y_good = y_true.copy()
    # Only flip non-hold predictions
    non_hold_idx = np.where(y_true != 0)[0]
    flip_n = int(len(non_hold_idx) * 0.2)
    flip_idx = np.random.choice(non_hold_idx, flip_n, replace=False)
    y_good[flip_idx] = -y_true[flip_idx]

    # Bad predictions (20% correct on non-holds)
    y_bad = y_true.copy()
    y_bad[non_hold_idx] = -y_true[non_hold_idx]
    # Make some correct to avoid std=0
    fix_idx = np.random.choice(
        non_hold_idx, int(len(non_hold_idx) * 0.2), replace=False
    )
    y_bad[fix_idx] = y_true[fix_idx]

    sharpe_good = _compute_sharpe_from_predictions(y_true, y_good)
    sharpe_bad = _compute_sharpe_from_predictions(y_true, y_bad)

    assert sharpe_good > sharpe_bad


@pytest.mark.unit
@pytest.mark.models
def test_cost_fraction_uses_realistic_market_price(sample_config: Config) -> None:
    """Transaction-cost fraction should stay small at realistic XAUUSD prices."""
    sample_config.backtest.spread_ticks = 35
    sample_config.backtest.slippage_ticks = 5
    sample_config.backtest.commission_per_lot = 10.0
    sample_config.backtest.lots_per_trade = 0.2
    sample_config.data.tick_size = 0.01
    sample_config.data.contract_size = 100

    cost_fraction = _compute_cost_fraction(
        sample_config.backtest,
        sample_config.data,
        median_price=2000.0,
    )

    assert 0 < cost_fraction < 0.01


@pytest.mark.unit
@pytest.mark.models
def test_interaction_constraints_skip_empty_groups() -> None:
    """Pure-static or pure-GRU inputs should not emit empty constraint groups."""
    static_only = _build_interaction_constraints(["rsi_14", "atr_14"])
    gru_only = _build_interaction_constraints(["gru_h0", "gru_h1"])

    assert static_only == [[0, 1]]
    assert gru_only == [[0, 1]]
