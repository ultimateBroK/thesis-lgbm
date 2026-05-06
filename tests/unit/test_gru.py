"""Tests for GRU feature extractor module.

Tests cover: model architecture, sequence preparation, hidden state extraction,
training loop (synthetic data), save/load round-trip.
"""

from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.shared.config import Config
from thesis.stage_4_training.gru import (
    GRUExtractor,
    SequenceDataset,
    extract_hidden_states,
    load_gru_classifier,
    load_gru_model,
    predict_gru_proba,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis.stage_4_training.gru.data import _sliding_windows

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gru_config() -> Config:
    """Minimal config for fast GRU tests."""
    config = Config()
    config.gru.objective = "multiclass"  # tests use label column, not regression_target
    config.gru.input_size = 4
    config.gru.hidden_size = 16
    config.gru.num_layers = 1
    config.gru.sequence_length = 5
    config.gru.dropout = 0.0
    config.gru.learning_rate = 0.01
    config.gru.batch_size = 8
    config.gru.epochs = 3
    config.gru.patience = 2
    config.workflow.random_seed = 42
    config.labels.num_classes = 3
    config.paths.model = "models/lightgbm_model.pkl"
    config.paths.session_dir = ""
    return config


@pytest.fixture
def synthetic_df() -> pl.DataFrame:
    """Create a synthetic DataFrame with GRU input columns + labels."""
    from datetime import datetime, timedelta

    n = 100
    rng = np.random.RandomState(42)
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": 2000.0 + rng.randn(n).cumsum(),
            "log_returns": rng.randn(n) * 0.001,
            "rsi_14": rng.uniform(20, 80, n),
            "atr_14": rng.uniform(5, 30, n),
            "macd_hist": rng.randn(n) * 0.5,
            "label": rng.choice([-1, 0, 1], n),
        }
    )


@pytest.fixture
def synthetic_sequences(synthetic_df: pl.DataFrame) -> np.ndarray:
    """Pre-built sequences for fast testing."""
    gru_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist"]
    data = synthetic_df.select(gru_cols).to_numpy()
    return _sliding_windows(data, 5)


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.models
def test_gru_extractor_output_shape() -> None:
    """GRUExtractor should output (batch, hidden_size) tensor."""
    model = GRUExtractor(input_size=2, hidden_size=32, num_layers=1, dropout=0.0)
    x = torch.randn(8, 5, 2)  # batch=8, seq_len=5, features=2
    out = model(x)
    assert out.shape == (8, 32)


@pytest.mark.unit
@pytest.mark.models
def test_gru_extractor_multi_layer() -> None:
    """Multi-layer GRU should still output (batch, hidden_size)."""
    model = GRUExtractor(input_size=2, hidden_size=64, num_layers=3, dropout=0.2)
    x = torch.randn(4, 10, 2)
    out = model(x)
    assert out.shape == (4, 64)


@pytest.mark.unit
@pytest.mark.models
def test_gru_extractor_deterministic() -> None:
    """Same input should produce same output in eval mode."""
    model = GRUExtractor(input_size=2, hidden_size=16, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randn(2, 5, 2)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


@pytest.mark.unit
@pytest.mark.models
def test_gru_no_dropout_single_layer() -> None:
    """Single-layer GRU should have zero dropout (no layer to apply it between)."""
    model = GRUExtractor(input_size=2, hidden_size=16, num_layers=1, dropout=0.3)
    # GRU dropout is only applied between layers, so single-layer should be dropout-free
    assert model.gru.dropout == 0.0


# ---------------------------------------------------------------------------
# SequenceDataset
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.models
def test_sequence_dataset_with_labels() -> None:
    """SequenceDataset should return (seq, label) tuples."""
    seqs = np.random.randn(10, 5, 2).astype(np.float32)
    labels = np.random.randint(0, 3, 10)
    ds = SequenceDataset(seqs, labels)

    assert len(ds) == 10
    seq, lab = ds[0]
    assert seq.shape == (5, 2)
    assert isinstance(lab, torch.Tensor)


@pytest.mark.unit
@pytest.mark.models
def test_sequence_dataset_without_labels() -> None:
    """SequenceDataset without labels should return (seq, None)."""
    seqs = np.random.randn(10, 5, 2).astype(np.float32)
    ds = SequenceDataset(seqs, labels=None)

    seq, lab = ds[0]
    assert seq.shape == (5, 2)
    assert lab is None


# ---------------------------------------------------------------------------
# Sliding Windows
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.models
def test_sliding_windows_shape() -> None:
    """Sliding window should produce correct 3D shape."""
    data = np.arange(20).reshape(10, 2).astype(np.float64)
    windows = _sliding_windows(data, 3)
    assert windows.shape == (8, 3, 2)  # n_samples = 10 - 3 + 1


@pytest.mark.unit
@pytest.mark.models
def test_sliding_windows_content() -> None:
    """Sliding windows should contain correct consecutive values."""
    data = np.arange(6).reshape(3, 2).astype(np.float64)
    windows = _sliding_windows(data, 2)
    # Expected: first window = [[0,1], [2,3]], second = [[2,3], [4,5]]
    assert windows.shape == (2, 2, 2)
    np.testing.assert_array_equal(windows[0, 0], [0, 1])
    np.testing.assert_array_equal(windows[0, 1], [2, 3])
    np.testing.assert_array_equal(windows[1, 0], [2, 3])
    np.testing.assert_array_equal(windows[1, 1], [4, 5])


# ---------------------------------------------------------------------------
# prepare_sequences
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.models
def test_prepare_sequences_shapes(synthetic_df: pl.DataFrame) -> None:
    """prepare_sequences should return correct shapes."""
    gru_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist"]
    seq_len = 5

    seqs, labels, static_cols = prepare_sequences(synthetic_df, gru_cols, seq_len)

    n_expected = len(synthetic_df) - seq_len + 1
    assert seqs.shape == (n_expected, seq_len, 4)
    assert labels is not None
    assert labels.shape == (n_expected,)
    assert "close" in static_cols  # non-excluded column


@pytest.mark.unit
@pytest.mark.models
def test_prepare_sequences_label_alignment(synthetic_df: pl.DataFrame) -> None:
    """Labels should align to end of each window."""
    gru_cols = ["log_returns", "rsi_14", "atr_14", "macd_hist"]
    seq_len = 5

    _, labels, _ = prepare_sequences(synthetic_df, gru_cols, seq_len)

    # First label should be at index seq_len-1
    expected_label = synthetic_df["label"].to_numpy()[seq_len - 1]
    assert labels[0] == expected_label


@pytest.mark.unit
@pytest.mark.models
def test_prepare_sequences_missing_col() -> None:
    """Should raise ValueError if GRU column is missing."""
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0], "label": [0, 1, -1]})
    with pytest.raises(ValueError, match="not found"):
        prepare_sequences(df, ["nonexistent_col"], 2)


@pytest.mark.unit
@pytest.mark.models
def test_prepare_sequences_too_short() -> None:
    """Should raise ValueError if DataFrame is shorter than sequence length."""
    df = pl.DataFrame(
        {
            "log_returns": [0.1],
            "rsi_14": [50.0],
            "label": [0],
        }
    )
    with pytest.raises(ValueError, match="need at least"):
        prepare_sequences(df, ["log_returns", "rsi_14"], 5)


# ---------------------------------------------------------------------------
# Hidden State Extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.models
def test_extract_hidden_states_shape() -> None:
    """Hidden states should be (n_samples, hidden_size)."""
    model = GRUExtractor(input_size=2, hidden_size=32, num_layers=1)
    model.eval()
    seqs = np.random.randn(10, 5, 2).astype(np.float32)
    hidden = extract_hidden_states(model, seqs, batch_size=4)
    assert hidden.shape == (10, 32)


@pytest.mark.unit
@pytest.mark.models
def test_extract_hidden_states_no_nan() -> None:
    """Hidden states should not contain NaN values."""
    model = GRUExtractor(input_size=2, hidden_size=16, num_layers=1)
    model.eval()
    seqs = np.random.randn(20, 5, 2).astype(np.float32)
    hidden = extract_hidden_states(model, seqs, batch_size=8)
    assert not np.isnan(hidden).any()


@pytest.mark.unit
@pytest.mark.models
def test_extract_hidden_states_deterministic() -> None:
    """Same sequences should produce same hidden states."""
    model = GRUExtractor(input_size=2, hidden_size=16, num_layers=1)
    model.eval()
    seqs = np.random.randn(10, 5, 2).astype(np.float32)
    h1 = extract_hidden_states(model, seqs, batch_size=4)
    h2 = extract_hidden_states(model, seqs, batch_size=4)
    np.testing.assert_array_almost_equal(h1, h2)


# ---------------------------------------------------------------------------
# Save / Load Round-Trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.models
def test_save_load_round_trip(gru_config: Config, tmp_path: Path) -> None:
    """Save and load should produce identical model outputs."""
    model = GRUExtractor(
        input_size=gru_config.gru.input_size,
        hidden_size=gru_config.gru.hidden_size,
        num_layers=gru_config.gru.num_layers,
        dropout=gru_config.gru.dropout,
    )
    model.eval()

    save_path = tmp_path / "gru_test.pt"
    save_gru_model(model, gru_config, save_path)

    loaded_model, metadata = load_gru_model(save_path)

    assert metadata["hidden_size"] == gru_config.gru.hidden_size
    assert metadata["input_size"] == gru_config.gru.input_size
    assert metadata["num_layers"] == gru_config.gru.num_layers

    # Verify identical outputs
    x = torch.randn(4, 5, gru_config.gru.input_size)
    with torch.no_grad():
        out_orig = model(x)
        out_loaded = loaded_model(x)
    assert torch.allclose(out_orig, out_loaded)


@pytest.mark.unit
@pytest.mark.models
def test_load_nonexistent_file() -> None:
    """load_gru_model should raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_gru_model("/nonexistent/path/model.pt")


@pytest.mark.unit
@pytest.mark.models
def test_save_load_round_trip_with_classifier(
    gru_config: Config, tmp_path: Path
) -> None:
    """Classifier-aware checkpoints should reload the same probabilities."""
    model = GRUExtractor(
        input_size=gru_config.gru.input_size,
        hidden_size=gru_config.gru.hidden_size,
        num_layers=gru_config.gru.num_layers,
        dropout=gru_config.gru.dropout,
    )
    classifier = torch.nn.Linear(
        gru_config.gru.hidden_size, gru_config.labels.num_classes
    )
    model.eval()
    classifier.eval()

    sequences = np.random.randn(6, 5, gru_config.gru.input_size).astype(np.float32)
    expected = predict_gru_proba(model, classifier, sequences, batch_size=3)

    save_path = tmp_path / "gru_with_classifier.pt"
    save_gru_model(model, gru_config, save_path, classifier=classifier)

    loaded_model, metadata = load_gru_model(save_path)
    loaded_classifier = load_gru_classifier(metadata)
    actual = predict_gru_proba(loaded_model, loaded_classifier, sequences, batch_size=3)

    np.testing.assert_allclose(actual, expected)


# ---------------------------------------------------------------------------
# Training (Integration-style, but fast with tiny model)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.models
def test_train_gru_returns_correct_shapes(
    gru_config: Config, synthetic_df: pl.DataFrame
) -> None:
    """train_gru should return (model, train_hidden, val_hidden) with correct shapes."""
    # Split synthetic data into train/val
    n = len(synthetic_df)
    split = int(n * 0.8)
    train_df = synthetic_df.slice(0, split)
    val_df = synthetic_df.slice(split)

    model, _classifier, train_hidden, val_hidden, history, mean, std, _gru_cols = (
        train_gru(gru_config, train_df, val_df)
    )

    seq_len = gru_config.gru.sequence_length
    expected_train = len(train_df) - seq_len + 1
    expected_val = len(val_df) - seq_len + 1

    # torch.compile wraps the model in OptimizedModule on CPU
    from torch._dynamo import OptimizedModule

    assert isinstance(model, (GRUExtractor, OptimizedModule))
    assert train_hidden.shape == (expected_train, gru_config.gru.hidden_size)
    assert val_hidden.shape == (expected_val, gru_config.gru.hidden_size)


@pytest.mark.integration
@pytest.mark.models
def test_train_gru_hidden_states_finite(
    gru_config: Config, synthetic_df: pl.DataFrame
) -> None:
    """Hidden states from trained GRU should be finite (no NaN/Inf)."""
    n = len(synthetic_df)
    split = int(n * 0.8)
    train_df = synthetic_df.slice(0, split)
    val_df = synthetic_df.slice(split)

    _, _classifier, train_hidden, val_hidden, history, _mean, _std, _gru_cols = (
        train_gru(gru_config, train_df, val_df)
    )

    assert np.isfinite(train_hidden).all()
    assert np.isfinite(val_hidden).all()


@pytest.mark.integration
@pytest.mark.models
def test_train_gru_seed_diversity(
    gru_config: Config, synthetic_df: pl.DataFrame
) -> None:
    """Different window_index values should produce different initial weights.

    When ``window_index`` differs, the effective seed (``seed + window_index``)
    changes, so each walk-forward window starts from a unique weight
    initialisation.  After training on identical data, the resulting hidden
    states should diverge.
    """
    n = len(synthetic_df)
    split = int(n * 0.8)
    train_df = synthetic_df.slice(0, split)
    val_df = synthetic_df.slice(split)

    model0, _, _, _, _, _, _, _ = train_gru(
        gru_config, train_df, val_df, window_index=0
    )
    model1, _, _, _, _, _, _, _ = train_gru(
        gru_config, train_df, val_df, window_index=1
    )

    # At least one parameter should differ between the two models.
    params0 = dict(model0.named_parameters())
    params1 = dict(model1.named_parameters())
    any_differ = False
    for name in params0:
        if not torch.equal(params0[name], params1[name]):
            any_differ = True
            break
    assert any_differ, (
        "Expected different parameters for window_index=0 vs window_index=1, "
        "but all parameters are identical — seed diversity is not working"
    )


# ---------------------------------------------------------------------------
# Regression objective tests (task 1)
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_config() -> Config:
    """Minimal config for GRU regression tests."""
    config = Config()
    config.gru.objective = "regression"
    config.gru.input_size = 4
    config.gru.hidden_size = 16
    config.gru.num_layers = 1
    config.gru.sequence_length = 5
    config.gru.dropout = 0.0
    config.gru.learning_rate = 0.01
    config.gru.batch_size = 8
    config.gru.epochs = 3
    config.gru.patience = 2
    config.workflow.random_seed = 42
    config.labels.num_classes = 3
    config.paths.model = "models/lightgbm_model.pkl"
    config.paths.session_dir = ""
    config.model.objective = "regression"
    return config


@pytest.fixture
def regression_df() -> pl.DataFrame:
    """Create a synthetic DataFrame with GRU input columns + regression target."""
    from datetime import datetime, timedelta

    n = 100
    rng = np.random.RandomState(42)
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": 2000.0 + rng.randn(n).cumsum(),
            "log_returns": rng.randn(n) * 0.001,
            "rsi_14": rng.uniform(20, 80, n),
            "atr_14": rng.uniform(5, 30, n),
            "macd_hist": rng.randn(n) * 0.5,
            "regression_target": rng.randn(n) * 0.01,
        }
    )


@pytest.mark.integration
@pytest.mark.models
def test_train_gru_regression_objective(
    regression_config: Config, regression_df: pl.DataFrame
) -> None:
    """train_gru with regression objective should return correct shapes and use MSE.

    Regression mode:
        - label_col becomes "regression_target"
        - labels are float32 (not remapped to {0,1,2})
        - criterion is MSELoss (not FocalLoss)
        - metric_label is "mae" (not "acc")
    """
    n = len(regression_df)
    split = int(n * 0.8)
    train_df = regression_df.slice(0, split)
    val_df = regression_df.slice(split)

    model, _classifier, train_hidden, val_hidden, history, mean, std, _gru_cols = (
        train_gru(regression_config, train_df, val_df)
    )

    seq_len = regression_config.gru.sequence_length
    expected_train = len(train_df) - seq_len + 1
    expected_val = len(val_df) - seq_len + 1

    # Shapes should match
    assert train_hidden.shape == (expected_train, regression_config.gru.hidden_size)
    assert val_hidden.shape == (expected_val, regression_config.gru.hidden_size)

    # Hidden states should be finite
    assert np.isfinite(train_hidden).all()
    assert np.isfinite(val_hidden).all()

    # History should contain "mae" metrics, not "acc"
    assert len(history) > 0
    first_entry = history[0]
    assert "train_mae" in first_entry, (
        f"Regression history should have 'train_mae', got keys: {list(first_entry.keys())}"
    )
    assert "val_mae" in first_entry
    assert "train_acc" not in first_entry, (
        "Regression history should NOT have 'train_acc' (that is classification-only)"
    )


@pytest.mark.integration
@pytest.mark.models
def test_gru_classifier_regression_output_size(
    regression_config: Config,
) -> None:
    """GRU classifier outputs 1 value for regression (vs 3 for multiclass)."""
    from thesis.stage_4_training.gru.training import _build_model_and_classifier

    model, classifier = _build_model_and_classifier(regression_config, input_size=4)

    # Regression classifier should output 1 value
    assert classifier.out_features == 1, (
        f"Regression classifier should have 1 output, got {classifier.out_features}"
    )
