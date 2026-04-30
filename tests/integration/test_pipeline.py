"""Integration tests for pipeline module.

Tests pipeline stage ordering, caching, and --force flag.
These tests use a temporary directory and do NOT write to the project's results/ directory.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.config import Config
from thesis.pipeline import run_pipeline
from thesis.features import generate_features
from thesis.labels import generate_labels


def create_synthetic_ohlcv(
    n_rows: int = 500, start_date: str = "2020-01-01"
) -> pl.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    base_price = 1800.0

    timestamps = pl.datetime_range(
        start=pl.datetime(2020, 1, 1, 0),
        end=pl.datetime(2020, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    returns = np.random.normal(0, 0.001, n_rows)
    closes = base_price * np.exp(np.cumsum(returns))
    opens = closes * (1 + np.random.normal(0, 0.0005, n_rows))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, n_rows)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, n_rows)))
    volumes = np.random.randint(1000, 10000, n_rows).astype(float)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture
def temp_pipeline_dir():
    """Create a temporary directory for pipeline testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pipeline_config(temp_pipeline_dir: Path) -> Config:
    """Create a config for pipeline testing."""
    config = Config()

    # Set paths to temp directory
    config.paths.data_raw = str(temp_pipeline_dir / "data" / "raw" / "XAUUSD")
    config.paths.data_processed = str(temp_pipeline_dir / "data" / "processed")
    config.paths.ohlcv = str(temp_pipeline_dir / "data" / "processed" / "ohlcv.parquet")
    config.paths.features = str(
        temp_pipeline_dir / "data" / "processed" / "features.parquet"
    )
    config.paths.labels = str(
        temp_pipeline_dir / "data" / "processed" / "labels.parquet"
    )
    config.paths.train_data = str(
        temp_pipeline_dir / "data" / "processed" / "train.parquet"
    )
    config.paths.val_data = str(
        temp_pipeline_dir / "data" / "processed" / "val.parquet"
    )
    config.paths.test_data = str(
        temp_pipeline_dir / "data" / "processed" / "test.parquet"
    )
    config.paths.model = str(temp_pipeline_dir / "models" / "lightgbm_model.pkl")
    config.paths.gru_model = str(temp_pipeline_dir / "models" / "gru_model.pt")
    config.paths.stack_bundle = str(
        temp_pipeline_dir / "models" / "stacking_bundle.joblib"
    )
    config.paths.predictions = str(
        temp_pipeline_dir / "data" / "predictions" / "final_predictions.parquet"
    )
    config.paths.backtest_results = str(
        temp_pipeline_dir / "results" / "backtest_results.json"
    )
    config.paths.report = str(temp_pipeline_dir / "results" / "thesis_report.md")

    # Point session_dir into temp directory so all report outputs go there
    config.paths.session_dir = str(temp_pipeline_dir / "session")

    # Adjust date ranges for synthetic data (500 hours = ~21 days starting 2020-01-01)
    # Split: 60% train, 20% val, 20% test
    config.splitting.train_start = "2020-01-01"
    config.splitting.train_end = "2020-01-13 23:59:59"  # ~300 hours
    config.splitting.val_start = "2020-01-14"
    config.splitting.val_end = "2020-01-18 23:59:59"  # ~100 hours
    config.splitting.test_start = "2020-01-19"
    config.splitting.test_end = "2020-01-31 23:59:59"  # ~100 hours
    config.splitting.purge_bars = 5  # Small purge for testing
    config.splitting.embargo_bars = 2

    # Use smaller model for speed
    config.model.n_estimators = 5
    config.model.num_leaves = 4
    config.model.max_depth = 3
    config.model.use_optuna = False

    # Enable all stages
    config.workflow.run_feature_engineering = True
    config.workflow.run_label_generation = True
    config.workflow.run_data_splitting = True
    config.workflow.run_model_training = True
    config.workflow.run_backtest = True
    config.workflow.run_reporting = True
    config.workflow.force_rerun = False

    return config


def setup_ohlcv_data(config: Config, n_rows: int = 500) -> None:
    """Set up synthetic OHLCV data for testing."""
    ohlcv_path = Path(config.paths.ohlcv)
    ohlcv_path.parent.mkdir(parents=True, exist_ok=True)

    df = create_synthetic_ohlcv(n_rows=n_rows)
    df.write_parquet(ohlcv_path)


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_stage_ordering(pipeline_config: Config) -> None:
    """Test pipeline stage ordering (features needs OHLCV, labels needs features, etc.)."""
    # Create OHLCV data
    setup_ohlcv_data(pipeline_config, n_rows=500)

    # Run features stage
    pipeline_config.workflow.run_label_generation = False
    pipeline_config.workflow.run_data_splitting = False
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    run_pipeline(pipeline_config)

    # Features should exist
    assert Path(pipeline_config.paths.features).exists(), "Features should be created"

    # Now run labels (requires features)
    pipeline_config.workflow.run_feature_engineering = False
    pipeline_config.workflow.run_label_generation = True

    run_pipeline(pipeline_config)

    # Labels should exist
    assert Path(pipeline_config.paths.labels).exists(), "Labels should be created"


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_respects_cache(pipeline_config: Config) -> None:
    """Test that pipeline respects cache (skip existing outputs)."""
    # Create OHLCV data
    setup_ohlcv_data(pipeline_config, n_rows=500)

    # Run features once
    pipeline_config.workflow.run_label_generation = False
    pipeline_config.workflow.run_data_splitting = False
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    run_pipeline(pipeline_config)

    features_path = Path(pipeline_config.paths.features)
    assert features_path.exists()

    # Get modification time
    first_mtime = features_path.stat().st_mtime

    # Run again without force - should skip
    run_pipeline(pipeline_config)

    # Modification time should be unchanged
    second_mtime = features_path.stat().st_mtime
    assert second_mtime == first_mtime, (
        "Features should not be regenerated without force"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_force_overwrites(pipeline_config: Config) -> None:
    """Test --force flag overwrites existing outputs."""
    # Create OHLCV data
    setup_ohlcv_data(pipeline_config, n_rows=500)

    # Run features once
    pipeline_config.workflow.run_label_generation = False
    pipeline_config.workflow.run_data_splitting = False
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    run_pipeline(pipeline_config)

    features_path = Path(pipeline_config.paths.features)
    assert features_path.exists()

    # Get modification time
    first_mtime = features_path.stat().st_mtime

    # Run again with force
    pipeline_config.workflow.force_rerun = True
    run_pipeline(pipeline_config)

    # Modification time should change
    second_mtime = features_path.stat().st_mtime
    assert second_mtime > first_mtime, "Features should be regenerated with force"


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_disabled_stages_skipped(pipeline_config: Config) -> None:
    """Test that disabled stages are skipped."""
    # Create OHLCV data
    setup_ohlcv_data(pipeline_config, n_rows=500)

    # Disable all stages except features
    pipeline_config.workflow.run_feature_engineering = True
    pipeline_config.workflow.run_label_generation = False
    pipeline_config.workflow.run_data_splitting = False
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    run_pipeline(pipeline_config)

    # Only features should exist
    assert Path(pipeline_config.paths.features).exists()
    assert not Path(pipeline_config.paths.labels).exists()
    assert not Path(pipeline_config.paths.train_data).exists()


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_labels_produced(pipeline_config: Config) -> None:
    """Test that labels stage produces the labels file."""
    # Create full pipeline through labels
    setup_ohlcv_data(pipeline_config, n_rows=500)

    # Run features and labels first
    generate_features(pipeline_config)
    generate_labels(pipeline_config)

    # Now verify labels exist
    pipeline_config.workflow.run_feature_engineering = False
    pipeline_config.workflow.run_label_generation = False
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    run_pipeline(pipeline_config)

    # Labels should exist
    assert Path(pipeline_config.paths.labels).exists()

    # Check that labels are non-empty
    labels_df = pl.read_parquet(pipeline_config.paths.labels)
    assert len(labels_df) > 0
    assert "label" in labels_df.columns


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_end_to_end_smoke(pipeline_config: Config) -> None:
    """Smoke test: full pipeline runs without errors (through features + labels only).

    Walk-forward training requires more data than synthetic provides,
    so this test validates the data processing pipeline stages (0-2).
    """
    # Create OHLCV data with more rows to cover all date ranges
    n_rows = 500  # ~21 days of hourly data
    setup_ohlcv_data(pipeline_config, n_rows=n_rows)

    # Run only data processing stages (walk-forward needs large dataset)
    pipeline_config.workflow.run_feature_engineering = True
    pipeline_config.workflow.run_label_generation = True
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    # Should not raise any exception
    run_pipeline(pipeline_config)

    # Verify outputs exist
    assert Path(pipeline_config.paths.features).exists()
    assert Path(pipeline_config.paths.labels).exists()


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_stage_dependencies(pipeline_config: Config) -> None:
    """Test that stages fail gracefully when dependencies are missing."""
    # Try to run labels without features
    setup_ohlcv_data(pipeline_config, n_rows=100)

    pipeline_config.workflow.run_feature_engineering = False
    pipeline_config.workflow.run_label_generation = True
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    # Should raise FileNotFoundError because features don't exist
    with pytest.raises(FileNotFoundError):
        run_pipeline(pipeline_config)


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_labels_without_features_fails(pipeline_config: Config) -> None:
    """Test that labels stage fails without features."""
    setup_ohlcv_data(pipeline_config, n_rows=100)

    pipeline_config.workflow.run_feature_engineering = False
    pipeline_config.workflow.run_label_generation = True
    pipeline_config.workflow.run_model_training = False
    pipeline_config.workflow.run_backtest = False
    pipeline_config.workflow.run_reporting = False

    # Should raise FileNotFoundError because features don't exist
    with pytest.raises(FileNotFoundError):
        run_pipeline(pipeline_config)


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_true_stacking_smoke(pipeline_config: Config) -> None:
    """Sliding true stacking should produce deployable and OOF artifacts."""
    setup_ohlcv_data(pipeline_config, n_rows=1200)

    pipeline_config.model.architecture = "stacking"
    pipeline_config.model.use_optuna = False
    pipeline_config.model.n_estimators = 10
    pipeline_config.model.num_leaves = 8
    pipeline_config.model.max_depth = 4

    pipeline_config.gru.sequence_length = 12
    pipeline_config.gru.hidden_size = 8
    pipeline_config.gru.batch_size = 16
    pipeline_config.gru.epochs = 2
    pipeline_config.gru.patience = 1

    pipeline_config.validation.method = "sliding"
    pipeline_config.validation.train_window_bars = 240
    pipeline_config.validation.test_window_bars = 80
    pipeline_config.validation.step_bars = 80
    pipeline_config.validation.purge_bars = 5
    pipeline_config.validation.embargo_bars = 5
    pipeline_config.validation.min_train_bars = 200
    pipeline_config.validation.wf_optuna_trials = 0

    pipeline_config.stacking.min_meta_train_folds = 1
    pipeline_config.stacking.min_meta_train_rows = 60
    pipeline_config.stacking.final_refit = True

    pipeline_config.backtest.confidence_threshold = 0.0
    pipeline_config.backtest.atr_tp_multiplier = 1.0

    run_pipeline(pipeline_config)

    assert Path(pipeline_config.paths.predictions).exists()
    assert Path(pipeline_config.paths.stack_bundle).exists()
    assert Path(pipeline_config.paths.model).exists()
    assert Path(pipeline_config.paths.gru_model).exists()
    assert Path(pipeline_config.paths.backtest_results).exists()

    base_oof_path = (
        Path(pipeline_config.paths.session_dir)
        / "predictions"
        / "base_oof_predictions.parquet"
    )
    assert base_oof_path.exists()

    preds = pl.read_parquet(pipeline_config.paths.predictions)
    assert len(preds) > 0
    assert {
        "timestamp",
        "true_label",
        "pred_label",
        "pred_proba_class_minus1",
        "pred_proba_class_0",
        "pred_proba_class_1",
    }.issubset(preds.columns)
