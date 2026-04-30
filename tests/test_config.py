"""Tests for config module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from thesis.config import load_config, get_config, reload_config, Config, LGBMConfig, WorkflowConfig
from thesis.constants import CORE_STATIC_FEATURES


def test_load_config_default():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert isinstance(cfg, Config)
    assert cfg.data.symbol == "XAUUSD"
    assert cfg.data.timeframe == "1H"


def test_get_config_reuses_cached_instance():
    config_path = Path(__file__).parent.parent / "config.toml"
    get_config.cache_clear()
    cfg1 = get_config(config_path)
    cfg2 = get_config(config_path)
    assert cfg1 is cfg2


def test_reload_config_refreshes_cached_instance():
    config_path = Path(__file__).parent.parent / "config.toml"
    cfg1 = get_config(config_path)
    cfg2 = reload_config(config_path)
    assert cfg1 is not cfg2
    assert cfg2.data.symbol == "XAUUSD"


def test_config_sections_exist():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "splitting")
    assert hasattr(cfg, "features")
    assert hasattr(cfg, "labels")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "stacking")
    assert hasattr(cfg, "backtest")
    assert hasattr(cfg, "workflow")
    assert hasattr(cfg, "paths")


def test_model_config_flat():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    # Model config is a flat dataclass, not a dict
    assert isinstance(cfg.model, LGBMConfig)
    assert cfg.model.architecture in {"hybrid", "stacking"}
    assert cfg.model.num_leaves > 0
    assert cfg.model.learning_rate > 0


def test_labels_no_session_atr():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    # No session_atr attribute
    assert not hasattr(cfg.labels, "session_atr")
    assert cfg.labels.atr_multiplier > 0


def test_paths_basic():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert cfg.paths.train_data.endswith(".parquet")
    assert cfg.paths.val_data.endswith(".parquet")
    assert cfg.paths.test_data.endswith(".parquet")
    assert cfg.paths.stack_bundle.endswith(".joblib")


def test_missing_config_raises():
    from thesis.config import load_config
    import pytest

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.toml")


def test_embargo_scales_for_daily_timeframe(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[data]
timeframe = "1D"

[splitting]
embargo_bars = 50
embargo_scale_by_timeframe = true
embargo_reference_timeframe = "1H"
""".strip()
    )

    cfg = load_config(cfg_file)
    assert cfg.splitting.embargo_bars == 2


def test_embargo_keeps_configured_bars_when_scaling_disabled(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[data]
timeframe = "1D"

[splitting]
embargo_bars = 50
embargo_scale_by_timeframe = false
embargo_reference_timeframe = "1H"
""".strip()
    )

    cfg = load_config(cfg_file)
    assert cfg.splitting.embargo_bars == 50


def test_data_config_has_download_max_retries():
    """Test that DataConfig has download_max_retries attribute."""
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert hasattr(cfg.data, "download_max_retries")
    assert cfg.data.download_max_retries > 0


def test_download_max_retries_default_value():
    """Test download_max_retries default value from config.toml."""
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert cfg.data.download_max_retries == 7


def test_stacking_defaults():
    cfg = load_config(Path(__file__).parent.parent / "config.toml")
    assert cfg.stacking.base_models == ["gru", "lgbm"]
    assert cfg.stacking.meta_model == "lightgbm"
    assert cfg.stacking.use_probability_features_only is True
    assert cfg.stacking.final_refit is True


# ---------------------------------------------------------------------------
# WorkflowConfig structure tests (Task 13d — run_data_splitting removed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_workflow_config_no_run_data_splitting():
    """run_data_splitting was removed in cleanup — must not exist."""
    wf = WorkflowConfig()
    assert not hasattr(wf, "run_data_splitting")


@pytest.mark.unit
def test_workflow_config_has_six_boolean_toggles():
    """WorkflowConfig should have exactly 6 run_* boolean toggles."""
    bool_fields = [
        attr
        for attr in dir(WorkflowConfig)
        if attr.startswith("run_") and isinstance(getattr(WorkflowConfig, attr, None), bool)
    ]
    assert len(bool_fields) == 6


@pytest.mark.unit
def test_workflow_config_expected_boolean_fields():
    """All expected boolean fields present after cleanup."""
    wf = WorkflowConfig()
    expected = [
        "run_data_pipeline",
        "run_feature_engineering",
        "run_label_generation",
        "run_model_training",
        "run_backtest",
        "run_reporting",
    ]
    for field_name in expected:
        assert hasattr(wf, field_name), f"Missing: {field_name}"
        assert getattr(wf, field_name) is True, f"{field_name} should default True"


# ---------------------------------------------------------------------------
# Docs/config consistency tests (Task 13b)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_static_feature_count_is_21():
    """CORE_STATIC_FEATURES must have exactly 21 features."""
    assert len(CORE_STATIC_FEATURES) == 21


@pytest.mark.unit
def test_gru_hidden_size_is_64():
    """GRU hidden_size must equal 64 (docs contract)."""
    cfg = Config()
    assert cfg.gru.hidden_size == 64


@pytest.mark.unit
def test_total_features_equals_hidden_plus_static():
    """total features = hidden_size + len(static_feature_cols)."""
    cfg = Config()
    total = cfg.gru.hidden_size + len(cfg.features.static_feature_cols)
    assert total == 64 + 21  # 85


@pytest.mark.unit
def test_static_features_match_constants():
    """Config defaults must match CORE_STATIC_FEATURES tuple."""
    cfg = Config()
    assert list(cfg.features.static_feature_cols) == list(CORE_STATIC_FEATURES)
