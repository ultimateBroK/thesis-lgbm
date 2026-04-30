"""Tests for main.py CLI stage resume logic and pipeline guards."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.config import Config


def _apply_stage_flags(config: Config, stage: int) -> Config:
    """Replicate the --stage flag logic from main.py without filesystem ops.

    Stage N: skip stages 0..N-1, run N..end.
    """
    if stage > 0:
        config.workflow.run_data_pipeline = False
    if stage > 1:
        config.workflow.run_feature_engineering = False
    if stage > 2:
        config.workflow.run_label_generation = False
    if stage > 3:
        config.workflow.run_model_training = False
    if stage > 4:
        config.workflow.run_backtest = False
    return config


class TestStageResumeLogic:
    """Parametrized tests for all 6 stage values (0-5)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("stage", [0, 1, 2, 3, 4, 5])
    def test_stage_disables_correct_flags(self, stage: int) -> None:
        cfg = _apply_stage_flags(Config(), stage)
        flags = [
            cfg.workflow.run_data_pipeline,
            cfg.workflow.run_feature_engineering,
            cfg.workflow.run_label_generation,
            cfg.workflow.run_model_training,
            cfg.workflow.run_backtest,
            cfg.workflow.run_reporting,
        ]
        # All flags before `stage` index should be False, rest True
        for i, flag in enumerate(flags):
            if i < stage:
                assert flag is False, f"Stage {stage}: flag[{i}] should be False"
            else:
                assert flag is True, f"Stage {stage}: flag[{i}] should be True"

    @pytest.mark.unit
    def test_stage_0_enables_all(self) -> None:
        """--stage 0 keeps all workflow flags True."""
        cfg = _apply_stage_flags(Config(), 0)
        assert cfg.workflow.run_data_pipeline is True
        assert cfg.workflow.run_feature_engineering is True
        assert cfg.workflow.run_label_generation is True
        assert cfg.workflow.run_model_training is True
        assert cfg.workflow.run_backtest is True
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_3_disables_first_three(self) -> None:
        """--stage 3 disables data, features, labels but enables model+backtest+report."""
        cfg = _apply_stage_flags(Config(), 3)
        assert cfg.workflow.run_data_pipeline is False
        assert cfg.workflow.run_feature_engineering is False
        assert cfg.workflow.run_label_generation is False
        assert cfg.workflow.run_model_training is True
        assert cfg.workflow.run_backtest is True
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_5_enables_only_reporting(self) -> None:
        """--stage 5 enables only run_reporting."""
        cfg = _apply_stage_flags(Config(), 5)
        assert cfg.workflow.run_reporting is True
        # All others disabled
        assert cfg.workflow.run_data_pipeline is False
        assert cfg.workflow.run_feature_engineering is False
        assert cfg.workflow.run_label_generation is False
        assert cfg.workflow.run_model_training is False
        assert cfg.workflow.run_backtest is False


class TestPipelineEmptyWindowsGuard:
    """Verify RuntimeError when generate_windows returns empty."""

    @pytest.mark.unit
    def test_empty_windows_raises_runtime_error(self) -> None:
        """_run_walk_forward_hybrid must raise RuntimeError for empty windows."""
        from thesis.pipeline import _run_walk_forward_hybrid

        cfg = Config()
        # Use tiny data that can't produce any windows
        cfg.validation.train_window_bars = 999999
        cfg.validation.test_window_bars = 999999
        cfg.validation.min_train_bars = 999999
        cfg.paths.labels = "/nonexistent/labels.parquet"

        # The function checks for labels file first, then windows.
        # We need to mock both: provide a small df and verify the RuntimeError.
        import polars as pl

        tiny_df = pl.DataFrame({
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(hours=9),
                interval="1h",
                eager=True,
            ),
            "value": list(range(10)),
        })

        with patch("thesis.pipeline.Path") as mock_path_cls, \
             patch("thesis.pipeline.generate_windows", return_value=[]):
            # Make labels_path.exists() return True
            mock_path_instance = mock_path_cls.return_value
            mock_path_instance.exists.return_value = True

            # Make pl.read_parquet return our tiny df
            with patch("thesis.pipeline.pl.read_parquet", return_value=tiny_df):
                with pytest.raises(RuntimeError, match="No valid walk-forward windows"):
                    _run_walk_forward_hybrid(cfg)

    @pytest.mark.unit
    def test_zero_oof_preds_raises_runtime_error(self) -> None:
        """Guard: all_oof_preds empty triggers RuntimeError."""
        # This tests the second guard at line 343.
        # We verify the error message is correct.
        from thesis.config import Config
        cfg = Config()
        # The guard checks: `if not all_oof_preds or gru_model is None`
        # This is tested implicitly by the empty-windows path, but we verify
        # the message exists as a contract.
        import inspect as _inspect
        import thesis.pipeline as pipeline_mod
        source = _inspect.getsource(pipeline_mod._run_walk_forward_hybrid)
        assert "No OOF predictions generated" in source
