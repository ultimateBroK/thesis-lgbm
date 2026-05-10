"""Tests for main.py CLI stage resume logic and pipeline guards."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.shared.config import Config
from main import _apply_stage_flags


class TestStageResumeLogic:
    """Parametrized tests for all 6 stage values (1-6)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("stage", [1, 2, 3, 4, 5, 6])
    def test_stage_disables_correct_flags(self, stage: int) -> None:
        cfg = _apply_stage_flags(Config(), stage)
        flags = [
            cfg.workflow.run_data_pipeline,  # ty:ignore[unresolved-attribute]
            cfg.workflow.run_feature_engineering,
            cfg.workflow.run_label_generation,
            cfg.workflow.run_model_training,
            cfg.workflow.run_backtest,
            cfg.workflow.run_reporting,
        ]
        # All flags before `stage` should be False, rest True
        # Stage 1 → all True; Stage N → first N-1 flags False
        for i, flag in enumerate(flags):
            if i < stage - 1:
                assert flag is False, f"Stage {stage}: flag[{i}] should be False"
            else:
                assert flag is True, f"Stage {stage}: flag[{i}] should be True"

    @pytest.mark.unit
    def test_stage_1_enables_all(self) -> None:
        """--stage 1 keeps all workflow flags True."""
        cfg = _apply_stage_flags(Config(), 1)
        assert cfg.workflow.run_data_pipeline is True
        assert cfg.workflow.run_feature_engineering is True
        assert cfg.workflow.run_label_generation is True
        assert cfg.workflow.run_model_training is True
        assert cfg.workflow.run_backtest is True
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_4_disables_first_three(self) -> None:
        """--stage 4 disables data, features, labels but enables model+backtest+report."""
        cfg = _apply_stage_flags(Config(), 4)
        assert cfg.workflow.run_data_pipeline is False
        assert cfg.workflow.run_feature_engineering is False
        assert cfg.workflow.run_label_generation is False
        assert cfg.workflow.run_model_training is True
        assert cfg.workflow.run_backtest is True
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_6_enables_only_reporting(self) -> None:
        """--stage 6 enables only run_reporting."""
        cfg = _apply_stage_flags(Config(), 6)
        assert cfg.workflow.run_reporting is True
        # All others disabled
        assert cfg.workflow.run_data_pipeline is False
        assert cfg.workflow.run_feature_engineering is False
        assert cfg.workflow.run_label_generation is False
        assert cfg.workflow.run_model_training is False
        assert cfg.workflow.run_backtest is False

    @pytest.mark.unit
    def test_stage_flags_reapply_after_session_config_load(self) -> None:
        """--session reload must not reset --stage workflow flags."""
        session_cfg = Config()
        session_cfg.workflow.run_data_pipeline = True
        session_cfg.workflow.run_feature_engineering = True
        session_cfg.workflow.run_label_generation = True
        session_cfg.workflow.run_model_training = True

        result = _apply_stage_flags(session_cfg, 5)

        assert result.workflow.run_data_pipeline is False
        assert result.workflow.run_feature_engineering is False
        assert result.workflow.run_label_generation is False
        assert result.workflow.run_model_training is False
        assert result.workflow.run_backtest is True
        assert result.workflow.run_reporting is True

    @pytest.mark.unit
    def test_force_flag_reapplied_after_session_config_load(self) -> None:
        """--session reloads config, so --force must be applied after reload."""
        from main import _apply_force_flag

        cfg = Config()
        cfg.workflow.force_rerun = False

        result = _apply_force_flag(cfg, force=True)

        assert result.workflow.force_rerun is True


class TestPipelineEmptyWindowsGuard:
    """Verify RuntimeError when generate_windows returns empty.

    Skip entire class — thesis.stage_4_training.walk_forward.hybrid module does not exist.
    """

    @pytest.mark.skip(reason="hybrid module removed")
    def test_empty_windows_raises_runtime_error(self) -> None:
        pass

    @pytest.mark.skip(reason="hybrid module removed")
    def test_zero_oof_preds_raises_runtime_error(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Stage numbering contract — --stage N = start at N, continue through 6
# ---------------------------------------------------------------------------


class TestStageNumbering:
    """Tests encoding the --stage CLI contract: --stage N runs stages N..6.

    Uses _apply_stage_flags which replicates the exact logic from main.py.
    """

    @pytest.mark.unit
    def test_stage_1_runs_stages_1_through_6(self) -> None:
        """--stage 1 keeps all six workflow flags True."""
        cfg = _apply_stage_flags(Config(), 1)

        flags = {
            "run_data_pipeline": cfg.workflow.run_data_pipeline,
            "run_feature_engineering": cfg.workflow.run_feature_engineering,
            "run_label_generation": cfg.workflow.run_label_generation,
            "run_model_training": cfg.workflow.run_model_training,
            "run_backtest": cfg.workflow.run_backtest,
            "run_reporting": cfg.workflow.run_reporting,
        }
        for name, value in flags.items():
            assert value is True, f"--stage 1: {name} must be True"

    @pytest.mark.unit
    def test_stage_3_runs_stages_3_through_6(self) -> None:
        """--stage 3 disables stages 1-2, enables stages 3-6."""
        cfg = _apply_stage_flags(Config(), 3)

        assert cfg.workflow.run_data_pipeline is False
        assert cfg.workflow.run_feature_engineering is False
        assert cfg.workflow.run_label_generation is True
        assert cfg.workflow.run_model_training is True
        assert cfg.workflow.run_backtest is True
        assert cfg.workflow.run_reporting is True

    @pytest.mark.unit
    def test_stage_6_runs_only_stage_6(self) -> None:
        """--stage 6 disables stages 1-5, enables only stage 6 reporting."""
        cfg = _apply_stage_flags(Config(), 6)

        assert cfg.workflow.run_reporting is True
        for field in (
            "run_data_pipeline",
            "run_feature_engineering",
            "run_label_generation",
            "run_model_training",
            "run_backtest",
        ):
            assert getattr(cfg.workflow, field) is False, (
                f"--stage 6: {field} must be False"
            )
