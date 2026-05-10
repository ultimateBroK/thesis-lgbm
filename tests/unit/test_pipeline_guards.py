"""Tests for pipeline module — OOF guards and validation checks."""

import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.shared.config import Config


@pytest.mark.unit
def test_backtest_barrier_mismatch_raises() -> None:
    """Stage 5 must fail fast when label/backtest ATR barriers diverge."""
    from thesis.pipeline import _run_backtest_with_barrier_guard

    config = Config()
    config.labels.barrier_atr_multiplier = 2.0
    config.backtest.atr_tp_multiplier = 3.0
    config.backtest.atr_stop_multiplier = 1.0

    with pytest.raises(ValueError, match="ATR barrier mismatch"):
        _run_backtest_with_barrier_guard(config)


@pytest.mark.unit
def test_backtest_barrier_match_calls_backtest() -> None:
    """Matching barriers should allow Stage 5 to reach run_backtest."""
    from thesis.pipeline import _run_backtest_with_barrier_guard

    config = Config()
    config.labels.barrier_atr_multiplier = 2.0
    config.backtest.atr_tp_multiplier = 2.0
    config.backtest.atr_stop_multiplier = 2.0

    with patch("thesis.pipeline.run_backtest") as run_bt:
        _run_backtest_with_barrier_guard(config)

    run_bt.assert_called_once_with(config)


# ---------------------------------------------------------------------------
# Stage numbering contract tests
# ---------------------------------------------------------------------------


class TestStageNumbering:
    """Stage numbering contract: 1-indexed stages, correct labels, docstring."""

    @pytest.mark.unit
    def test_stage_header_1_outputs_correct_text(self, caplog) -> None:
        """stage_header(1) logs 'STAGE 1/6' with 'Data Preparation' label."""
        from thesis.shared.ui import stage_header

        with caplog.at_level(logging.INFO, logger="thesis"):
            stage_header(1)

        log_text = " ".join(record.message for record in caplog.records)
        assert "STAGE 1/6" in log_text
        assert "Data Preparation" in log_text

    @pytest.mark.unit
    def test_stage_labels_keys_are_1_to_6(self) -> None:
        """STAGE_LABELS dict keys must be 1–6, not 0–5."""
        from thesis.shared.ui import STAGE_LABELS

        assert sorted(STAGE_LABELS.keys()) == [1, 2, 3, 4, 5, 6]

    @pytest.mark.unit
    def test_stage_skip_outputs_correct_text(self, caplog) -> None:
        """stage_skip logs skipped stage labels for file capture."""
        from thesis.shared.ui import stage_skip

        with caplog.at_level(logging.INFO, logger="thesis"):
            stage_skip(2, "cached")

        log_text = " ".join(record.message for record in caplog.records)
        assert "SKIP Feature Engineering" in log_text
        assert "cached" in log_text

    @pytest.mark.unit
    def test_ui_console_singleton(self) -> None:
        """All UI imports should resolve to the shared Rich Console."""
        import thesis.shared.ui as ui_a
        import thesis.pipeline as pipeline
        from thesis.stage_4_training.lgbm import training as _lgbm
        from thesis.stage_4_training.walk_forward import lgbm as _wf_lgbm

        assert pipeline.console is ui_a.console
        assert _lgbm.console is ui_a.console
        assert _wf_lgbm.console is ui_a.console

    @pytest.mark.unit
    def test_run_pipeline_docstring_stage_numbering(self) -> None:
        """run_pipeline.__doc__ must document Stage 1 (Data Preparation)."""
        from thesis.pipeline import run_pipeline

        assert run_pipeline.__doc__ is not None
        assert "1. Data preparation" in run_pipeline.__doc__

    @pytest.mark.unit
    @pytest.mark.parametrize("stage", [0, 7])
    def test_stage_header_out_of_range_is_graceful(self, stage: int) -> None:
        """stage_header(0) and stage_header(7) must not raise exceptions."""
        from thesis.shared.ui import stage_header

        # Should complete without raising — uses .get() fallback
        stage_header(stage)
