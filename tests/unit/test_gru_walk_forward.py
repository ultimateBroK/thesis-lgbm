"""Tests for GRU walk-forward artifact persistence."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.gru import _save_gru_artifacts


@pytest.mark.unit
def test_save_gru_artifacts_writes_prediction_outputs(tmp_path) -> None:
    """GRU artifact writer should persist prediction/report files."""
    config = Config()
    config.paths.session_dir = str(tmp_path)
    config.paths.predictions = str(
        tmp_path / "predictions" / "final_predictions.parquet"
    )

    oof = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 1) + pl.duration(hours=2),
                interval="1h",
                eager=True,
            ),
            "true_label": [-1, 0, 1],
            "pred_label": [-1, 1, 1],
            "pred_proba_class_minus1": [0.8, 0.1, 0.1],
            "pred_proba_class_0": [0.1, 0.2, 0.2],
            "pred_proba_class_1": [0.1, 0.7, 0.7],
        }
    )
    windows = [
        SimpleNamespace(
            train_start_idx=0,
            train_end_idx=10,
            test_start_idx=10,
            test_end_idx=13,
        )
    ]
    diagnostics = [{"window": 1, "accuracy": 2 / 3}]

    with patch(
        "thesis.stage_4_training.walk_forward.gru.save_gru_model"
    ) as mock_save:
        _save_gru_artifacts(
            config,
            [oof],
            last_model=object(),
            last_classifier=object(),
            last_mean=None,
            last_std=None,
            last_history=[{"epoch": 1, "val_accuracy": 0.5}],
            last_window_accuracy=2 / 3,
            last_window_index=1,
            windows=windows,
            window_diagnostics=diagnostics,
            stage_start=0.0,
        )

    assert (tmp_path / "predictions" / "final_predictions.parquet").exists()
    assert (tmp_path / "predictions" / "prediction_manifest.json").exists()
    assert (tmp_path / "models" / "training_history.json").exists()
    assert (tmp_path / "reports" / "walk_forward_history.json").exists()
    mock_save.assert_called_once()
