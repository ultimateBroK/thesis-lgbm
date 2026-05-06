"""Hybrid walk-forward artifact persistence helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import polars as pl

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.utils import (
    _add_confidence_columns,
    _validate_predictions,
    _write_prediction_manifest,
)

logger = logging.getLogger("thesis.pipeline")


def _build_lgbm_info(
    last_lgbm_model: Any,
    last_feature_cols: list[str],
    last_window_accuracy: float | None,
    window_index: int | None = None,
    total_windows: int = 0,
    window_train_dates: dict[str, str] | None = None,
    window_test_dates: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build LightGBM metadata dict for training history JSON.

    Args:
        last_lgbm_model: Trained LightGBM model from the last window.
        last_feature_cols: Feature column names used by the model.
        last_window_accuracy: OOF accuracy of the last window, or ``None``.
        window_index: 1-based index of the window that produced the model.
        total_windows: Total number of walk-forward windows.
        window_train_dates: ``{"start": ..., "end": ...}`` date range for
            the last window's training data.
        window_test_dates: ``{"start": ..., "end": ...}`` date range for
            the last window's test data.

    Returns:
        Dictionary with validation protocol, window provenance, best
        iteration, feature count, and class count.
    """
    info: dict[str, Any] = {
        "artifact_strategy": "last_walk_forward_window",
        "validation_protocol": {
            "outer_windows": "bar_based_walk_forward_with_purge_embargo",
            "gru_validation": "tail_20_percent_of_outer_train",
            "lgbm_validation": "tail_20_percent_of_sequence_aligned_outer_train",
        },
        "last_window_accuracy": last_window_accuracy,
        "best_iteration": int(last_lgbm_model.best_iteration_)
        if hasattr(last_lgbm_model, "best_iteration_")
        else None,
        "n_features": len(last_feature_cols),
        "n_classes": len(last_lgbm_model.classes_)
        if hasattr(last_lgbm_model, "classes_")
        else None,
    }
    if window_index is not None:
        info["window_index"] = window_index
        info["total_windows"] = total_windows
        info["window_train_date_range"] = window_train_dates or {}
        info["window_test_date_range"] = window_test_dates or {}
        info["window_oof_accuracy"] = last_window_accuracy
    return info


def _build_wf_history(
    windows: list,
    window_diagnostics: list[dict[str, Any]],
    oof_len: int,
) -> dict[str, Any]:
    """Build walk-forward history dict with per-window details.

    Args:
        windows: List of walk-forward window objects with index attrs.
        window_diagnostics: Per-window diagnostic dictionaries.
        oof_len: Total number of OOF predictions.

    Returns:
        Dictionary with ``num_windows``, ``total_oof_predictions``,
        and ``window_details`` list.
    """
    return {
        "num_windows": len(windows),
        "total_oof_predictions": oof_len,
        "window_details": [
            {
                "window": i + 1,
                "train_start_idx": w.train_start_idx,
                "train_end_idx": w.train_end_idx,
                "test_start_idx": w.test_start_idx,
                "test_end_idx": w.test_end_idx,
                **next(
                    (item for item in window_diagnostics if item["window"] == i + 1),
                    {},
                ),
            }
            for i, w in enumerate(windows)
        ],
    }


def _save_wf_artifacts(
    config: Config,
    all_oof_preds: list[pl.DataFrame],
    gru_model: Any,
    gru_mean: Any,
    gru_std: Any,
    last_lgbm_model: Any,
    last_feature_cols: list[str],
    last_window_accuracy: float | None,
    last_window_index: int,
    last_gru_history: list[dict],
    windows: list,
    window_diagnostics: list[dict[str, Any]],
    stage_start: float,
    is_regression: bool,
) -> None:
    """Validate OOF predictions and persist all walk-forward artifacts.

    Args:
        config: Application configuration.
        all_oof_preds: List of per-window OOF Polars DataFrames.
        gru_model: GRU model from the last window.
        gru_mean: Normalization mean for the GRU model.
        gru_std: Normalization std for the GRU model.
        last_lgbm_model: LightGBM model from the last window.
        last_feature_cols: Feature column names.
        last_window_accuracy: Accuracy of the last window.
        last_window_index: 1-based index of the last window.
        last_gru_history: GRU training history list.
        windows: Walk-forward window objects.
        window_diagnostics: Per-window diagnostic dictionaries.
        stage_start: ``time.perf_counter()`` start timestamp.
        is_regression: Whether the objective is regression.

    Raises:
        RuntimeError: If no predictions were generated.
        ValueError: If duplicate timestamps are found in OOF data.
    """
    import joblib

    from thesis.stage_4_training.gru import save_gru_model
    from thesis.stage_4_training.lgbm.utils import _save_feature_importance

    # ── Guard ──
    if not all_oof_preds or gru_model is None:
        raise RuntimeError(
            "No OOF predictions generated — all walk-forward windows were skipped"
        )

    # ── Save GRU model (last window) ──
    if config.paths.session_dir:
        gru_path = Path(config.paths.session_dir) / "models" / "gru_model.pt"
        save_gru_model(gru_model, config, gru_path, mean=gru_mean, std=gru_std)

    # ── Concatenate & validate OOF ──
    oof_df = pl.concat(all_oof_preds)
    oof_df = _add_confidence_columns(oof_df)

    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_predictions(oof_df, preds_path)
    oof_df.write_parquet(preds_path)
    oof_df.write_csv(preds_path.with_suffix(".csv"))
    _write_prediction_manifest(
        oof_df,
        preds_path,
        windows_count=len(window_diagnostics),
    )

    # ── Save LGBM model + feature importance ──
    if last_lgbm_model is not None:
        model_path = Path(config.paths.model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(last_lgbm_model, model_path)
    if last_lgbm_model is not None and last_feature_cols:
        _save_feature_importance(last_lgbm_model, last_feature_cols, config)

    # ── Save training history ──
    if config.paths.session_dir:
        models_dir = Path(config.paths.session_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        history_path = models_dir / "training_history.json"

        # Build per-window accuracy map from diagnostics
        per_window_accuracies: dict[str, float | None] = {}
        for d in window_diagnostics:
            key = str(d.get("window", ""))
            if key:
                per_window_accuracies[key] = d.get("accuracy")

        # Extract last-window date ranges from diagnostics
        last_train_dates: dict[str, str] = {}
        last_test_dates: dict[str, str] = {}
        last_win_key = str(last_window_index)
        for d in window_diagnostics:
            if str(d.get("window", "")) == last_win_key:
                last_train_dates = d.get("train_dates", {})
                last_test_dates = d.get("test_dates", {})
                break

        lgbm_info = (
            _build_lgbm_info(
                last_lgbm_model,
                last_feature_cols,
                last_window_accuracy,
                window_index=last_window_index,
                total_windows=len(windows),
                window_train_dates=last_train_dates,
                window_test_dates=last_test_dates,
            )
            if last_lgbm_model is not None
            else {}
        )

        deployment_note = (
            f"Model saved from window {last_window_index}/{len(windows)} "
            "(the last chronological walk-forward window). "
            "This model has NOT seen any future data beyond its training window."
        )

        with open(history_path, "w") as f:
            json.dump(
                {
                    "gru": last_gru_history,
                    "lightgbm": lgbm_info,
                    "deployment_note": deployment_note,
                    "per_window_accuracies": per_window_accuracies,
                },
                f,
                indent=2,
            )
        logger.info("Training history saved to %s", history_path)

    # ── Save walk-forward history ──
    if config.paths.session_dir:
        wf_path = (
            Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
        )
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(wf_path, "w") as f:
            json.dump(
                _build_wf_history(windows, window_diagnostics, len(oof_df)), f, indent=2
            )

    total_time = time.perf_counter() - stage_start
    logger.info(
        "Walk-forward complete: %d windows, %d OOF predictions (%.1fs)",
        len(windows),
        len(oof_df),
        total_time,
    )
