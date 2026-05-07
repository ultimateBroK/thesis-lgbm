"""LightGBM training entry points (walk-forward + fixed split)."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import EXCLUDE_COLS
from thesis.shared.ui import console
from thesis.stage_4_training.validation import generate_windows, log_windows
from thesis.stage_4_training.walk_forward.artifacts import (
    _log_walk_forward_complete,
    _save_arch_copy,
    _save_oof_predictions,
    _save_training_history,
    _save_walk_forward_history,
)
from thesis.stage_4_training.walk_forward.hybrid import _compute_regression_target
from thesis.stage_4_training.walk_forward.utils import (
    _CLASS_ORDER,
    _add_prediction_diagnostics,
    _align_probability_matrix,
    _one_hot_proba_columns,
    _probability_columns,
    _select_static_feature_cols,
    _window_diagnostics,
    fit_static_feature_pipeline,
)

logger = logging.getLogger("thesis.pipeline")

# --- Minimum Sample Thresholds ---
_STATIC_MIN_TRAIN_ROWS = 2  # Minimum training rows for static walk-forward

# --- Validation Split ---
_VALIDATION_SPLIT_FRACTION = 0.2  # Tail validation split for GRU/LGBM/static


def _prepare_static_wf_data(
    config: Config,
) -> tuple[pl.DataFrame, list[Any], list[str], bool]:
    """Load labeled data, pre-compute regression target, and generate windows.

    Args:
        config: Application configuration.

    Returns:
        ``(df, windows, feature_cols, is_regression)`` — the full labeled
        DataFrame, walk-forward window objects, sorted feature column
        names, and a boolean indicating regression objective.
    """
    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    with console.status(f"[cyan]Loading labels[/] {labels_path}"):
        df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data for static baseline: %d rows", len(df))
    df, is_regression_static = _compute_regression_target(df, config)

    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None
    if event_end is None:
        logger.warning(
            "Labels lack event_end column — falling back to fixed-bar purge. "
            "Regenerate labels to enable event-time purging."
        )

    windows = generate_windows(
        total_bars=len(df),
        train_window_bars=config.validation.train_window_bars,
        test_window_bars=config.validation.test_window_bars,
        step_bars=config.validation.step_bars,
        purge_bars=config.validation.purge_bars,
        embargo_bars=config.validation.embargo_bars,
        min_train_bars=config.validation.min_train_bars,
        event_end=event_end,
    )
    if not windows:
        raise RuntimeError("No valid walk-forward windows generated")

    log_windows(windows, df, "timestamp")
    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)
    return df, windows, feature_cols, is_regression_static


def _train_and_predict_static_window(
    config: Config,
    w_idx: int,
    window: Any,
    df: pl.DataFrame,
    feature_cols: list[str],
    is_regression_static: bool,
    expanded_features: bool,
) -> dict[str, Any] | None:
    """Train LightGBM and generate predictions for a single static window.

    Returns a dict with ``oof_chunk``, ``model``, ``static_cols``,
    ``accuracy``, and ``diag``, or ``None`` if the window is too small.
    """
    from thesis.stage_4_training.lgbm.utils import (
        _compute_class_weights,
        _train_fixed,
        _wrap_np,
    )

    train_df = df.slice(
        window.train_start_idx, window.train_end_idx - window.train_start_idx
    )
    test_df = df.slice(
        window.test_start_idx, window.test_end_idx - window.test_start_idx
    )
    if len(train_df) < _STATIC_MIN_TRAIN_ROWS or test_df.is_empty():
        logger.warning("Static window %d too small; skipping", w_idx + 1)
        return None
    if expanded_features:
        static_cols = [
            c
            for c in feature_cols
            if c in train_df.columns
            and not c.startswith("gru_")
            and c != "regression_target"
        ]
        mode_tag = "expanded"
    else:
        static_cols = _select_static_feature_cols(config, train_df, feature_cols)
        mode_tag = "whitelist"
    logger.info(
        "Static baseline using %d features (%s mode)", len(static_cols), mode_tag
    )
    if is_regression_static:
        y_train = train_df["regression_target"].to_numpy().astype(np.float64)
        y_test = test_df["regression_target"].to_numpy().astype(np.float64)
        y_train_cls = train_df["label"].to_numpy().astype(np.int32)
        y_test_cls = test_df["label"].to_numpy().astype(np.int32)
    else:
        y_train = train_df["label"].to_numpy().astype(np.int32)
        y_test = test_df["label"].to_numpy().astype(np.int32)
        y_train_cls, y_test_cls = y_train, y_test
    sw = (
        train_df["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_df.columns
        else None
    )
    diag = _window_diagnostics(w_idx + 1, train_df, test_df, y_train_cls, y_test_cls)
    val_split_idx = max(1, int(len(train_df) * _VALIDATION_SPLIT_FRACTION))
    pipeline_fit_df = train_df.slice(0, len(train_df) - val_split_idx)
    pipeline_fit_y = y_train_cls[:-val_split_idx]
    static_pipeline, selected_static_cols = fit_static_feature_pipeline(
        config,
        pipeline_fit_df,
        static_cols,
        pipeline_fit_y,
    )
    X_train = static_pipeline.transform(train_df.select(static_cols).to_pandas())
    X_test = static_pipeline.transform(test_df.select(static_cols).to_pandas())
    X_tr, y_tr = X_train[:-val_split_idx], y_train[:-val_split_idx]
    X_val, y_val = X_train[-val_split_idx:], y_train[-val_split_idx:]
    w_tr = sw[:-val_split_idx] if sw is not None else None
    class_weights = None if is_regression_static else _compute_class_weights(y_tr)
    diag["class_weights"] = (
        {str(k): v for k, v in class_weights.items()} if class_weights else None
    )
    diag["shift_weights_per_class"] = None  # static baseline: no shift weights
    model = _train_fixed(
        X_tr,
        y_tr,
        X_val,
        y_val,
        class_weights,
        config,
        selected_static_cols,
        sample_weight=w_tr,
    )
    if is_regression_static:
        raw_preds = model.predict(_wrap_np(X_test, selected_static_cols))
        preds = np.sign(raw_preds).astype(np.int32)  # threshold=0
        aligned_proba = np.zeros((len(raw_preds), 3), dtype=np.float64)
        aligned_proba[np.arange(len(preds)), preds + 1] = 1.0
        oof_chunk = pl.DataFrame(
            {
                "timestamp": test_df["timestamp"],
                "true_label": y_test_cls,
                "pred_label": preds,
                "pred_raw": raw_preds.astype(np.float64),
                **_one_hot_proba_columns(preds),
            }
        )
    else:
        proba = model.predict_proba(_wrap_np(X_test, selected_static_cols))
        aligned_proba = _align_probability_matrix(proba, model.classes_)
        preds = _CLASS_ORDER[np.argmax(aligned_proba, axis=1)]
        oof_chunk = pl.DataFrame(
            {
                "timestamp": test_df["timestamp"],
                "true_label": y_test_cls,
                "pred_label": preds.astype(np.int32),
                **_probability_columns(proba, model.classes_),
            }
        )
    _add_prediction_diagnostics(diag, preds, y_test_cls, aligned_proba)
    acc = float((preds == y_test_cls).mean())
    logger.info(
        "Static window %d: accuracy=%.4f, test_samples=%d",
        w_idx + 1,
        acc,
        len(y_test_cls),
    )
    return {
        "oof_chunk": oof_chunk,
        "model": model,
        "static_cols": selected_static_cols,
        "accuracy": acc,
        "diag": diag,
    }


def _save_lgbm_wf_artifacts(
    config: Config,
    all_oof_preds: list[pl.DataFrame],
    last_lgbm_model: Any,
    last_feature_cols: list[str],
    last_window_accuracy: float | None,
    last_window_index: int,
    windows: list[Any],
    window_diagnostics: list[dict[str, Any]],
    stage_start: float,
) -> None:
    """Validate OOF predictions and persist static walk-forward artifacts.

    Args:
        config: Application configuration.
        all_oof_preds: List of per-window OOF Polars DataFrames.
        last_lgbm_model: LightGBM model from the last window.
        last_feature_cols: Feature column names.
        last_window_accuracy: Accuracy of the last window.
        last_window_index: 1-based index of the last window.
        windows: Walk-forward window objects.
        window_diagnostics: Per-window diagnostic dictionaries.
        stage_start: ``time.perf_counter()`` start timestamp.

    Raises:
        RuntimeError: If no predictions were generated.
        ValueError: If duplicate timestamps are found in OOF data.
    """
    import joblib

    from thesis.stage_4_training.lgbm.utils import _save_feature_importance

    if not all_oof_preds or last_lgbm_model is None:
        raise RuntimeError("No static OOF predictions generated")

    oof_df = _save_oof_predictions(
        config,
        all_oof_preds=all_oof_preds,
        window_diagnostics=window_diagnostics,
    )

    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_lgbm_model, model_path)
    _save_feature_importance(last_lgbm_model, last_feature_cols, config)

    if config.paths.session_dir:
        # Build per-window accuracy map from diagnostics
        per_window_accuracies: dict[str, float | None] = {}
        for d in window_diagnostics:
            key = str(d.get("window", ""))
            if key:
                per_window_accuracies[key] = d.get("accuracy")

        deployment_note = (
            f"Model saved from window {last_window_index}/{len(windows)} "
            "(the last chronological walk-forward window). "
            "This model has NOT seen any future data beyond its training window."
        )

        _save_training_history(
            config,
            {
                "architecture": "lgbm",
                "lightgbm": {
                    "artifact_strategy": "last_walk_forward_window",
                    "validation_protocol": {
                        "outer_windows": "bar_based_walk_forward_with_purge_embargo",
                        "lgbm_validation": "tail_20_percent_of_outer_train",
                    },
                    "last_window_accuracy": last_window_accuracy,
                    "best_iteration": int(last_lgbm_model.best_iteration_)
                    if hasattr(last_lgbm_model, "best_iteration_")
                    else None,
                    "n_features": len(last_feature_cols),
                    "n_classes": len(last_lgbm_model.classes_)
                    if hasattr(last_lgbm_model, "classes_")
                    else None,
                },
                "deployment_note": deployment_note,
                "per_window_accuracies": per_window_accuracies,
            },
        )
        _save_walk_forward_history(
            config,
            windows=windows,
            window_diagnostics=window_diagnostics,
            oof_len=len(oof_df),
            architecture="lgbm",
        )

    _log_walk_forward_complete(
        arch_name="lgbm",
        windows_count=len(windows),
        oof_len=len(oof_df),
        stage_start=stage_start,
        prefix="LGBM walk-forward complete",
    )

    _save_arch_copy(oof_df, "lgbm", config)


def train_lgbm_walk_forward(config: Config, *, expanded_features: bool = False) -> None:
    """Train LightGBM with walk-forward validation.

    Isolates whether GRU hidden states add value. Uses event-time purged
    windows, LightGBM, sample weights, and OOF prediction output. When
    ``expanded_features`` is True, uses all available feature columns.

    Args:
        config: Application configuration.
        expanded_features: If True, use all available features rather
            than the whitelist.
    """
    # 1. Prepare data and windows
    df, windows, feature_cols, is_regression_static = _prepare_static_wf_data(config)

    # 2. Walk-forward loop
    all_oof_preds: list[pl.DataFrame] = []
    last_lgbm_model = None
    last_feature_cols: list[str] = []
    last_window_accuracy: float | None = None
    last_window_index = 0
    window_diagnostics: list[dict[str, Any]] = []
    stage_start = time.perf_counter()
    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        console.rule(
            f"[bold cyan]LGBM window {w_idx + 1}/{len(windows)}[/]", style="cyan"
        )
        logger.info(
            "=== LGBM window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )
        result = _train_and_predict_static_window(
            config,
            w_idx,
            window,
            df,
            feature_cols,
            is_regression_static,
            expanded_features,
        )
        if result is None:
            continue
        all_oof_preds.append(result["oof_chunk"])
        window_diagnostics.append(result["diag"])
        last_lgbm_model = result["model"]
        last_feature_cols = result["static_cols"]
        last_window_accuracy = result["accuracy"]
        last_window_index = w_idx + 1
        logger.info(
            "LGBM window %d done (%.1fs)",
            w_idx + 1,
            time.perf_counter() - window_start,
        )

    # 3. Validate and persist
    _save_lgbm_wf_artifacts(
        config,
        all_oof_preds,
        last_lgbm_model,
        last_feature_cols,
        last_window_accuracy,
        last_window_index,
        windows,
        window_diagnostics,
        stage_start,
    )


def train_lgbm_fixed(config: Config) -> None:
    """Train LightGBM using the fixed train/val/test split.

    Args:
        config: Application configuration.

    Fixed split does not apply purge or embargo at split boundaries. With
    triple-barrier labels, boundary samples can leak future information from
    the adjacent split. For evaluation, prefer walk-forward validation.
    """
    from thesis.stage_4_training.lgbm import train_model

    logger.warning(
        "Fixed split mode does not apply purge/embargo — potential label leakage "
        "at split boundaries. Recommended: validation.method = 'sliding'."
    )
    train_model(config)
