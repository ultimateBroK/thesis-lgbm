"""Hybrid GRU + LightGBM — static train_model orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import polars as pl
from rich.panel import Panel
from rich.table import Table

from thesis.shared.config import Config
from thesis.shared.constants import EXCLUDE_COLS
from thesis.shared.ui import console
from thesis.stage_4_training.gru import (
    extract_hidden_states,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis.stage_4_training.lgbm.utils import (
    _align_splits_with_sequences,
    _build_hybrid_matrix,
    _compute_class_weights,
    _save_feature_importance,
    _train_fixed,
    _wrap_np,
)

logger = logging.getLogger("thesis.model")


def _normalize_label(lbl: int) -> str:
    """Normalize a class label for probability column naming."""
    if lbl < 0:
        return f"minus{abs(lbl)}"
    return str(lbl)


def _save_predictions(
    test_aligned: pl.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray,
    class_order: list,
    preds_path: Path,
) -> None:
    """Save predictions as Parquet and CSV files."""
    proba_cols = {
        f"pred_proba_class_{_normalize_label(cls)}": proba[:, idx]
        for idx, cls in enumerate(class_order)
    }
    preds_df = pl.DataFrame(
        {
            "timestamp": test_aligned["timestamp"],
            "true_label": y_test,
            "pred_label": preds.astype(np.int32),
            **proba_cols,
        }
    )
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(preds_path)
    csv_path = preds_path.with_suffix(".csv")
    preds_df.write_csv(csv_path)


def train_model(config: Config) -> None:
    """Train and evaluate the hybrid GRU + LightGBM model.

    This stage trains the GRU feature extractor, builds hybrid features,
    trains LightGBM, saves artifacts, and computes interpretation outputs.

    Args:
        config: Resolved application configuration.

    Raises:
        FileNotFoundError: If required split parquet files are missing.
    """
    stage_start = time.perf_counter()

    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)
    test_path = Path(config.paths.test_data)

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Split data not found: {p}. Run split stage first."
            )

    with console.status("[cyan]Loading train/val/test splits[/]"):
        train_df = pl.read_parquet(train_path)
        val_df = pl.read_parquet(val_path)
        test_df = pl.read_parquet(test_path)

    logger.info(
        "Splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
    )

    # --- 1. Train GRU feature extractor ---
    console.print(
        Panel(
            "Stage 4.1: [bold]GRU Feature Extractor[/]", style="magenta", padding=(0, 2)
        )
    )
    (
        gru_model,
        _gru_classifier,
        train_hidden,
        val_hidden,
        gru_history,
        gru_mean,
        gru_std,
        gru_cols,
    ) = train_gru(config, train_df, val_df)

    # Save GRU model (single source of truth: paths.gru_model)
    gru_path = Path(config.paths.gru_model)
    gru_path.parent.mkdir(parents=True, exist_ok=True)
    save_gru_model(gru_model, config, gru_path, mean=gru_mean, std=gru_std)

    # Extract hidden states for test set (using dynamically filtered gru_cols)
    test_seq, _, _ = prepare_sequences(test_df, gru_cols, config.gru.sequence_length)
    test_hidden = extract_hidden_states(
        gru_model,
        test_seq,
        config.gru.batch_size,
        mean=gru_mean,
        std=gru_std,
    )

    # --- 2. Align DataFrames with GRU sequences ---
    seq_len = config.gru.sequence_length
    train_aligned, val_aligned, test_aligned = _align_splits_with_sequences(
        train_df,
        val_df,
        test_df,
        train_hidden,
        val_hidden,
        test_hidden,
        seq_len,
    )

    # --- 3. Build hybrid feature matrix ---
    static_cols = [c for c in train_aligned.columns if c not in EXCLUDE_COLS]
    hidden_size = config.gru.hidden_size
    X_train, X_val, X_test, all_feature_cols = _build_hybrid_matrix(
        train_hidden,
        val_hidden,
        test_hidden,
        train_aligned,
        val_aligned,
        test_aligned,
        static_cols,
        hidden_size,
    )

    y_train = train_aligned["label"].to_numpy().astype(np.int32)
    y_val = val_aligned["label"].to_numpy().astype(np.int32)
    y_test = test_aligned["label"].to_numpy().astype(np.int32)
    train_weights = (
        train_aligned["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_aligned.columns
        else None
    )

    logger.info(
        "Features: %d total (%d GRU + %d static)",
        len(all_feature_cols),
        hidden_size,
        len(static_cols),
    )

    # --- 4. Train LightGBM ---
    console.print(
        Panel("Stage 4.2: [bold]LightGBM[/] (Fixed)", style="magenta", padding=(0, 2))
    )
    class_weights = _compute_class_weights(y_train)

    model = _train_fixed(
        X_train,
        y_train,
        X_val,
        y_val,
        class_weights,
        config,
        all_feature_cols,
        sample_weight=train_weights,
    )

    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model: %s", model_path)

    is_regression = config.model.objective == "regression"

    models_dir = model_path.parent
    history_path = models_dir / "training_history.json"
    lgbm_info: dict = {
        "best_iteration": int(model.best_iteration_)
        if hasattr(model, "best_iteration_")
        else None,
        "n_features": len(all_feature_cols),
        "objective": config.model.objective,
        "n_classes": int(model.n_classes_) if hasattr(model, "n_classes_") else None,
    }
    training_history = {
        "gru": gru_history,
        "lightgbm": lgbm_info,
    }
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    # --- 5. Generate test predictions ---
    console.print(
        Panel(
            "Stage 4.3: [bold]Predictions & Evaluation[/]",
            style="magenta",
            padding=(0, 2),
        )
    )

    if is_regression:
        raw_preds = model.predict(_wrap_np(X_test, all_feature_cols))
        # Threshold at 0: pred > 0 → Long (1), pred < 0 → Short (-1)
        preds = np.where(raw_preds > 0, 1, np.where(raw_preds < 0, -1, 0))
        proba = None  # No probability matrix for regression
    else:
        proba = model.predict_proba(_wrap_np(X_test, all_feature_cols))
        preds = model.classes_[np.argmax(proba, axis=1)]  # Explicit class mapping

    acc = (preds == y_test).mean()

    # Rich table for per-class results
    table = Table(title="Test Set Results", show_header=True, header_style="bold")
    table.add_column("Class", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Predicted", justify="right")

    label_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    for cls in [-1, 0, 1]:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = (preds[mask] == cls).mean()
            table.add_row(
                f"{label_map[cls]} ({cls})",
                str(mask.sum()),
                f"{cls_acc:.3f}",
                str((preds == cls).sum()),
            )

    console.print(table)
    logger.info("Test accuracy: %.4f", acc)

    if is_regression:
        # Save predictions with raw values and thresholded labels
        preds_path = Path(config.paths.predictions)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df = pl.DataFrame(
            {
                "timestamp": test_aligned["timestamp"],
                "true_label": y_test,
                "pred_label": preds.astype(np.int32),
                "pred_raw": raw_preds.astype(np.float64),
            }
        )
        preds_df.write_parquet(preds_path)
        csv_path = preds_path.with_suffix(".csv")
        preds_df.write_csv(csv_path)
    else:
        class_order = model.classes_.tolist()
        preds_path = Path(config.paths.predictions)
        _save_predictions(test_aligned, y_test, preds, proba, class_order, preds_path)

    # --- 6. Feature importance ---
    _save_feature_importance(model, all_feature_cols, config)

    # Final summary panel
    stage_time = time.perf_counter() - stage_start
    console.print(
        Panel(
            f"[bold green]Stage 4 complete[/]\n"
            f"  Accuracy: [bold]{acc:.4f}[/]\n"
            f"  GRU: {hidden_size} features ({config.gru.num_layers} layers)\n"
            f"  LightGBM: {len(all_feature_cols)} features,"
            f" best_iter={getattr(model, 'best_iteration_', 'N/A')}\n"
            f"  Time: {stage_time:.1f}s",
            style="green",
            padding=(0, 2),
        )
    )
