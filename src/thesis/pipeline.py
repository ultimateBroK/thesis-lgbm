"""Pipeline orchestration — sequential stage runner with walk-forward validation.

Stages:
    0. Data preparation (tick → OHLCV)
    1. Feature engineering
    2. Triple-barrier labeling
    3. Walk-forward training (sliding window: GRU + LightGBM per window)
    4. Backtest (on concatenated OOF predictions)
    5. Report generation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from thesis.config import Config
from thesis.constants import EXCLUDE_COLS
from thesis.ui import console, stage_header, stage_skip
from thesis.data import prepare_data
from thesis.features import generate_features
from thesis.labels import generate_labels
from thesis.validation import generate_windows, log_windows
from thesis.backtest import run_backtest
from thesis.report import generate_report

logger = logging.getLogger("thesis.pipeline")
STACK_CLASS_ORDER = np.array([-1, 0, 1], dtype=np.int32)
STACK_META_FEATURE_COLS = [
    "gru_pred_proba_class_minus1",
    "gru_pred_proba_class_0",
    "gru_pred_proba_class_1",
    "lgbm_pred_proba_class_minus1",
    "lgbm_pred_proba_class_0",
    "lgbm_pred_proba_class_1",
]


def _select_static_feature_cols(
    config: Config,
    df: pl.DataFrame,
    candidate_cols: list[str],
) -> list[str]:
    """Return compact, interpretable static features for LightGBM.

    Args:
        config: Runtime configuration containing the static feature whitelist.
        df: DataFrame slice used for model training or inference.
        candidate_cols: Fallback feature columns discovered from the dataset.

    Returns:
        Ordered feature names present in ``df``. Uses the centralized whitelist
        first and falls back to discovered candidates for tests or partial data.
    """
    available = [c for c in config.features.static_feature_cols if c in df.columns]
    if available:
        return available
    # Fallback keeps tests and partial feature sets usable.
    return [c for c in candidate_cols if c in df.columns]


# ---------------------------------------------------------------------------
# Stage runner with cache checking
# ---------------------------------------------------------------------------


def _run_stage(
    stage_num: int,
    config: Config,
    flag_name: str,
    cache_path: str | Path | None,
    work_fn: callable,
) -> None:
    """Execute a pipeline stage with cache checking."""
    flag = getattr(config.workflow, flag_name, False)
    if not flag:
        stage_skip(stage_num, "disabled")
        return

    if cache_path is not None:
        cache_path = Path(cache_path)
        if not config.workflow.force_rerun and cache_path.exists():
            stage_skip(stage_num, f"cached ({cache_path.name})")
            return

    stage_header(stage_num)
    work_fn(config)


# ---------------------------------------------------------------------------
# Walk-forward training loop
# ---------------------------------------------------------------------------


def _run_walk_forward_hybrid(config: Config) -> None:
    """Execute walk-forward sliding window training across all windows.

    For each window:
        1. Slice labeled data into train/test
        2. Apply purge & embargo
        3. Train GRU feature extractor on train
        4. Extract hidden states for train and test
        5. Build hybrid feature matrix (GRU hidden + static features)
        6. Train LightGBM on hybrid features
        7. Generate predictions on test slice
        8. Collect OOF predictions

    After all windows: concatenate OOF predictions and save for backtest.
    """
    import joblib
    import torch

    from thesis.gru import (
        train_gru,
        extract_hidden_states,
        prepare_sequences,
        save_gru_model,
    )
    from thesis.model import (
        _compute_class_weights,
        _train_fixed,
        _train_optuna,
        _wrap_np,
    )

    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data: %d rows", len(df))

    # Generate walk-forward windows
    v = config.validation
    windows = generate_windows(
        total_bars=len(df),
        train_window_bars=v.train_window_bars,
        test_window_bars=v.test_window_bars,
        step_bars=v.step_bars,
        purge_bars=v.purge_bars,
        embargo_bars=v.embargo_bars,
        min_train_bars=v.min_train_bars,
    )

    if not windows:
        raise RuntimeError("No valid walk-forward windows generated — check data size and window parameters")

    log_windows(windows, df, "timestamp")
    logger.info("Walk-forward: %d windows", len(windows))

    # Identify feature columns (exclude non-features)
    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)

    all_oof_preds: list[pl.DataFrame] = []
    gru_model = None
    gru_mean = None
    gru_std = None
    best_model = None
    best_feature_cols: list[str] = []
    best_accuracy = 0.0
    last_gru_history: list[dict] = []

    stage_start = time.perf_counter()

    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        logger.info(
            "=== Window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )

        # Slice data
        train_df = df.slice(
            window.train_start_idx, window.train_end_idx - window.train_start_idx
        )
        test_df = df.slice(
            window.test_start_idx, window.test_end_idx - window.test_start_idx
        )

        if len(train_df) < config.gru.sequence_length:
            logger.warning(
                "Window %d: train too small (%d), skipping", w_idx + 1, len(train_df)
            )
            continue

        # --- Train GRU ---
        # Use last 20% of training as validation for early stopping
        val_split = max(1, int(len(train_df) * 0.2))
        gru_train_df = train_df.head(len(train_df) - val_split)
        gru_val_df = train_df.tail(val_split)

        (
            gru_model,
            _classifier,
            _,  # train_hidden from train_gru – overwritten below after full-train extraction
            val_hidden,
            gru_history,
            gru_mean,
            gru_std,
            dynamic_gru_cols,
        ) = train_gru(config, gru_train_df, gru_val_df)

        # Extract hidden states for full train and test
        seq_len = config.gru.sequence_length
        train_seq, _, _ = prepare_sequences(train_df, dynamic_gru_cols, seq_len)
        test_seq, _, _ = prepare_sequences(test_df, dynamic_gru_cols, seq_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_hidden = extract_hidden_states(
            gru_model,
            train_seq,
            config.gru.batch_size,
            device=device,
            mean=gru_mean,
            std=gru_std,
        )
        test_hidden = extract_hidden_states(
            gru_model,
            test_seq,
            config.gru.batch_size,
            device=device,
            mean=gru_mean,
            std=gru_std,
        )

        # Align DataFrames with sequence outputs
        train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
        test_aligned = test_df.slice(seq_len - 1, len(test_hidden))

        if len(train_aligned) == 0 or len(test_aligned) == 0:
            logger.warning("Window %d: aligned data empty, skipping", w_idx + 1)
            continue

        # --- Build hybrid feature matrix ---
        static_cols = _select_static_feature_cols(config, train_aligned, feature_cols)
        hidden_size = config.gru.hidden_size
        gru_feat_names = [f"gru_h{i}" for i in range(hidden_size)]
        all_feature_cols = gru_feat_names + static_cols

        X_train = np.concatenate(
            [train_hidden, train_aligned.select(static_cols).to_numpy()], axis=1
        )
        X_test = np.concatenate(
            [test_hidden, test_aligned.select(static_cols).to_numpy()], axis=1
        )

        y_train = train_aligned["label"].to_numpy().astype(np.int32)
        y_test = test_aligned["label"].to_numpy().astype(np.int32)

        # Validation set for LightGBM = last portion of training
        val_split_idx = max(1, int(len(X_train) * 0.2))
        X_tr = X_train[:-val_split_idx]
        y_tr = y_train[:-val_split_idx]
        X_val = X_train[-val_split_idx:]
        y_val = y_train[-val_split_idx:]
        val_reference_price = float(
            train_aligned.slice(len(train_aligned) - val_split_idx, val_split_idx)[
                "close"
            ].median()
        )

        # --- Train LightGBM ---
        class_weights = _compute_class_weights(y_tr)

        # Use per-window Optuna trial limit if configured
        wf_trials = config.validation.wf_optuna_trials
        original_trials = config.model.optuna_trials
        if wf_trials > 0:
            config.model.optuna_trials = wf_trials

        if config.model.use_optuna:
            model = _train_optuna(
                X_tr,
                y_tr,
                X_val,
                y_val,
                class_weights,
                config,
                all_feature_cols,
                val_reference_price,
            )
        else:
            model = _train_fixed(
                X_tr, y_tr, X_val, y_val, class_weights, config, all_feature_cols
            )

        # Restore original Optuna setting
        config.model.optuna_trials = original_trials

        # --- Predict on test ---
        proba = model.predict_proba(_wrap_np(X_test, all_feature_cols))
        preds = model.classes_[np.argmax(proba, axis=1)]

        acc = (preds == y_test).mean()
        logger.info(
            "Window %d: accuracy=%.4f, test_samples=%d",
            w_idx + 1,
            acc,
            len(y_test),
        )

        # Track best model and its feature cols
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_feature_cols = all_feature_cols

        last_gru_history = gru_history

        # Collect OOF predictions
        class_order = model.classes_.tolist()
        proba_cols = {}
        for idx, cls in enumerate(class_order):
            label = f"minus{abs(cls)}" if cls < 0 else str(cls)
            proba_cols[f"pred_proba_class_{label}"] = proba[:, idx]

        oof_chunk = pl.DataFrame(
            {
                "timestamp": test_aligned["timestamp"],
                "true_label": y_test,
                "pred_label": preds.astype(np.int32),
                **proba_cols,
            }
        )
        all_oof_preds.append(oof_chunk)

        window_time = time.perf_counter() - window_start
        logger.info("Window %d done (%.1fs)", w_idx + 1, window_time)

    # --- Validate OOF predictions before saving ---
    if not all_oof_preds or gru_model is None:
        raise RuntimeError(
            "No OOF predictions generated — all walk-forward windows were skipped"
        )

    # --- Save final GRU model (last window only) ---
    if config.paths.session_dir:
        gru_path = Path(config.paths.session_dir) / "models" / "gru_model.pt"
        save_gru_model(gru_model, config, gru_path, mean=gru_mean, std=gru_std)

    # --- Concatenate OOF predictions ---

    oof_df = pl.concat(all_oof_preds)
    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.write_parquet(preds_path)
    oof_df.write_csv(preds_path.with_suffix(".csv"))

    # Save best LightGBM model
    if best_model is not None:
        model_path = Path(config.paths.model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)

    # Save feature importance from best LightGBM model
    if best_model is not None and best_feature_cols:
        from thesis.model import _save_feature_importance

        _save_feature_importance(best_model, best_feature_cols, config)

    # Save training history (GRU + LightGBM info)
    if config.paths.session_dir:
        models_dir = Path(config.paths.session_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        history_path = models_dir / "training_history.json"

        lgbm_info: dict[str, Any] = {}
        if best_model is not None:
            lgbm_info = {
                "best_iteration": int(best_model.best_iteration_)
                if hasattr(best_model, "best_iteration_")
                else None,
                "n_features": len(best_feature_cols),
                "n_classes": len(best_model.classes_)
                if hasattr(best_model, "classes_")
                else None,
            }

        with open(history_path, "w") as f:
            json.dump({"gru": last_gru_history, "lightgbm": lgbm_info}, f, indent=2)
        logger.info("Training history saved to %s", history_path)

    # Save walk-forward history
    if config.paths.session_dir:
        wf_path = (
            Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
        )
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        history = {
            "num_windows": len(windows),
            "total_oof_predictions": len(oof_df),
            "window_details": [
                {
                    "window": i + 1,
                    "train_start_idx": w.train_start_idx,
                    "train_end_idx": w.train_end_idx,
                    "test_start_idx": w.test_start_idx,
                    "test_end_idx": w.test_end_idx,
                }
                for i, w in enumerate(windows)
            ],
        }
        with open(wf_path, "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.perf_counter() - stage_start
    logger.info(
        "Walk-forward complete: %d windows, %d OOF predictions (%.1fs)",
        len(windows),
        len(oof_df),
        total_time,
    )


def _label_suffix(class_label: int) -> str:
    """Return the canonical probability-column suffix for a class label."""
    return f"minus{abs(class_label)}" if class_label < 0 else str(class_label)


def _align_probability_matrix(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
) -> np.ndarray:
    """Align class probabilities to the canonical ``[-1, 0, 1]`` order."""
    aligned = np.zeros((len(proba), len(STACK_CLASS_ORDER)), dtype=np.float64)
    index_by_class = {int(cls): idx for idx, cls in enumerate(class_order)}
    for target_idx, cls in enumerate(STACK_CLASS_ORDER):
        source_idx = index_by_class.get(int(cls))
        if source_idx is not None:
            aligned[:, target_idx] = proba[:, source_idx]
    return aligned


def _probability_columns(
    proba: np.ndarray,
    class_order: list[int] | np.ndarray,
    *,
    prefix: str = "pred_proba_class_",
) -> dict[str, np.ndarray]:
    """Build canonical probability columns for ``{-1, 0, 1}``."""
    aligned = _align_probability_matrix(proba, class_order)
    return {
        f"{prefix}{_label_suffix(int(cls))}": aligned[:, idx]
        for idx, cls in enumerate(STACK_CLASS_ORDER)
    }


def _split_tail_frame(
    df: pl.DataFrame,
    *,
    fraction: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a DataFrame into head train and tail validation slices."""
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to create a train/validation split")
    val_size = min(max(1, int(len(df) * fraction)), len(df) - 1)
    return df.head(len(df) - val_size), df.tail(val_size)


def _split_tail_arrays(
    X: np.ndarray,
    y: np.ndarray,
    reference_prices: np.ndarray,
    *,
    fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Split matrices into train/validation tails and return val price anchor."""
    if len(X) < 2:
        raise ValueError("Need at least 2 rows to create a train/validation split")
    val_size = min(max(1, int(len(X) * fraction)), len(X) - 1)
    X_tr = X[:-val_size]
    y_tr = y[:-val_size]
    X_val = X[-val_size:]
    y_val = y[-val_size:]
    reference_price = float(np.median(reference_prices[-val_size:]))
    return X_tr, y_tr, X_val, y_val, reference_price


def _fit_lgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    reference_price: float,
    config: Config,
    feature_cols: list[str],
    *,
    trials_override: int | None = None,
) -> Any:
    """Train a LightGBM model using the repo's fixed or Optuna path."""
    from thesis.model import (
        _compute_class_weights,
        _train_fixed,
        _train_optuna,
    )

    class_weights = _compute_class_weights(y_train)
    original_trials = config.model.optuna_trials
    if trials_override is not None and trials_override > 0:
        config.model.optuna_trials = trials_override

    try:
        if config.model.use_optuna:
            return _train_optuna(
                X_train,
                y_train,
                X_val,
                y_val,
                class_weights,
                config,
                feature_cols,
                reference_price,
            )
        return _train_fixed(
            X_train,
            y_train,
            X_val,
            y_val,
            class_weights,
            config,
            feature_cols,
        )
    finally:
        config.model.optuna_trials = original_trials


def _stacking_artifact_path(config: Config, filename: str) -> Path:
    """Resolve a stacking-side artifact path under the session models directory."""
    if config.paths.session_dir:
        base_dir = Path(config.paths.session_dir) / "models"
    else:
        base_dir = Path(config.paths.model).parent
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / filename


def _stacking_predictions_path(config: Config, filename: str) -> Path:
    """Resolve a stacking-side artifact path under the session predictions directory."""
    if config.paths.session_dir:
        base_dir = Path(config.paths.session_dir) / "predictions"
    else:
        base_dir = Path(config.paths.predictions).parent
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / filename


def _validate_stacking_config(config: Config) -> None:
    """Fail fast on unsupported stacking configuration combinations."""
    if config.validation.method != "sliding":
        raise ValueError(
            "True stacking is implemented only for validation.method='sliding'"
        )
    if config.model.architecture != "stacking":
        raise ValueError("Stacking path requires model.architecture='stacking'")
    if config.stacking.base_models != ["gru", "lgbm"]:
        raise ValueError("Stacking v1 supports base_models=['gru', 'lgbm'] only")
    if config.stacking.meta_model != "lightgbm":
        raise ValueError("Stacking v1 supports meta_model='lightgbm' only")
    if not config.stacking.use_probability_features_only:
        raise ValueError("Stacking v1 supports probability-only meta features")


def _fit_stacking_base_models(
    config: Config,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
    *,
    trials_override: int | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Train GRU + static LightGBM base learners and score one outer test fold."""
    import torch

    from thesis.gru import predict_gru_proba, prepare_sequences, train_gru
    from thesis.model import _wrap_np

    gru_train_df, gru_val_df = _split_tail_frame(train_df)
    (
        gru_model,
        gru_classifier,
        _train_hidden,
        _val_hidden,
        gru_history,
        gru_mean,
        gru_std,
        dynamic_gru_cols,
    ) = train_gru(config, gru_train_df, gru_val_df)

    seq_len = config.gru.sequence_length
    train_seq, _, _ = prepare_sequences(train_df, dynamic_gru_cols, seq_len)
    test_seq, _, _ = prepare_sequences(test_df, dynamic_gru_cols, seq_len)

    train_aligned = train_df.slice(seq_len - 1, len(train_seq))
    test_aligned = test_df.slice(seq_len - 1, len(test_seq))

    if len(train_aligned) < 2 or len(test_aligned) == 0:
        raise ValueError("Aligned train/test slices are too small for stacking")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru_proba = predict_gru_proba(
        gru_model,
        gru_classifier,
        test_seq,
        config.gru.batch_size,
        device=device,
        mean=gru_mean,
        std=gru_std,
    )

    static_cols = _select_static_feature_cols(config, train_aligned, feature_cols)
    X_train_static = train_aligned.select(static_cols).to_numpy()
    X_test_static = test_aligned.select(static_cols).to_numpy()
    y_train_static = train_aligned["label"].to_numpy().astype(np.int32)

    (
        X_tr,
        y_tr,
        X_val,
        y_val,
        val_reference_price,
    ) = _split_tail_arrays(
        X_train_static,
        y_train_static,
        train_aligned["close"].to_numpy(),
    )

    lgbm_model = _fit_lgbm_model(
        X_tr,
        y_tr,
        X_val,
        y_val,
        val_reference_price,
        config,
        static_cols,
        trials_override=trials_override,
    )
    lgbm_raw_proba = lgbm_model.predict_proba(_wrap_np(X_test_static, static_cols))

    y_test = test_aligned["label"].to_numpy().astype(np.int32)
    gru_aligned_proba = _align_probability_matrix(gru_proba, STACK_CLASS_ORDER)
    lgbm_aligned_proba = _align_probability_matrix(lgbm_raw_proba, lgbm_model.classes_)

    base_chunk = pl.DataFrame(
        {
            "timestamp": test_aligned["timestamp"],
            "close": test_aligned["close"],
            "true_label": y_test,
            **_probability_columns(
                gru_aligned_proba, STACK_CLASS_ORDER, prefix="gru_pred_proba_class_"
            ),
            **_probability_columns(
                lgbm_aligned_proba,
                STACK_CLASS_ORDER,
                prefix="lgbm_pred_proba_class_",
            ),
        }
    )

    metadata = {
        "gru_model": gru_model,
        "gru_classifier": gru_classifier,
        "gru_history": gru_history,
        "gru_mean": gru_mean,
        "gru_std": gru_std,
        "gru_cols": dynamic_gru_cols,
        "static_cols": static_cols,
        "lgbm_model": lgbm_model,
    }
    return base_chunk, metadata


def _fit_meta_model(
    meta_train_df: pl.DataFrame,
    config: Config,
    *,
    trials_override: int | None = None,
) -> Any:
    """Fit the LightGBM meta-learner on prior-fold OOF base probabilities."""
    X_meta = meta_train_df.select(STACK_META_FEATURE_COLS).to_numpy()
    y_meta = meta_train_df["true_label"].to_numpy().astype(np.int32)
    (
        X_tr,
        y_tr,
        X_val,
        y_val,
        val_reference_price,
    ) = _split_tail_arrays(
        X_meta,
        y_meta,
        meta_train_df["close"].to_numpy(),
    )
    return _fit_lgbm_model(
        X_tr,
        y_tr,
        X_val,
        y_val,
        val_reference_price,
        config,
        STACK_META_FEATURE_COLS,
        trials_override=trials_override,
    )


def _predict_meta_chunk(meta_model: Any, chunk: pl.DataFrame) -> pl.DataFrame:
    """Generate canonical final predictions from a fitted meta-learner."""
    from thesis.model import _wrap_np

    X_chunk = chunk.select(STACK_META_FEATURE_COLS).to_numpy()
    raw_proba = meta_model.predict_proba(_wrap_np(X_chunk, STACK_META_FEATURE_COLS))
    aligned_proba = _align_probability_matrix(raw_proba, meta_model.classes_)
    preds = STACK_CLASS_ORDER[np.argmax(aligned_proba, axis=1)].astype(np.int32)
    return pl.DataFrame(
        {
            "timestamp": chunk["timestamp"],
            "true_label": chunk["true_label"],
            "pred_label": preds,
            **_probability_columns(aligned_proba, STACK_CLASS_ORDER),
        }
    )


def _save_stacking_final_refit(
    config: Config,
    df: pl.DataFrame,
    feature_cols: list[str],
    base_oof_df: pl.DataFrame,
    last_base_metadata: dict[str, Any],
) -> None:
    """Train final deployable stacking artifacts on the latest full dataset."""
    import joblib

    from thesis.gru import save_gru_model, train_gru

    full_train_df, full_val_df = _split_tail_frame(df)
    (
        final_gru_model,
        final_gru_classifier,
        _train_hidden,
        _val_hidden,
        final_gru_history,
        final_gru_mean,
        final_gru_std,
        final_gru_cols,
    ) = train_gru(config, full_train_df, full_val_df)

    gru_path = Path(config.paths.gru_model)
    gru_path.parent.mkdir(parents=True, exist_ok=True)
    save_gru_model(
        final_gru_model,
        config,
        gru_path,
        mean=final_gru_mean,
        std=final_gru_std,
        classifier=final_gru_classifier,
    )

    seq_len = config.gru.sequence_length
    from thesis.gru import prepare_sequences

    full_seq, _, _ = prepare_sequences(df, final_gru_cols, seq_len)
    full_aligned = df.slice(seq_len - 1, len(full_seq))
    static_cols = _select_static_feature_cols(config, full_aligned, feature_cols)
    X_full_static = full_aligned.select(static_cols).to_numpy()
    y_full_static = full_aligned["label"].to_numpy().astype(np.int32)
    (
        X_tr,
        y_tr,
        X_val,
        y_val,
        val_reference_price,
    ) = _split_tail_arrays(
        X_full_static,
        y_full_static,
        full_aligned["close"].to_numpy(),
    )
    wf_trials = config.validation.wf_optuna_trials
    lgbm_base_model = _fit_lgbm_model(
        X_tr,
        y_tr,
        X_val,
        y_val,
        val_reference_price,
        config,
        static_cols,
        trials_override=wf_trials if wf_trials > 0 else None,
    )

    lgbm_base_path = _stacking_artifact_path(config, "lgbm_base_model.pkl")
    joblib.dump(lgbm_base_model, lgbm_base_path)

    meta_model = _fit_meta_model(
        base_oof_df.sort("timestamp"),
        config,
        trials_override=wf_trials if wf_trials > 0 else None,
    )
    meta_model_path = Path(config.paths.model)
    meta_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, meta_model_path)

    bundle = {
        "architecture": "stacking",
        "class_order": STACK_CLASS_ORDER.tolist(),
        "base_models": list(config.stacking.base_models),
        "meta_model": config.stacking.meta_model,
        "gru_model_path": str(gru_path),
        "lgbm_base_model_path": str(lgbm_base_path),
        "meta_model_path": str(meta_model_path),
        "gru_feature_cols": final_gru_cols,
        "static_feature_cols": static_cols,
        "meta_feature_cols": list(STACK_META_FEATURE_COLS),
        "oof_base_predictions_path": str(
            _stacking_predictions_path(config, "base_oof_predictions.parquet")
        ),
        "config_snapshot": {
            "validation_method": config.validation.method,
            "wf_optuna_trials": config.validation.wf_optuna_trials,
            "final_refit": config.stacking.final_refit,
        },
    }
    stack_bundle_path = Path(config.paths.stack_bundle)
    stack_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, stack_bundle_path)

    history_path = _stacking_artifact_path(config, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {
                "architecture": "stacking",
                "base_models": {
                    "gru": {
                        "feature_cols": final_gru_cols,
                        "hidden_size": config.gru.hidden_size,
                        "history": final_gru_history,
                    },
                    "lgbm": {
                        "feature_cols": static_cols,
                        "best_iteration": int(lgbm_base_model.best_iteration_)
                        if hasattr(lgbm_base_model, "best_iteration_")
                        else None,
                    },
                },
                "meta_model": {
                    "feature_cols": list(STACK_META_FEATURE_COLS),
                    "best_iteration": int(meta_model.best_iteration_)
                    if hasattr(meta_model, "best_iteration_")
                    else None,
                },
                "last_oof_base_models": {
                    "gru_feature_cols": last_base_metadata["gru_cols"],
                    "static_feature_cols": last_base_metadata["static_cols"],
                },
            },
            f,
            indent=2,
        )


def _run_walk_forward_stacking(config: Config) -> None:
    """Execute true OOF stacking with GRU and LightGBM base learners."""
    _validate_stacking_config(config)

    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data for stacking: %d rows", len(df))

    windows = generate_windows(
        total_bars=len(df),
        train_window_bars=config.validation.train_window_bars,
        test_window_bars=config.validation.test_window_bars,
        step_bars=config.validation.step_bars,
        purge_bars=config.validation.purge_bars,
        embargo_bars=config.validation.embargo_bars,
        min_train_bars=config.validation.min_train_bars,
    )
    if not windows:
        raise ValueError("No valid walk-forward windows generated for stacking")

    log_windows(windows, df, "timestamp")
    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)
    all_base_oof: list[pl.DataFrame] = []
    causal_meta_preds: list[pl.DataFrame] = []
    last_base_metadata: dict[str, Any] | None = None
    skipped_meta_folds = 0
    stage_start = time.perf_counter()
    wf_trials = config.validation.wf_optuna_trials

    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        train_df = df.slice(
            window.train_start_idx, window.train_end_idx - window.train_start_idx
        )
        test_df = df.slice(
            window.test_start_idx, window.test_end_idx - window.test_start_idx
        )
        if (
            len(train_df) < config.gru.sequence_length
            or len(test_df) < config.gru.sequence_length
        ):
            logger.warning(
                "Stacking window %d too small for GRU sequences; skipping", w_idx + 1
            )
            continue

        base_chunk, last_base_metadata = _fit_stacking_base_models(
            config,
            train_df,
            test_df,
            feature_cols,
            trials_override=wf_trials if wf_trials > 0 else None,
        )
        base_chunk = base_chunk.with_columns(pl.lit(w_idx + 1).alias("fold"))
        all_base_oof.append(base_chunk)

        prior_base_folds = len(all_base_oof) - 1
        if prior_base_folds < config.stacking.min_meta_train_folds:
            skipped_meta_folds += 1
            logger.info(
                "Stacking window %d: meta warmup skip (need %d prior folds)",
                w_idx + 1,
                config.stacking.min_meta_train_folds,
            )
            continue

        meta_train_df = pl.concat(all_base_oof[:-1]).sort("timestamp")
        if len(meta_train_df) < config.stacking.min_meta_train_rows:
            skipped_meta_folds += 1
            logger.info(
                "Stacking window %d: meta warmup skip (have %d rows, need %d)",
                w_idx + 1,
                len(meta_train_df),
                config.stacking.min_meta_train_rows,
            )
            continue

        meta_model = _fit_meta_model(
            meta_train_df,
            config,
            trials_override=wf_trials if wf_trials > 0 else None,
        )
        current_preds = _predict_meta_chunk(meta_model, base_chunk.sort("timestamp"))
        causal_meta_preds.append(current_preds)

        logger.info(
            "Stacking window %d done: base_rows=%d meta_rows=%d (%.1fs)",
            w_idx + 1,
            len(base_chunk),
            len(current_preds),
            time.perf_counter() - window_start,
        )

    if not all_base_oof:
        raise ValueError("No base OOF predictions generated for stacking")
    if not causal_meta_preds:
        raise ValueError(
            "No causal meta predictions were produced; lower stacking.min_meta_train_rows "
            "or provide more walk-forward folds"
        )
    if last_base_metadata is None:
        raise ValueError("Stacking base-model metadata was not captured")

    base_oof_df = pl.concat(all_base_oof).sort("timestamp")
    base_oof_path = _stacking_predictions_path(config, "base_oof_predictions.parquet")
    base_oof_df.write_parquet(base_oof_path)
    base_oof_df.write_csv(base_oof_path.with_suffix(".csv"))

    final_preds_df = pl.concat(causal_meta_preds).sort("timestamp")
    preds_path = Path(config.paths.predictions)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    final_preds_df.write_parquet(preds_path)
    final_preds_df.write_csv(preds_path.with_suffix(".csv"))

    if config.stacking.final_refit:
        _save_stacking_final_refit(
            config,
            df,
            feature_cols,
            base_oof_df,
            last_base_metadata,
        )

    if config.paths.session_dir:
        wf_path = (
            Path(config.paths.session_dir) / "reports" / "walk_forward_history.json"
        )
        wf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(wf_path, "w") as f:
            json.dump(
                {
                    "architecture": "stacking",
                    "num_windows": len(windows),
                    "base_oof_rows": len(base_oof_df),
                    "meta_oof_rows": len(final_preds_df),
                    "skipped_meta_folds": skipped_meta_folds,
                    "window_details": [
                        {
                            "window": i + 1,
                            "train_start_idx": w.train_start_idx,
                            "train_end_idx": w.train_end_idx,
                            "test_start_idx": w.test_start_idx,
                            "test_end_idx": w.test_end_idx,
                        }
                        for i, w in enumerate(windows)
                    ],
                },
                f,
                indent=2,
            )

    total_time = time.perf_counter() - stage_start
    logger.info(
        "Stacking walk-forward complete: %d windows, %d base OOF rows, %d meta rows (%.1fs)",
        len(windows),
        len(base_oof_df),
        len(final_preds_df),
        total_time,
    )


def _run_walk_forward(config: Config) -> None:
    """Dispatch walk-forward training to the configured architecture."""
    architecture = config.model.architecture
    if architecture == "stacking":
        logger.info("Using true stacking walk-forward pipeline")
        _run_walk_forward_stacking(config)
        return

    if architecture != "hybrid":
        raise ValueError(f"Unsupported model.architecture: {architecture!r}")

    logger.info("Using hybrid walk-forward pipeline")
    _run_walk_forward_hybrid(config)


def _run_static_train(config: Config) -> None:
    """Run traditional static train/val/test split training."""
    from thesis.model import train_model

    train_model(config)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(config: Config) -> None:
    """Execute the full thesis pipeline.

    Stages:
        0. Data preparation (tick → OHLCV)
        1. Feature engineering
        2. Triple-barrier labeling
        3. Walk-forward model training (GRU + LightGBM per window)
        4. Backtest (on concatenated OOF predictions)
        5. Report generation

    Args:
        config: Loaded application configuration.
    """
    # Stage 0: Prepare OHLCV from raw ticks
    _run_stage(0, config, "run_data_pipeline", config.paths.ohlcv, prepare_data)

    # Stage 1: Features
    _run_stage(
        1,
        config,
        "run_feature_engineering",
        config.paths.features,
        generate_features,
    )

    # Stage 2: Labels
    _run_stage(2, config, "run_label_generation", config.paths.labels, generate_labels)

    # Stage 3: Training (walk-forward or static)
    if config.validation.method == "sliding":
        stage_header(3)
        logger.info(
            "Using walk-forward sliding window validation (%s architecture)",
            config.model.architecture,
        )
        if config.workflow.run_model_training:
            _run_walk_forward(config)
        else:
            stage_skip(3, "disabled")
    else:
        if config.model.architecture == "stacking":
            raise ValueError(
                "True stacking is implemented only for validation.method='sliding'"
            )
        logger.info("Using static train/val/test split")
        _run_stage(3, config, "run_model_training", None, _run_static_train)

    # Stage 4: Backtest
    _run_stage(
        4,
        config,
        "run_backtest",
        None,
        run_backtest,
    )

    # Stage 5: Report
    _run_stage(
        5,
        config,
        "run_reporting",
        None,
        generate_report,
    )

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")
    console.print()
