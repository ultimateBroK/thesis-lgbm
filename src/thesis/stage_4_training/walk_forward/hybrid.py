"""Hybrid (GRU+LGBM) walk-forward training loop."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import CENSORED_LABEL, EXCLUDE_COLS
from thesis.shared.ui import console
from thesis.stage_4_training.validation import generate_windows, log_windows
from thesis.stage_4_training.walk_forward.artifacts import _save_wf_artifacts
from thesis.stage_4_training.walk_forward.utils import (
    _CLASS_ORDER,
    _add_prediction_diagnostics,
    _align_probability_matrix,
    _log_gru_signal_quality,
    _one_hot_proba_columns,
    _probability_columns,
    _select_static_feature_cols,
    _window_dates,
    _window_diagnostics,
    fit_static_feature_pipeline,
)

logger = logging.getLogger("thesis.pipeline")

# --- Orchestration-level constants ---

_PCA_VARIANCE_THRESHOLD = 0.50  # Explained variance below → mostly noise

# --- Validation Split ---
_VALIDATION_SPLIT_FRACTION = 0.2  # Tail validation split for GRU/LGBM/static

# --- Regression Threshold ---
_REGRESSION_DIRECTION_THRESHOLD = (
    0  # Zero threshold for regression-to-direction mapping
)


def _compute_regression_target(
    df: pl.DataFrame, config: Config
) -> tuple[pl.DataFrame, bool]:
    """Pre-compute regression target column when objective is 'regression'.

    The last ``horizon_bars`` rows are set to NaN (insufficient forward
    data) and their ``label`` is set to ``CENSORED_LABEL`` so downstream
    filters exclude them.  The rows are dropped before returning.

    Args:
        df: Labeled Polars DataFrame containing a ``close`` column.
        config: Application configuration.

    Returns:
        ``(df_maybe_augmented, is_regression)`` — the DataFrame is
        augmented with a ``regression_target`` column when the
        objective is ``"regression"``; otherwise returned unchanged
        with ``is_regression=False``.
    """
    is_regression = config.model.objective == "regression"
    gru_needs_regression = config.gru.objective == "regression"
    if not is_regression and not gru_needs_regression:
        return df, False

    if "close" not in df.columns:
        raise ValueError(
            "Regression objective requires 'close' column in labeled data. "
            "Ensure feature engineering includes OHLCV data."
        )
    horizon = config.labels.horizon_bars
    close = df["close"].to_numpy()
    n = len(close)

    # Compute forward returns, leaving the last ``horizon`` rows as NaN.
    reg_target = np.full(n, np.nan, dtype=np.float64)
    close_future = np.roll(close, -horizon)[: n - horizon]
    reg_target[: n - horizon] = (close_future - close[: n - horizon]) / close[
        : n - horizon
    ]

    # Mark tail rows as censored so downstream _filter_censored excludes them.
    label_arr = df["label"].to_numpy().copy()
    tail_start = max(0, n - horizon)
    label_arr[tail_start:] = CENSORED_LABEL

    df = df.with_columns(
        [
            pl.Series("regression_target", reg_target),
            pl.Series("label", label_arr),
        ]
    )

    # Drop rows where the regression target is NaN (tail censored rows).
    n_before = len(df)
    df = df.filter(pl.col("regression_target").is_not_nan())
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Dropped %d regression tail rows (%d horizon bars) — "
            "insufficient forward horizon",
            n_dropped,
            horizon,
        )

    logger.info(
        "Regression target computed: horizon=%d bars, mean=%.6f, std=%.6f",
        horizon,
        float(np.nanmean(reg_target)),
        float(np.nanstd(reg_target)),
    )
    return df, is_regression


def _prepare_wf_data(
    config: Config,
) -> tuple[pl.DataFrame, list, list[str], bool]:
    """Load labeled data, generate walk-forward windows, and return prepared state.

    Args:
        config: Application configuration.

    Returns:
        ``(df, windows, feature_cols, is_regression)`` tuple containing the
        full labeled DataFrame, list of walk-forward window objects, sorted
        feature column names, and whether regression objective is active.

    Raises:
        FileNotFoundError: If the labels parquet file does not exist.
        RuntimeError: If no valid walk-forward windows were generated.
        ValueError: If the purge/embargo gap is smaller than the GRU
            sequence length (sequence leakage risk).
    """
    labels_path = Path(config.paths.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    with console.status(f"[cyan]Loading labels[/] {labels_path}"):
        df = pl.read_parquet(labels_path)
    logger.info("Loaded labeled data: %d rows", len(df))
    df, is_regression = _compute_regression_target(df, config)

    event_end = df["event_end"].to_numpy() if "event_end" in df.columns else None
    if event_end is None:
        logger.warning(
            "Labels lack event_end column — falling back to fixed-bar purge. "
            "Regenerate labels to enable event-time purging."
        )

    v = config.validation
    windows = generate_windows(
        total_bars=len(df),
        train_window_bars=v.train_window_bars,
        test_window_bars=v.test_window_bars,
        step_bars=v.step_bars,
        purge_bars=v.purge_bars,
        embargo_bars=v.embargo_bars,
        min_train_bars=v.min_train_bars,
        event_end=event_end,
    )
    if not windows:
        raise RuntimeError(
            "No valid walk-forward windows generated"
            " — check data size and window parameters"
        )

    # P0-1: Guard against sequence leakage
    gap_bars = (
        v.embargo_bars if event_end is not None else v.purge_bars + v.embargo_bars
    )
    seq_len = config.gru.sequence_length
    if gap_bars < seq_len:
        raise ValueError(
            f"Leakage risk: purge/embargo gap ({gap_bars} bars) < GRU sequence_length "
            f"({seq_len} bars). Test sequences would overlap with training data. "
            f"Increase embargo_bars to at least {seq_len}."
        )

    log_windows(windows, df, "timestamp")
    logger.info("Walk-forward: %d bar-based windows", len(windows))

    feature_cols = sorted(c for c in df.columns if c not in EXCLUDE_COLS)
    return df, windows, feature_cols, is_regression


def _wf_gru_phase(
    config: Config, w_idx: int, window: Any, df: pl.DataFrame
) -> dict[str, Any] | None:
    """GRU phase of a hybrid window: slice, train, extract hidden, align, PCA.

    Args:
        config: Application configuration.
        w_idx: Zero-based window index for logging.
        window: Walk-forward window object with ``train_start_idx``,
            ``train_end_idx``, ``test_start_idx``, ``test_end_idx``.
        df: Full labeled Polars DataFrame.

    Returns:
        State dictionary with ``gru_model``, normalization
        parameters, training history, aligned DataFrames, and
        hidden-state arrays, or ``None`` if the window is too small.
    """
    import torch

    from thesis.stage_4_training.gru import (
        extract_hidden_states,
        prepare_sequences,
        train_gru,
    )

    # Slice
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
        return None

    # Train GRU
    val_split = max(1, int(len(train_df) * _VALIDATION_SPLIT_FRACTION))
    gru_train_df = train_df.head(len(train_df) - val_split)
    gru_val_df = train_df.tail(val_split)
    (gru_model, _, _, _, gru_history, gru_mean, gru_std, dynamic_gru_cols) = train_gru(
        config, gru_train_df, gru_val_df, window_index=w_idx
    )

    # Extract hidden states
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

    # Align DataFrames
    train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
    test_aligned = test_df.slice(seq_len - 1, len(test_hidden))
    if train_aligned.is_empty() or test_aligned.is_empty():
        logger.warning("Window %d: aligned data empty, skipping", w_idx + 1)
        return None
    train_dates = _window_dates(train_df)
    test_dates = _window_dates(test_df)
    pred_dates = _window_dates(test_aligned)
    logger.info(
        "Window %d alignment: train_start=%s train_end=%s test_start=%s "
        "test_end=%s raw_test_rows=%d aligned_test_rows=%d "
        "dropped_by_sequence=%d pred_start=%s pred_end=%s",
        w_idx + 1,
        train_dates["start"],
        train_dates["end"],
        test_dates["start"],
        test_dates["end"],
        len(test_df),
        len(test_aligned),
        len(test_df) - len(test_aligned),
        pred_dates["start"],
        pred_dates["end"],
    )

    # PCA on GRU hidden states
    pca_k = config.gru.pca_components
    if pca_k > 0:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=pca_k, random_state=config.workflow.random_seed)
        pca.fit(train_hidden)
        train_hidden = pca.transform(train_hidden)
        test_hidden = pca.transform(test_hidden)
        explained = float(pca.explained_variance_ratio_.sum())
        logger.info(
            "GRU hidden states: %d→%d PCs, explained variance=%.1f%%",
            config.gru.hidden_size,
            pca_k,
            explained * 100,
        )
        if explained < _PCA_VARIANCE_THRESHOLD:
            logger.warning(
                "GRU hidden state space appears mostly noise"
                " (%.1f%% explained by %d PCs)",
                explained * 100,
                pca_k,
            )

    return {
        "gru_model": gru_model,
        "gru_mean": gru_mean,
        "gru_std": gru_std,
        "gru_history": gru_history,
        "train_aligned": train_aligned,
        "test_aligned": test_aligned,
        "train_hidden": train_hidden,
        "test_hidden": test_hidden,
    }


def _wf_format_predictions(
    model: Any, X_test: np.ndarray, all_feature_cols: list[str], is_regression: bool
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Generate predictions and aligned probability matrix.

    Args:
        model: Trained LightGBM model (classifier or regressor).
        X_test: Test feature matrix.
        all_feature_cols: Feature column names for wrapping.
        is_regression: Whether the model is a regressor.

    Returns:
        ``(preds, aligned_proba, proba, raw_preds)`` — predicted class
        labels, aligned 3-column probability matrix, raw LightGBM
        probability output, and raw regression predictions (``None``
        for classification).
    """
    from thesis.stage_4_training.lgbm.utils import _wrap_np

    if is_regression:
        raw_preds = model.predict(_wrap_np(X_test, all_feature_cols))
        preds = np.where(
            raw_preds > _REGRESSION_DIRECTION_THRESHOLD,
            1,
            np.where(raw_preds < _REGRESSION_DIRECTION_THRESHOLD, -1, 0),
        ).astype(np.int32)
        aligned_proba = np.zeros((len(raw_preds), 3), dtype=np.float64)
        for i, p in enumerate(preds):
            aligned_proba[i, {-1: 0, 0: 1, 1: 2}[int(p)]] = 1.0
        return preds, aligned_proba, None, raw_preds
    else:
        proba = model.predict_proba(_wrap_np(X_test, all_feature_cols))
        aligned_proba = _align_probability_matrix(proba, model.classes_)
        preds = _CLASS_ORDER[np.argmax(aligned_proba, axis=1)]
        return preds, aligned_proba, proba, None


def _wf_build_predict_phase(
    config: Config,
    w_idx: int,
    gru_state: dict[str, Any],
    feature_cols: list[str],
    is_regression: bool,
) -> dict[str, Any]:
    """Build hybrid matrix, train LGBM, predict, return full window result.

    Args:
        config: Application configuration.
        w_idx: Zero-based window index.
        gru_state: GRU phase state dict from ``_wf_gru_phase``.
        feature_cols: Candidate feature column names.
        is_regression: Whether the model objective is regression.

    Returns:
        Full window result dict containing the GRU state, test labels,
        predictions, probabilities, trained LGBM model, feature
        columns, accuracy, diagnostics, and class ordering.
    """
    from thesis.stage_4_training.lgbm.utils import (
        _compute_class_weights,
        _compute_distribution_shift_weights,
        _train_fixed,
    )

    train_aligned = gru_state["train_aligned"]
    test_aligned = gru_state["test_aligned"]
    train_hidden = gru_state["train_hidden"]
    test_hidden = gru_state["test_hidden"]

    # ── Build hybrid feature matrix ──
    static_cols = _select_static_feature_cols(config, train_aligned, feature_cols)
    pca_k = config.gru.pca_components
    hidden_components = pca_k if pca_k > 0 else config.gru.hidden_size
    gru_feat_names = [
        f"gru_pc_{i}" if pca_k > 0 else f"gru_h{i}" for i in range(hidden_components)
    ]
    y_train = train_aligned["label"].to_numpy().astype(np.int32)
    y_test = test_aligned["label"].to_numpy().astype(np.int32)
    val_split_idx = max(1, int(len(train_aligned) * _VALIDATION_SPLIT_FRACTION))
    pipeline_fit_df = train_aligned.slice(0, len(train_aligned) - val_split_idx)
    pipeline_fit_y = y_train[:-val_split_idx]
    static_pipeline, selected_static_cols = fit_static_feature_pipeline(
        config,
        pipeline_fit_df,
        static_cols,
        pipeline_fit_y,
    )
    X_train_static = static_pipeline.transform(
        train_aligned.select(static_cols).to_pandas()
    )
    X_test_static = static_pipeline.transform(
        test_aligned.select(static_cols).to_pandas()
    )
    all_feature_cols = gru_feat_names + selected_static_cols
    X_train = np.concatenate([train_hidden, X_train_static], axis=1)
    X_test = np.concatenate([test_hidden, X_test_static], axis=1)
    reg_y_train: np.ndarray | None = None
    if is_regression:
        reg_y_train = train_aligned["regression_target"].to_numpy().astype(np.float64)

    # ── Diagnostics & weights ──
    _log_gru_signal_quality(train_hidden, y_train, config)
    diag = _window_diagnostics(w_idx + 1, train_aligned, test_aligned, y_train, y_test)
    train_weights = (
        train_aligned["sample_weight"].to_numpy().astype(np.float64)
        if "sample_weight" in train_aligned.columns
        else None
    )

    # ── Train LightGBM ──
    X_tr = X_train[:-val_split_idx]
    w_tr = train_weights[:-val_split_idx] if train_weights is not None else None
    X_val = X_train[-val_split_idx:]
    shift_ratios: dict[str, float] | None = None
    if is_regression:
        y_tr = reg_y_train[:-val_split_idx]  # type: ignore[index]
        y_val = reg_y_train[-val_split_idx:]  # type: ignore[index]
        class_weights = None
        combined_weights = w_tr
    else:
        y_tr = y_train[:-val_split_idx]
        y_val = y_train[-val_split_idx:]
        class_weights = _compute_class_weights(y_tr)
        shift_weights, shift_ratios = _compute_distribution_shift_weights(y_tr, y_val)
        # Combine with existing sample weights (average-uniqueness) if present
        if w_tr is not None:
            combined_weights = w_tr * shift_weights
        else:
            combined_weights = shift_weights

    # ── Attach weight diagnostics to window diag ──
    diag["class_weights"] = (
        {str(k): v for k, v in class_weights.items()} if class_weights else None
    )
    diag["shift_weights_per_class"] = shift_ratios

    model = _train_fixed(
        X_tr,
        y_tr,
        X_val,
        y_val,
        class_weights,
        config,
        all_feature_cols,
        sample_weight=combined_weights,
    )

    # ── Predict ──
    preds, aligned_proba, proba, raw_preds = _wf_format_predictions(
        model, X_test, all_feature_cols, is_regression
    )
    _add_prediction_diagnostics(diag, preds, y_test, aligned_proba)
    acc = (preds == y_test).mean()
    logger.info(
        "Window %d: accuracy=%.4f, test_samples=%d", w_idx + 1, acc, len(y_test)
    )

    return {
        **gru_state,
        "test_aligned": test_aligned,
        "y_test": y_test,
        "preds": preds,
        "raw_preds": raw_preds,
        "proba": proba,
        "model": model,
        "all_feature_cols": all_feature_cols,
        "accuracy": float(acc),
        "diag": diag,
        "classes": model.classes_,
    }


def _run_hybrid_window(
    config: Config,
    w_idx: int,
    window: Any,
    df: pl.DataFrame,
    feature_cols: list[str],
    is_regression: bool,
) -> dict[str, Any] | None:
    """Run a single hybrid walk-forward window: GRU → PCA → LGBM → predict.

    Delegates to ``_wf_gru_phase`` and ``_wf_build_predict_phase``.
    """
    gru_state = _wf_gru_phase(config, w_idx, window, df)
    if gru_state is None:
        return None
    return _wf_build_predict_phase(
        config, w_idx, gru_state, feature_cols, is_regression
    )


def _collect_oof_predictions(
    result: dict[str, Any], is_regression: bool
) -> pl.DataFrame:
    """Build OOF prediction chunk from a single window result.

    Args:
        result: Window result dict containing ``test_aligned`` DataFrame,
            ``y_test`` array, ``preds`` array, and (for classification)
            ``proba`` matrix with ``classes`` ordering.
        is_regression: Whether the model is a regressor.

    Returns:
        Polars DataFrame with ``timestamp``, ``true_label``,
        ``pred_label``, and probability columns (classification) or
        ``pred_raw`` (regression).
    """
    test_aligned: pl.DataFrame = result["test_aligned"]
    y_test: np.ndarray = result["y_test"]
    preds: np.ndarray = result["preds"]
    if is_regression:
        preds_int = preds.astype(np.int32)
        return pl.DataFrame(
            {
                "timestamp": test_aligned["timestamp"],
                "true_label": y_test,
                "pred_label": preds_int,
                "pred_raw": result["raw_preds"].astype(np.float64),
                **_one_hot_proba_columns(preds_int),
            }
        )
    return pl.DataFrame(
        {
            "timestamp": test_aligned["timestamp"],
            "true_label": y_test,
            "pred_label": preds.astype(np.int32),
            **_probability_columns(result["proba"], result["classes"]),
        }
    )


def train_hybrid_walk_forward(config: Config) -> None:
    """Train hybrid (GRU → LightGBM) with walk-forward validation.

    Orchestration: load → windows → loop(GRU→PCA→LGBM→collect) → save.
    Each step delegates to a focused helper ≤ 80 lines.
    """
    # 1. Load labeled data, compute regression target, generate windows
    df, windows, feature_cols, is_regression = _prepare_wf_data(config)

    # 2. Initialize loop state
    all_oof_preds: list[pl.DataFrame] = []
    gru_model = None
    gru_mean = None
    gru_std = None
    last_lgbm_model = None
    last_feature_cols: list[str] = []
    last_window_accuracy: float | None = None
    last_window_index = 0
    last_gru_history: list[dict] = []
    window_diagnostics: list[dict[str, Any]] = []
    stage_start = time.perf_counter()

    # 3. Process each walk-forward window
    for w_idx, window in enumerate(windows):
        window_start = time.perf_counter()
        console.rule(
            f"[bold cyan]Walk-forward window {w_idx + 1}/{len(windows)}[/]",
            style="cyan",
        )
        logger.info(
            "=== Window %d/%d: train=[%d:%d] test=[%d:%d] ===",
            w_idx + 1,
            len(windows),
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )

        result = _run_hybrid_window(
            config, w_idx, window, df, feature_cols, is_regression
        )
        if result is None:
            continue

        # Collect OOF predictions
        all_oof_preds.append(_collect_oof_predictions(result, is_regression))
        window_diagnostics.append(result["diag"])

        # Update deployable state (latest chronological window)
        gru_model = result["gru_model"]
        gru_mean = result["gru_mean"]
        gru_std = result["gru_std"]
        last_lgbm_model = result["model"]
        last_feature_cols = result["all_feature_cols"]
        last_window_accuracy = result["accuracy"]
        last_window_index = w_idx + 1
        last_gru_history = result["gru_history"]

        logger.info(
            "Window %d done (%.1fs)", w_idx + 1, time.perf_counter() - window_start
        )

    # 4. Validate and persist all artifacts
    _save_wf_artifacts(
        config,
        all_oof_preds,
        gru_model,
        gru_mean,
        gru_std,
        last_lgbm_model,
        last_feature_cols,
        last_window_accuracy,
        last_window_index,
        last_gru_history,
        windows,
        window_diagnostics,
        stage_start,
        is_regression,
    )
