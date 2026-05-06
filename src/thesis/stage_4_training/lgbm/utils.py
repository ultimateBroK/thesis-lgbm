"""LightGBM utilities, training, and hybrid feature-matrix helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import polars as pl
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from thesis.shared.config import Config
from thesis.shared.constants import (
    DIST_SHIFT_CLIP_MAX,
    DIST_SHIFT_CLIP_MIN,
)
from thesis.shared.ui import console

logger = logging.getLogger("thesis.model")


# ── LightGBM utilities ──────────────────────────────────────────────


def _wrap_np(X: np.ndarray, feature_cols: list[str]) -> Any:
    """Wrap a NumPy matrix as a pandas DataFrame.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        feature_cols: Feature names aligned to matrix columns.

    Returns:
        A pandas DataFrame preserving feature names for LightGBM.
    """
    import pandas as pd

    return pd.DataFrame(X, columns=feature_cols)


def _build_interaction_constraints(feature_cols: list[str]) -> list[list[int]]:
    """Interaction constraints for LightGBM feature groups.

    Currently disabled — returning empty list allows full cross-group
    interaction between GRU hidden states and static price-action features.
    This lets LightGBM discover the most informative feature combinations
    without artificial restrictions.
    """
    return []


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights for multiclass labels.

    Args:
        y: Target labels.

    Returns:
        Mapping from class label to balanced class weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _compute_distribution_shift_weights(
    y_train: np.ndarray,
    y_val: np.ndarray,
    clip_range: tuple[float, float] = (DIST_SHIFT_CLIP_MIN, DIST_SHIFT_CLIP_MAX),
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute per-sample training weights to reduce stale-regime bias.

    Compares class frequencies between the training head and its internal
    validation tail.  Classes that have become *more* common in the recent
    tail relative to the full training window are up-weighted so the model
    pays more attention to emerging regimes.  Classes that are fading
    receive lower weight, reducing the influence of stale patterns.

    Time-safe: only training-window labels are used — no future/test labels
    are ever consulted.

    Args:
        y_train: Training-head labels in ``{-1, 0, 1}``.
        y_val: Training-tail (internal validation) labels in ``{-1, 0, 1}``.
        clip_range: Min/max bounds for per-class weight ratios.

    Returns:
        ``(sample_weights, ratio_dict)`` — Per-sample weight array aligned
        to ``y_train`` (mean ≈ 1.0) and per-class shift-weight ratio dict
        with string keys ``{"-1", "0", "1"}`` for JSON serialization.
    """
    classes = np.array([-1, 0, 1])
    train_counts = np.array([np.sum(y_train == c) for c in classes], dtype=np.float64)
    val_counts = np.array([np.sum(y_val == c) for c in classes], dtype=np.float64)

    train_freq = train_counts / train_counts.sum()
    val_freq = val_counts / val_counts.sum()

    # Ratio = val_freq / train_freq:
    #   > 1.0 → class is MORE common in recent data → up-weight
    #   < 1.0 → class is LESS common in recent data → down-weight
    # Avoid division by zero — classes absent from train get clip min.
    train_freq_safe = np.where(train_freq > 0, train_freq, 1e-8)
    ratios = val_freq / train_freq_safe
    ratios = np.clip(ratios, clip_range[0], clip_range[1])

    # Map per-class weight to per-sample (training head)
    weight_map = {int(c): float(r) for c, r in zip(classes, ratios)}
    sample_weights = np.array([weight_map[int(y)] for y in y_train], dtype=np.float64)

    # Build ratio dict with string keys for JSON-friendly diagnostics
    ratio_dict: dict[str, float] = {
        str(int(c)): float(r) for c, r in zip(classes, ratios)
    }

    logger.info(
        "Distribution-shift weights: SHORT=%d→%.2f HOLD=%d→%.2f LONG=%d→%.2f "
        "(train freq: [%.1f%%, %.1f%%, %.1f%%] val freq: [%.1f%%, %.1f%%, %.1f%%]) "
        "min=%.3f median=%.3f max=%.3f mean=%.3f",
        int(train_counts[0]),
        ratios[0],
        int(train_counts[1]),
        ratios[1],
        int(train_counts[2]),
        ratios[2],
        train_freq[0] * 100,
        train_freq[1] * 100,
        train_freq[2] * 100,
        val_freq[0] * 100,
        val_freq[1] * 100,
        val_freq[2] * 100,
        float(np.min(sample_weights)),
        float(np.median(sample_weights)),
        float(np.max(sample_weights)),
        float(np.mean(sample_weights)),
    )

    return sample_weights, ratio_dict


def _filter_validation_to_seen_classes(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> tuple[Any, np.ndarray] | None:
    """Drop validation rows whose class is absent from the training fold.

    LightGBM's sklearn wrapper label-encodes classes from ``y_train`` and
    cannot transform an ``eval_set`` containing unseen labels. Small
    walk-forward folds can miss the rare Hold class, so validation is
    filtered to classes actually learnable in that fold.

    Returns ``None`` when the validation set has **no** overlapping classes
    with training — early stopping should be skipped for that fold.

    Returns:
        ``(X_val_filtered, y_val_filtered)`` or ``None``.
    """
    seen = np.unique(y_train)
    mask = np.isin(y_val, seen)
    if not mask.all():
        logger.warning(
            "LightGBM validation contains %d row(s) from unseen train classes %s; "
            "dropping them from early-stopping eval_set",
            int((~mask).sum()),
            sorted(set(map(int, y_val[~mask]))),
        )
    if not mask.any():
        logger.warning(
            "Validation set has no overlapping classes with training "
            "— skipping early stopping"
        )
        return None
    return _wrap_np(X_val[mask], feature_cols), y_val[mask]


# ── LightGBM training — fixed hyperparameters ───────────────────────


def _train_fixed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float] | None,
    config: Config,
    feature_cols: list[str],
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Train LightGBM with fixed hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (multiclass) or continuous targets (regression).
        X_val: Validation feature matrix.
        y_val: Validation labels or targets.
        class_weights: Balanced class weights (None for regression).
        config: Resolved application configuration.
        feature_cols: Ordered feature names.
        sample_weight: Optional per-row training weights.

    Returns:
        Fitted LightGBM model (classifier or regressor).
    """
    import lightgbm as lgb

    m = config.model
    is_regression = m.objective == "regression"
    constraints = _build_interaction_constraints(feature_cols)
    gru_feature_count = sum(1 for name in feature_cols if name.startswith("gru_h"))
    static_feature_count = len(feature_cols) - gru_feature_count
    logger.info(
        "LightGBM: %s leaves=%d depth=%d lr=%.4f"
        " n_est=%d constraints=[%d GRU, %d static]",
        "regressor" if is_regression else "classifier",
        m.num_leaves,
        m.max_depth,
        m.learning_rate,
        m.n_estimators,
        gru_feature_count,
        static_feature_count,
    )

    start_time = time.perf_counter()

    if is_regression:
        model = lgb.LGBMRegressor(
            num_leaves=m.num_leaves,
            max_depth=m.max_depth,
            learning_rate=m.learning_rate,
            n_estimators=m.n_estimators,
            min_child_samples=m.min_child_samples,
            subsample=m.subsample,
            subsample_freq=m.subsample_freq,
            colsample_bytree=m.feature_fraction,
            reg_alpha=m.reg_alpha,
            reg_lambda=m.reg_lambda,
            objective="regression",
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
        )
    else:
        model = lgb.LGBMClassifier(
            num_leaves=m.num_leaves,
            max_depth=m.max_depth,
            learning_rate=m.learning_rate,
            n_estimators=m.n_estimators,
            min_child_samples=m.min_child_samples,
            subsample=m.subsample,
            subsample_freq=m.subsample_freq,
            colsample_bytree=m.feature_fraction,
            reg_alpha=m.reg_alpha,
            reg_lambda=m.reg_lambda,
            interaction_constraints=constraints,
            class_weight=class_weights,
            objective="multiclass",
            num_class=3,
            random_state=config.workflow.random_seed,
            n_jobs=config.workflow.n_jobs,
            verbose=-1,
            use_missing=False,
            zero_as_missing=False,
        )

    # Rich progress bar over boosting iterations
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]LightGBM boosting"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    )

    with progress:
        task = progress.add_task("iter", total=m.n_estimators, v_loss=0.0)

        if is_regression:
            filtered = _wrap_np(X_val, feature_cols), y_val
        else:
            filtered = _filter_validation_to_seen_classes(
                X_train, X_val, y_val, y_train, feature_cols
            )

        def _progress_cb(env: Any) -> None:
            """Update progress bar from LightGBM callback state.

            Args:
                env: LightGBM callback environment.
            """
            progress.update(
                task,
                advance=1,
                v_loss=env.evaluation_result_list[0][2]
                if env.evaluation_result_list
                else 0.0,
            )

        if filtered is None:
            logger.warning(
                "Validation set has no overlapping classes with training "
                "— skipping early stopping"
            )
            model.fit(
                _wrap_np(X_train, feature_cols),
                y_train,
                sample_weight=sample_weight,
            )
        else:
            X_val_df, y_val_eval = filtered
            model.fit(
                _wrap_np(X_train, feature_cols),
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val_df, y_val_eval)],
                callbacks=[
                    lgb.early_stopping(m.early_stopping_rounds, verbose=False),
                    _progress_cb,
                ],
            )

    train_time = time.perf_counter() - start_time
    logger.info(
        "LightGBM done: best_iter=%d (%.1fs)",
        model.best_iteration_,
        train_time,
    )
    return model


def _save_feature_importance(
    model: Any, feature_cols: list[str], config: Config
) -> None:
    """Save sorted model feature importances to JSON.

    Args:
        model: Fitted model exposing ``feature_importances_``.
        feature_cols: Ordered feature names.
        config: Resolved application configuration.
    """
    try:
        imp = model.feature_importances_
        pairs = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
        if config.paths.session_dir:
            out_path = (
                Path(config.paths.session_dir) / "reports" / "feature_importance.json"
            )
        else:
            out_path = Path("results/feature_importance.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({name: float(val) for name, val in pairs}, f, indent=2)
        logger.info(
            "Feature importance saved (top 5: %s)",
            [p[0] for p in pairs[:5]],
        )
    except (OSError, ValueError) as e:
        logger.warning("Feature importance save failed: %s", e)


# ── Hybrid matrix helpers ───────────────────────────────────────────


def _align_splits_with_sequences(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_hidden: np.ndarray,
    val_hidden: np.ndarray,
    test_hidden: np.ndarray,
    seq_len: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Align DataFrames with GRU sequence outputs.

    Args:
        train_df: Full training DataFrame.
        val_df: Full validation DataFrame.
        test_df: Full test DataFrame.
        train_hidden: GRU hidden states for training.
        val_hidden: GRU hidden states for validation.
        test_hidden: GRU hidden states for test.
        seq_len: GRU sequence length.

    Returns:
        Tuple of (train_aligned, val_aligned, test_aligned) DataFrames.
    """
    train_aligned = train_df.slice(seq_len - 1, len(train_hidden))
    val_aligned = val_df.slice(seq_len - 1, len(val_hidden))
    test_aligned = test_df.slice(seq_len - 1, len(test_hidden))
    logger.info(
        "Aligned: train=%d val=%d test=%d",
        len(train_aligned),
        len(val_aligned),
        len(test_aligned),
    )
    return train_aligned, val_aligned, test_aligned


def _build_hybrid_matrix(
    train_hidden: np.ndarray,
    val_hidden: np.ndarray,
    test_hidden: np.ndarray,
    train_aligned: pl.DataFrame,
    val_aligned: pl.DataFrame,
    test_aligned: pl.DataFrame,
    static_cols: list[str],
    hidden_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build hybrid feature matrices combining GRU hidden states with static features.

    Args:
        train_hidden: GRU hidden states for training.
        val_hidden: GRU hidden states for validation.
        test_hidden: GRU hidden states for test.
        train_aligned: Aligned training DataFrame.
        val_aligned: Aligned validation DataFrame.
        test_aligned: Aligned test DataFrame.
        static_cols: List of static feature column names.
        hidden_size: GRU hidden size (number of GRU features).

    Returns:
        Tuple of (X_train, X_val, X_test, feature_names).
    """
    gru_feat_names = [f"gru_h{i}" for i in range(hidden_size)]
    all_feature_cols = gru_feat_names + static_cols

    X_train = np.concatenate(
        [train_hidden, train_aligned.select(static_cols).to_numpy()], axis=1
    )
    X_val = np.concatenate(
        [val_hidden, val_aligned.select(static_cols).to_numpy()], axis=1
    )
    X_test = np.concatenate(
        [test_hidden, test_aligned.select(static_cols).to_numpy()], axis=1
    )
    return X_train, X_val, X_test, all_feature_cols
