"""Hybrid GRU + LightGBM model — training, tuning, and interpretation.

Merged from ``hybrid/train.py``, ``hybrid/lgbm.py``, and ``hybrid/interpret.py``
into a single module for walk-forward pipeline integration.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from thesis.config import Config
from thesis.constants import EXCLUDE_COLS
from thesis.gru import (
    extract_hidden_states,
    prepare_sequences,
    save_gru_model,
    train_gru,
)
from thesis.ui import console

logger = logging.getLogger("thesis.model")

# ---------------------------------------------------------------------------
# LightGBM utilities
# ---------------------------------------------------------------------------


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
    clip_range: tuple[float, float] = (0.5, 3.0),
) -> np.ndarray:
    """Compute per-sample weights for validation to correct distribution shift.

    Compares class frequencies between training and validation sets.
    Classes under-represented in validation relative to training are
    up-weighted so the validation metric better reflects the training
    distribution's priorities.

    Args:
        y_train: Training labels in ``{-1, 0, 1}``.
        y_val: Validation labels in ``{-1, 0, 1}``.
        clip_range: Min/max bounds for per-class weight ratios.

    Returns:
        Per-sample weight array aligned to ``y_val``.
    """
    classes = np.array([-1, 0, 1])
    train_counts = np.array([np.sum(y_train == c) for c in classes], dtype=np.float64)
    val_counts = np.array([np.sum(y_val == c) for c in classes], dtype=np.float64)

    train_freq = train_counts / train_counts.sum()
    val_freq = val_counts / val_counts.sum()

    # Avoid division by zero — classes absent from val get max weight
    val_freq_safe = np.where(val_freq > 0, val_freq, 1e-8)
    ratios = train_freq / val_freq_safe
    ratios = np.clip(ratios, clip_range[0], clip_range[1])

    # Map per-class weight to per-sample
    weight_map = {int(c): float(r) for c, r in zip(classes, ratios)}
    _sample_weights = np.array([weight_map[int(y)] for y in y_val], dtype=np.float64)

    logger.info(
        "Distribution-shift weights: SHORT=%d→%.2f HOLD=%d→%.2f LONG=%d→%.2f "
        "(train freq: [%.1f%%, %.1f%%, %.1f%%] val freq: [%.1f%%, %.1f%%, %.1f%%])",
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
    )

    # Disabled: distribution shift weights caused regression.
    # Keep logging above for future diagnostics; return uniform weights.
    logger.info(
        "Distribution-shift weights DISABLED — returning uniform weights (all 1.0)"
    )
    return np.ones(len(y_val))


# ---------------------------------------------------------------------------
# LightGBM training — fixed hyperparameters
# ---------------------------------------------------------------------------


def _train_fixed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    config: Config,
    feature_cols: list[str],
) -> Any:
    """Train LightGBM with fixed hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        class_weights: Balanced class weights.
        config: Resolved application configuration.
        feature_cols: Ordered feature names.

    Returns:
        Fitted ``lightgbm.LGBMClassifier`` model.
    """
    import lightgbm as lgb

    m = config.model
    constraints = _build_interaction_constraints(feature_cols)
    gru_feature_count = sum(1 for name in feature_cols if name.startswith("gru_h"))
    static_feature_count = len(feature_cols) - gru_feature_count
    logger.info(
        "LightGBM: leaves=%d depth=%d lr=%.4f n_est=%d constraints=[%d GRU, %d static]",
        m.num_leaves,
        m.max_depth,
        m.learning_rate,
        m.n_estimators,
        gru_feature_count,
        static_feature_count,
    )

    start_time = time.perf_counter()

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
        zero_as_missing=True,
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
        transient=False,
    )

    with progress:
        task = progress.add_task("iter", total=m.n_estimators, v_loss=0.0)

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

        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            eval_set=[(_wrap_np(X_val, feature_cols), y_val)],
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


# ---------------------------------------------------------------------------
# LightGBM training — Optuna multi-objective tuning
# ---------------------------------------------------------------------------

# Default bars per year for H1 timeframe (approx 23h/day × 365 days)
_H1_BARS_PER_YEAR = 8400


def _convert_hard_to_proba(y_pred: np.ndarray) -> np.ndarray:
    """Convert 1D hard class predictions to 3-column probability matrix.

    Args:
        y_pred: 1D array of hard class labels (-1, 0, 1).

    Returns:
        2D array of shape (n, 3) with one-hot encoded probabilities.
    """
    n = len(y_pred)
    proba = np.zeros((n, 3), dtype=np.float64)
    for i, pred in enumerate(y_pred):
        if pred == -1:
            proba[i, 0] = 1.0
        elif pred == 0:
            proba[i, 1] = 1.0
        else:
            proba[i, 2] = 1.0
    return proba


def _apply_confidence_filter(
    y_pred: np.ndarray,
    max_probs: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """Apply confidence threshold by forcing low-confidence predictions to Hold.

    Args:
        y_pred: Predicted class indices.
        max_probs: Confidence values (max probability per sample).
        confidence_threshold: Minimum confidence to trade.

    Returns:
        Modified predictions where low-confidence are set to 0 (Hold).
    """
    y_pred = y_pred.copy()
    y_pred[max_probs < confidence_threshold] = 0
    return y_pred


def _compute_trade_returns(
    correct: np.ndarray,
    spread_cost: float,
) -> np.ndarray:
    """Compute cost-aware returns from correct/incorrect predictions.

    Args:
        correct: Boolean array indicating winning trades.
        spread_cost: Round-trip spread cost fraction.

    Returns:
        Array of returns per trade.
    """
    return np.where(correct, 1.0 - spread_cost, -1.0 - spread_cost)


def _compute_sharpe_from_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    confidence_threshold: float = 0.0,
    spread_cost: float = 0.0002,
    annualize: bool = False,
    bars_per_year: int = _H1_BARS_PER_YEAR,
    min_trades: int = 3,
) -> float:
    """Compute Sharpe ratio from predicted class probabilities.

    Uses a smooth penalty when trade count is below 10 to avoid zeroing
    the signal for conservative models while still penalizing too-few trades.

    Args:
        y_true: True labels in ``{-1, 0, 1}``.
        y_pred_proba: Class probabilities or hard class predictions.
        confidence_threshold: Minimum confidence required to trade.
        spread_cost: Round-trip transaction cost fraction.
        annualize: Whether to annualize Sharpe.
        bars_per_year: Bars used for annualization.
        min_trades: Absolute minimum trades before returning 0.

    Returns:
        Sharpe ratio (penalized for low trade count), or ``0.0`` when
        trades are below ``min_trades``.
    """
    if y_pred_proba.ndim == 1:
        y_pred_proba = _convert_hard_to_proba(y_pred_proba)

    max_probs = np.max(y_pred_proba, axis=1)
    pred_indices = np.argmax(y_pred_proba, axis=1)
    classes = np.array([-1, 0, 1])
    y_pred = classes[pred_indices]
    y_pred = _apply_confidence_filter(y_pred, max_probs, confidence_threshold)

    mask = y_pred != 0
    n_trades = int(mask.sum())

    if n_trades < min_trades:
        return 0.0

    correct = y_pred == y_true
    returns = _compute_trade_returns(correct, spread_cost)[mask]

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0.0

    if std_ret < 1e-10:
        return float(np.clip(mean_ret, -10.0, 10.0))

    sharpe = mean_ret / std_ret

    # Smooth penalty: scale down Sharpe when trade count is below 10
    # so Optuna can still differentiate between models, but strongly
    # prefers models that generate more trades.
    if n_trades < 10:
        penalty = n_trades / 10.0
        sharpe *= penalty

    # Clamp to prevent overflow in Optuna
    sharpe = float(np.clip(sharpe, -10.0, 10.0))

    # Annualize only when actual trade count is known (e.g., final backtest).
    # During Optuna we return unannualized Sharpe to avoid inflated estimates.
    if annualize and n_trades > 0:
        trades_per_year = min(bars_per_year, n_trades * 2)
        sharpe = sharpe * np.sqrt(trades_per_year)

    return float(sharpe)


def _compute_cost_fraction(bc: Any, dc: Any, median_price: float) -> float:
    """Compute round-trip cost fraction matching the backtest cost model.

    Combines spread, slippage, and commission into a single per-trade
    cost fraction relative to the notional value.

    Args:
        bc: BacktestConfig with spread/slippage/commission parameters.
        dc: DataConfig with tick_size and contract_size.

    Returns:
        Total round-trip cost as a fraction of notional value.
    """
    if not np.isfinite(median_price) or median_price <= 0:
        raise ValueError(
            f"Reference price must be positive and finite, got {median_price!r}"
        )

    spread_rate = (bc.spread_ticks + bc.slippage_ticks) * dc.tick_size / median_price
    commission_rate = bc.commission_per_lot / (
        bc.lots_per_trade * dc.contract_size * median_price
    )
    return spread_rate + commission_rate


def _train_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: dict[int, float],
    config: Config,
    feature_cols: list[str],
    reference_price: float,
) -> Any:
    """Tune and train LightGBM with multi-objective Optuna.

    Objectives (Pareto front):
      1. **Maximize** validation-set Sharpe ratio — trading profitability.
      2. **Minimize** multi-class logloss — calibration quality.

    Multi-objective optimisation prevents the model from achieving high
    Sharpe by being overly conservative (trading very rarely).  After all
    trials, the best trial is selected from the Pareto front by highest
    Sharpe; logloss breaks ties.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        class_weights: Balanced class weights.
        config: Resolved application configuration.
        feature_cols: Ordered feature names.
        reference_price: Representative market price for the validation slice.

    Returns:
        Fitted ``lightgbm.LGBMClassifier`` model with best Optuna params.
    """
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import log_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = config.workflow.random_seed

    # Feature interaction constraints: GRU features ↔ GRU features only,
    # static features ↔ static features only.
    constraints = _build_interaction_constraints(feature_cols)
    gru_feature_count = sum(1 for name in feature_cols if name.startswith("gru_h"))
    static_feature_count = len(feature_cols) - gru_feature_count
    logger.info(
        "Optuna interaction_constraints: [%d GRU features, %d static features]",
        gru_feature_count,
        static_feature_count,
    )

    # Cost model matching the full backtest
    bc = config.backtest
    dc = config.data
    cost_fraction = _compute_cost_fraction(bc, dc, reference_price)
    confidence_threshold = bc.confidence_threshold

    logger.info(
        "Optuna cost model: reference_price=%.4f spread+slip=%.6f commission=%.6f total=%.6f",
        reference_price,
        (bc.spread_ticks + bc.slippage_ticks) * dc.tick_size / reference_price,
        bc.commission_per_lot
        / (bc.lots_per_trade * dc.contract_size * reference_price),
        cost_fraction,
    )

    X_val_df = _wrap_np(X_val, feature_cols)
    val_sample_weights = _compute_distribution_shift_weights(y_train, y_val)

    def objective(trial: Any) -> tuple[float, float]:
        """Score a trial on Sharpe (maximize) and logloss (minimize).

        Args:
            trial: Optuna trial proposing hyperparameters.

        Returns:
            Tuple of ``(sharpe, logloss)`` for multi-objective optimisation.
        """
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "extra_trees": True,
            "interaction_constraints": constraints,
            "class_weight": class_weights,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": seed,
            "n_jobs": config.workflow.n_jobs,
            "verbose": -1,
            "use_missing": False,
            "zero_as_missing": True,
        }

        model = lgb.LGBMClassifier(**params)

        is_multi_objective = len(study.directions) > 1

        def _pruning_cb(env: Any) -> None:
            """Report intermediate validation logloss for MedianPruner.

            Skipped for multi-objective optimization — ``trial.report()``
            is not supported when there are multiple objectives.
            """
            if is_multi_objective:
                return
            if env.evaluation_result_list:
                val_loss = env.evaluation_result_list[0][2]
                trial.report(val_loss, step=env.iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            eval_set=[(X_val_df, y_val)],
            eval_sample_weight=[val_sample_weights],
            callbacks=[
                lgb.early_stopping(config.model.early_stopping_rounds, verbose=False),
                _pruning_cb,
            ],
        )

        preds_proba = model.predict_proba(X_val_df)

        sharpe = _compute_sharpe_from_predictions(
            y_val,
            preds_proba,
            confidence_threshold=confidence_threshold,
            spread_cost=cost_fraction,
        )

        # Map labels from {-1, 0, 1} to {0, 1, 2} for log_loss
        y_val_mapped = (y_val + 1).astype(int)
        ll = float(log_loss(y_val_mapped, preds_proba, labels=[0, 1, 2]))

        return sharpe, ll

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=20,
        interval_steps=5,
    )
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5),
        pruner=pruner,
    )

    n_trials = config.model.optuna_trials

    # Rich progress for Optuna trials
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]Optuna tuning (Sharpe + logloss)"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[green]best_sharpe={task.fields[best_sharpe]:.4f}"),
        TextColumn("•"),
        TextColumn("[cyan]best_ll={task.fields[best_ll]:.4f}"),
        TextColumn("•"),
        TextColumn("[dim]pruned={task.fields[pruned]}"),
        TimeElapsedColumn(),
        transient=False,
    )

    best_sharpe = -float("inf")
    best_ll = float("inf")
    pruned_count = 0

    with progress:
        task = progress.add_task(
            "trials", total=n_trials, best_sharpe=0.0, best_ll=0.0, pruned=0
        )

        def _optuna_cb(
            _study: optuna.study.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            """Update multi-objective progress tracking after each trial.

            Args:
                _study: Optuna study state.
                trial: Completed trial.
            """
            nonlocal best_sharpe, best_ll, pruned_count
            if trial.state == optuna.trial.TrialState.PRUNED:
                pruned_count += 1
            # Track the best Sharpe / logloss seen across all completed trials
            if trial.state == optuna.trial.TrialState.COMPLETE:
                t_sharpe, t_ll = trial.values  # type: ignore[misc]
                if t_sharpe > best_sharpe:
                    best_sharpe = t_sharpe
                if t_ll < best_ll:
                    best_ll = t_ll
            progress.update(
                task,
                advance=1,
                best_sharpe=best_sharpe,
                best_ll=best_ll,
                pruned=pruned_count,
            )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=config.model.optuna_timeout,
            callbacks=[_optuna_cb],
        )

    # --- Pareto front selection ---
    # Select from the Pareto front: highest Sharpe first, then lowest logloss.
    pareto_trials = study.best_trials
    selected = max(
        pareto_trials,
        key=lambda t: (t.values[0], -t.values[1]),
    )

    logger.info(
        "Optuna done: pareto_front=%d trials, selected=#%d "
        "sharpe=%.4f logloss=%.4f (%d pruned)",
        len(pareto_trials),
        selected.number,
        selected.values[0],
        selected.values[1],
        pruned_count,
    )

    # Final model with selected Pareto-optimal params
    best = selected.params
    model = lgb.LGBMClassifier(
        **best,
        interaction_constraints=constraints,
        class_weight=class_weights,
        objective="multiclass",
        num_class=3,
        random_state=seed,
        n_jobs=config.workflow.n_jobs,
        verbose=-1,
        use_missing=False,
        zero_as_missing=True,
    )

    n_est = best.get("n_estimators", 500)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Final LightGBM"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TimeElapsedColumn(),
        transient=False,
    )

    with progress:
        task = progress.add_task("iter", total=n_est, v_loss=0.0)

        def _progress_cb(env: Any) -> None:
            """Update final-model progress from callback state.

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

        model.fit(
            _wrap_np(X_train, feature_cols),
            y_train,
            eval_set=[(_wrap_np(X_val, feature_cols), y_val)],
            eval_sample_weight=[val_sample_weights],
            callbacks=[
                lgb.early_stopping(config.model.early_stopping_rounds, verbose=False),
                _progress_cb,
            ],
        )

    logger.info(f"Final model: best_iteration={model.best_iteration_}")
    return model


# ---------------------------------------------------------------------------
# SHAP analysis and feature importance
# ---------------------------------------------------------------------------


def _compute_shap(
    model: Any, X_test: np.ndarray, feature_cols: list[str], config: Config
) -> None:
    """Compute and persist SHAP artifacts for the test split.

    Computes SHAP values on a capped sample, renders a summary chart, and saves
    both image and JSON summaries under the report output directory.

    Args:
        model: Tree-based model compatible with ``shap.TreeExplainer``.
        X_test: Test feature matrix.
        feature_cols: Ordered feature names aligned with ``X_test`` columns.
        config: Resolved application configuration.
    """
    try:
        import shap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_samples = min(500, len(X_test))
        shap_start = time.perf_counter()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]SHAP analysis"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        )

        with progress:
            task = progress.add_task("steps", total=2)

            # Step 1: Compute SHAP values
            progress.update(task, description="[bold cyan]SHAP computing")
            explainer = shap.TreeExplainer(model)
            X_sample = _wrap_np(X_test[:n_samples], feature_cols)
            shap_values = explainer.shap_values(X_sample)
            progress.update(task, advance=1)

            # Step 2: Render plot
            progress.update(task, description="[bold cyan]SHAP rendering")

            # Ensure shap_values is a list of 2D arrays for multi-class classification
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_values = [
                    shap_values[:, :, i] for i in range(shap_values.shape[2])
                ]

            plt.figure(figsize=(10, 8))

            # Use bar plot for per-class feature importance with clear legend for 3 classes
            rng = np.random.default_rng(config.workflow.random_seed)
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_cols,
                plot_type="bar",
                class_names=["Short", "Hold", "Long"],  # Class indices 0, 1, 2
                show=False,
                rng=rng,
            )

            # Custom title for clarity
            plt.title("Feature Importance by Class (SHAP Values)", fontsize=14, pad=20)

            if config.paths.session_dir:
                out = Path(config.paths.session_dir) / "reports" / "shap_summary.png"
            else:
                out = Path("results/shap_summary.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()

            # Save SHAP values as JSON for interactive pyecharts rendering
            n_features = min(20, len(feature_cols))
            mean_abs_shap = []
            for sv in shap_values:
                mean_abs_shap.append(np.abs(sv).mean(axis=0).tolist()[:n_features])
            shap_json = {
                "features": feature_cols[:n_features],
                "class_names": ["Short", "Hold", "Long"],
                "mean_abs_shap": mean_abs_shap,
            }
            if config.paths.session_dir:
                json_out = (
                    Path(config.paths.session_dir) / "reports" / "shap_values.json"
                )
            else:
                json_out = Path("results/shap_values.json")
            with open(json_out, "w") as f:
                json.dump(shap_json, f, indent=2)

            progress.update(task, advance=1)

        shap_time = time.perf_counter() - shap_start
        logger.info("SHAP done: %d samples, %.1fs", n_samples, shap_time)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)


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
    except Exception as e:
        logger.warning("Feature importance save failed: %s", e)


# ---------------------------------------------------------------------------
# Hybrid matrix helpers
# ---------------------------------------------------------------------------


def _normalize_label(lbl: int) -> str:
    """Normalize a class label for probability column naming.

    Args:
        lbl: Integer class label.

    Returns:
        A string-safe label where negatives are prefixed with ``minus``.
    """
    if lbl < 0:
        return f"minus{abs(lbl)}"
    return str(lbl)


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


def _save_predictions(
    test_aligned: pl.DataFrame,
    y_test: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray,
    class_order: list,
    preds_path: Path,
) -> None:
    """Save predictions as Parquet and CSV files.

    Args:
        test_aligned: Aligned test DataFrame with timestamps.
        y_test: True labels.
        preds: Predicted labels.
        proba: Prediction probabilities.
        class_order: Class order mapping.
        preds_path: Destination path for Parquet file.
    """
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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

    # Load splits
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

    logger.info(
        "Features: %d total (%d GRU + %d static)",
        len(all_feature_cols),
        hidden_size,
        len(static_cols),
    )

    # --- 4. Train LightGBM ---
    method = "Optuna" if config.model.use_optuna else "Fixed"
    console.print(
        Panel(
            f"Stage 4.2: [bold]LightGBM[/] ({method})", style="magenta", padding=(0, 2)
        )
    )
    class_weights = _compute_class_weights(y_train)

    if config.model.use_optuna:
        reference_price = float(val_aligned["close"].median())
        model = _train_optuna(
            X_train,
            y_train,
            X_val,
            y_val,
            class_weights,
            config,
            all_feature_cols,
            reference_price,
        )
    else:
        model = _train_fixed(
            X_train, y_train, X_val, y_val, class_weights, config, all_feature_cols
        )

    # Save LightGBM model
    model_path = Path(config.paths.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model: %s", model_path)

    # Save training history
    models_dir = model_path.parent
    history_path = models_dir / "training_history.json"
    lgbm_info: dict = {
        "best_iteration": int(model.best_iteration_)
        if hasattr(model, "best_iteration_")
        else None,
        "n_features": len(all_feature_cols),
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

    class_order = model.classes_.tolist()
    preds_path = Path(config.paths.predictions)
    _save_predictions(test_aligned, y_test, preds, proba, class_order, preds_path)

    # --- 6. SHAP ---
    console.print(
        Panel(
            "Stage 4.4: [bold]SHAP Feature Importance[/]",
            style="magenta",
            padding=(0, 2),
        )
    )
    _compute_shap(model, X_test, all_feature_cols, config)

    # --- 7. Feature importance ---
    _save_feature_importance(model, all_feature_cols, config)

    # Final summary panel
    stage_time = time.perf_counter() - stage_start
    console.print(
        Panel(
            f"[bold green]Stage 4 complete[/]\n"
            f"  Accuracy: [bold]{acc:.4f}[/]\n"
            f"  GRU: {hidden_size} features ({config.gru.num_layers} layers)\n"
            f"  LightGBM: {len(all_feature_cols)} features, best_iter={getattr(model, 'best_iteration_', 'N/A')}\n"
            f"  Time: {stage_time:.1f}s",
            style="green",
            padding=(0, 2),
        )
    )
