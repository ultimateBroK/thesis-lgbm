"""Pipeline orchestration for the sequential thesis workflow.

Runs data preparation, feature engineering, labeling, model training,
optional backtesting, and report generation in order.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from thesis.shared.config import Config
from thesis.shared.ui import console, stage_header, stage_skip
from thesis.stage_1_data import generate_data
from thesis.stage_2_features import generate_features
from thesis.stage_3_labels import generate_labels
from thesis.stage_4_training.walk_forward import train_lgbm_fixed, train_walk_forward
from thesis.stage_5_backtest import run_backtest
from thesis.stage_6_reporting import generate_report

logger = logging.getLogger("thesis.pipeline")


# Cache fingerprinting

# Config sections whose values affect each stage's output.
# Used by _cache_hash to fingerprint the inputs.
_STAGE_CONFIG_SECTIONS: dict[int, list[str]] = {
    1: ["data"],
    2: ["features"],
    3: ["labels"],
    4: ["model", "validation"],
    5: ["backtest", "labels"],
    6: [],
}


def _cache_hash(config: Config, stage_num: int) -> str:
    """Compute an 8-char SHA-256 fingerprint of config sections relevant to a stage.

    Args:
        config: Application configuration.
        stage_num: Pipeline stage number (1–6).

    Returns:
        Hex digest string, or empty string if no sections are mapped.
    """
    sections = _STAGE_CONFIG_SECTIONS.get(stage_num, [])
    if not sections:
        return ""

    payload: dict[str, Any] = {}
    for name in sections:
        section_cfg = getattr(config, name, None)
        if section_cfg is not None:
            payload[name] = asdict(section_cfg)

    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def _resolve_cache_path(
    base: str | Path | None,
    invalidation: str,
    config: Config,
    stage_num: int,
) -> Path | None:
    """Resolve the effective cache check path based on invalidation strategy.

    Args:
        base: Raw cache path (e.g. ``data/processed/features.parquet``).
        invalidation: One of ``"path"``, ``"hash"``, ``"none"``.
        config: Application configuration.
        stage_num: Pipeline stage number.

    Returns:
        The resolved ``Path`` to use for cache existence checks, or
        ``None`` when caching is disabled.
    """
    if base is None or invalidation == "none":
        return None

    p = Path(base)
    if invalidation == "hash":
        h = _cache_hash(config, stage_num)
        if h:
            return p.with_stem(f"{p.stem}_{h}")
        return p

    return p


# Stage runner with cache checking


def _run_stage(
    stage_num: int,
    config: Config,
    flag_name: str,
    cache_path: str | Path | None,
    work_fn: callable,
) -> None:
    """Execute a pipeline stage with cache checking.

    Checks the workflow flag and optional cache file; skips the stage
    if disabled or cached unless ``force_rerun`` is set.

    Args:
        stage_num: Stage number for console display.
        config: Application configuration.
        flag_name: Workflow boolean flag name on ``config.workflow``.
        cache_path: Path to the cached output file, or ``None`` for
            no cache check.
        work_fn: Callable ``(Config) -> None`` that performs the
            actual stage work.
    """
    flag = getattr(config.workflow, flag_name, False)
    if not flag:
        stage_skip(stage_num, "disabled")
        return

    effective = _resolve_cache_path(
        cache_path,
        config.workflow.cache_invalidation,
        config,
        stage_num,
    )

    if effective is not None and not config.workflow.force_rerun and effective.exists():
        stage_skip(stage_num, f"cached ({effective.name})")
        return

    stage_header(stage_num)
    work_fn(config)

    # Create cache marker so subsequent runs with the same config skip.
    if effective is not None and not effective.exists():
        effective.touch()


def _run_backtest_with_barrier_guard(config: Config) -> None:
    """Run backtest only when label and execution ATR barriers match."""
    barrier = config.labels.barrier_atr_multiplier
    backtest_tp = config.backtest.atr_tp_multiplier
    backtest_sl = config.backtest.atr_stop_multiplier
    if barrier != backtest_tp or barrier != backtest_sl:
        raise ValueError(
            "Label/Backtest ATR barrier mismatch: "
            f"labels.barrier_atr_multiplier={barrier} != "
            f"backtest(tp={backtest_tp}, sl={backtest_sl}). "
            "Expected matching multipliers so training target and "
            "execution exits measure the same event."
        )
    run_backtest(config)


# Main pipeline


def run_pipeline(config: Config) -> None:
    """Execute the full thesis pipeline.

    Runs all six stages in order:

    1. Data preparation — download and cache OHLCV bars.
    2. Feature engineering — compute technical indicators.
    3. Triple-barrier labeling — generate directional labels.
    4. Model training — walk-forward LightGBM training.
    5. Backtesting — optional simulation of model signals.
    6. Report generation — write Markdown and HTML artefacts.

    Args:
        config: Loaded application configuration.
    """
    # Stage 1: Prepare OHLCV from raw ticks
    _run_stage(1, config, "run_data_pipeline", config.paths.ohlcv, generate_data)

    # Stage 2: Features
    _run_stage(
        2,
        config,
        "run_feature_engineering",
        config.paths.features,
        generate_features,
    )

    # Stage 3: Labels
    _run_stage(3, config, "run_label_generation", config.paths.labels, generate_labels)

    # Stage 4: Training (walk-forward or static)
    if config.validation.method == "sliding":
        stage_header(4)
        logger.info(
            "Using walk-forward sliding window validation (%s architecture)",
            config.model.architecture,
        )
        if config.workflow.run_model_training:
            train_walk_forward(config)
        else:
            stage_skip(4, "disabled")
    else:
        logger.info("Using fixed train/val/test split")
        _run_stage(4, config, "run_model_training", None, train_lgbm_fixed)

    # Stage 5: Backtest (Optional Application Demo)
    _run_stage(
        5,
        config,
        "run_backtest",
        None,
        _run_backtest_with_barrier_guard,
    )

    # Stage 6: Report
    _run_stage(
        6,
        config,
        "run_reporting",
        None,
        generate_report,
    )

    console.print()
    console.rule("[bold green]Pipeline Complete[/]")
    console.print()
