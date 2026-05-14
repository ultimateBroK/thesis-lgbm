"""Generic walk-forward loop. Used by both LGBM and stacking."""

from __future__ import annotations

from collections.abc import Callable
import logging
import time
from typing import Any

from thesis.shared.config import Config
from thesis.shared.ui import console

logger = logging.getLogger("thesis")


def run_walk_forward(
    config: Config,
    *,
    prepare_fn: Callable[[Config], tuple[Any, list[Any], list[str], dict[str, Any]]],
    window_fn: Callable[..., dict[str, Any] | None],
    save_fn: Callable[[Config, list[dict[str, Any]], list[Any], float], None],
) -> None:
    """Execute the walk-forward loop.

    Args:
        config: Application config.
        prepare_fn: Loads data, returns (df, windows, feature_cols, extra).
        window_fn: Trains one window, returns result or None to skip.
        save_fn: Persists all results after loop finishes.
    """
    t0 = time.perf_counter()
    df, windows, feature_cols, extra_data = prepare_fn(config)
    logger.info(
        "Walk-forward: %d windows, %d features", len(windows), len(feature_cols)
    )

    results: list[dict[str, Any]] = []
    for w_idx, window in enumerate(windows):
        wt = time.perf_counter()
        console.rule(f"[bold cyan]Window {w_idx + 1}/{len(windows)}[/]")
        logger.info(
            "  [%d:%d] train | [%d:%d] test",
            window.train_start_idx,
            window.train_end_idx,
            window.test_start_idx,
            window.test_end_idx,
        )
        result = window_fn(config, w_idx, window, df, feature_cols, **extra_data)
        if result is not None:
            results.append(result)
        logger.info("  Window %d done (%.1fs)", w_idx + 1, time.perf_counter() - wt)

    logger.info(
        "Walk-forward: %d/%d windows produced results (%.1fs total)",
        len(results),
        len(windows),
        time.perf_counter() - t0,
    )
    save_fn(config, results, windows, time.perf_counter() - t0)
