"""Walk-forward training dispatcher — routes to hybrid or static."""

from __future__ import annotations

import logging

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.hybrid import _run_walk_forward_hybrid
from thesis.stage_4_training.walk_forward.static import _run_walk_forward_static

logger = logging.getLogger("thesis.pipeline")


def _run_walk_forward(config: Config) -> None:
    """Dispatch walk-forward training to the configured architecture.

    Args:
        config: Application configuration. Reads ``model.architecture``
            to route to ``_run_walk_forward_static`` or
            ``_run_walk_forward_hybrid``.

    Raises:
        ValueError: If ``model.architecture`` is unsupported.
    """
    architecture = config.model.architecture

    if architecture == "static":
        logger.info("Using static-feature-only walk-forward baseline")
        _run_walk_forward_static(config, expanded_features=config.model.static_expanded)
        return

    if architecture != "hybrid":
        raise ValueError(f"Unsupported model.architecture: {architecture!r}")

    logger.info("Using hybrid walk-forward pipeline")
    _run_walk_forward_hybrid(config)
