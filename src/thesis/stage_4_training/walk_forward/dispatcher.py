"""Walk-forward training dispatcher — routes model architecture variants."""

from __future__ import annotations

import logging

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.gru import train_gru_walk_forward
from thesis.stage_4_training.walk_forward.hybrid import train_hybrid_walk_forward
from thesis.stage_4_training.walk_forward.lgbm import train_lgbm_walk_forward

logger = logging.getLogger("thesis.pipeline")


def train_walk_forward(config: Config) -> None:
    """Dispatch walk-forward training to the configured architecture.

    Args:
        config: Application configuration. Reads ``model.architecture``
            to route to LightGBM-only, GRU-only, or hybrid workflows.

    Raises:
        ValueError: If ``model.architecture`` is not one of
            ``'hybrid'``, ``'lgbm'``, or ``'gru'``.
    """
    architecture = config.model.architecture

    if architecture == "lgbm":
        logger.info("Using LightGBM walk-forward pipeline")
        train_lgbm_walk_forward(
            config, expanded_features=config.model.lgbm_expanded_features
        )
        return

    if architecture == "gru":
        logger.info("Using GRU walk-forward pipeline")
        train_gru_walk_forward(config)
        return

    if architecture != "hybrid":
        raise ValueError(
            f"Unsupported model.architecture: {architecture!r}. "
            "Must be one of: 'hybrid', 'lgbm', 'gru'"
        )

    logger.info("Using hybrid walk-forward pipeline")
    train_hybrid_walk_forward(config)
