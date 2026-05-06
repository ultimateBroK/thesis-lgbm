"""GRU model persistence — save, load, and rebuild checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from thesis.shared.config import Config
from thesis.stage_4_training.gru.arch import GRUExtractor

logger = logging.getLogger("thesis.gru")


def save_gru_model(
    model: GRUExtractor,
    config: Config,
    path: str | Path,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    classifier: nn.Linear | None = None,
    temperature: float | None = None,
) -> None:
    """Persist GRU weights, architecture metadata, and normalization stats.

    Args:
        model: Trained GRU extractor to serialize.
        config: Application configuration providing GRU hyperparameters.
        path: Destination checkpoint path. Parent directories are created when
            needed.
        mean: Per-feature mean array from training standardization, or
            ``None`` when not available.
        std: Per-feature standard deviation array from training
            standardization, or ``None`` when not available.
        classifier: Optional trained classification head to persist alongside
            the GRU backbone for end-to-end inference.
        temperature: Temperature scaling parameter from calibration.  When
            ``None`` and the model has a ``temperature`` attribute, that value
            is used automatically.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gru_cfg = config.gru
    raw_model = getattr(model, "_orig_mod", model)

    # Auto-detect temperature from model attribute if not explicitly provided.
    if temperature is None and hasattr(raw_model, "temperature"):
        temperature = float(raw_model.temperature)

    # Use actual model input_size (may differ from config if correlation
    # filtering dropped some GRU features)
    actual_input_size = raw_model.input_norm.normalized_shape[0]
    checkpoint: dict[str, Any] = {
        "model_state_dict": raw_model.state_dict(),
        "input_size": actual_input_size,
        "hidden_size": gru_cfg.hidden_size,
        "num_layers": gru_cfg.num_layers,
        "dropout": gru_cfg.dropout,
        "sequence_length": gru_cfg.sequence_length,
        "bidirectional": getattr(raw_model, "bidirectional", False),
    }
    if classifier is not None:
        raw_classifier = getattr(classifier, "_orig_mod", classifier)
        checkpoint["classifier_state_dict"] = raw_classifier.state_dict()
        checkpoint["num_classes"] = raw_classifier.out_features
    if mean is not None:
        checkpoint["mean"] = mean
    if std is not None:
        checkpoint["std"] = std
    if temperature is not None:
        checkpoint["temperature"] = temperature
    torch.save(checkpoint, path)
    logger.info("GRU model saved: %s", path)


def load_gru_model(path: str | Path) -> tuple[GRUExtractor, dict[str, Any]]:
    """Load a saved GRU extractor and checkpoint metadata.

    Args:
        path: Filesystem path to a checkpoint produced by ``save_gru_model``.

    Returns:
        A tuple ``(model, metadata)`` where ``model`` is initialized with the
        saved weights and set to evaluation mode, and ``metadata`` contains all
        checkpoint fields except ``model_state_dict``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GRU model not found: {path}")

    checkpoint = torch.load(path, weights_only=False)

    model = GRUExtractor(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        bidirectional=checkpoint.get("bidirectional", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Restore temperature scaling parameter if present in checkpoint.
    if "temperature" in checkpoint:
        model.temperature = float(checkpoint["temperature"])  # type: ignore[attr-defined]

    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    logger.info(
        "GRU model loaded: %s (hidden_size=%d, has_norm=%s)",
        path,
        checkpoint["hidden_size"],
        "mean" in metadata and "std" in metadata,
    )
    return model, metadata


def load_gru_classifier(metadata: dict[str, Any]) -> nn.Linear:
    """Rebuild a persisted GRU classification head from checkpoint metadata.

    Args:
        metadata: Metadata returned by :func:`load_gru_model`.

    Returns:
        ``torch.nn.Linear`` classifier set to evaluation mode.

    Raises:
        ValueError: If classifier weights were not stored in the checkpoint.
    """
    if "classifier_state_dict" not in metadata or "num_classes" not in metadata:
        raise ValueError("GRU checkpoint does not include a classifier head")

    classifier = nn.Linear(metadata["hidden_size"], metadata["num_classes"])
    classifier.load_state_dict(metadata["classifier_state_dict"])
    classifier.eval()
    return classifier
