"""Temperature-scaling calibration for GRU classifiers.

Collects logits on the validation set and optimises a single temperature
parameter to minimise cross-entropy, following Guo et al. (2017).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 – standard PyTorch abbreviation
from torch.utils.data import DataLoader

from thesis.shared.constants import CALIB_LR, CALIB_MAX_ITER, ECE_N_BINS
from thesis.stage_4_training.gru.arch import GRUExtractor

logger = logging.getLogger("thesis.gru")


def _compute_ece(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = ECE_N_BINS
) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into ``n_bins`` equal-width confidence bins
    and measures the absolute difference between average confidence
    and accuracy within each bin, weighted by bin size.

    Args:
        probs: Softmax probabilities with shape ``(N, C)``.
        labels: Ground-truth class indices with shape ``(N,)``.
        n_bins: Number of confidence bins (default 10).

    Returns:
        ECE value (0.0 = perfectly calibrated).
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()

    ece = 0.0
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum().item()
        if bin_size > 0:
            bin_conf = confidences[in_bin].mean().item()
            bin_acc = accuracies[in_bin].mean().item()
            ece += (bin_size / len(probs)) * abs(bin_conf - bin_acc)

    return ece


def _calibrate_model(
    model: GRUExtractor,
    classifier: nn.Linear,
    val_loader: DataLoader,
    device: torch.device,
    accelerator: Any | None = None,
) -> float:
    """Calibrate classifier probabilities via temperature scaling.

    Collects logits from the validation set and optimizes a single
    temperature parameter ``T`` to minimise cross-entropy, following
    Guo et al. (2017) *"On Calibration of Modern Neural Networks"*.

    After calibration the predicted probabilities are computed as
    ``softmax(logits / T)`` instead of ``softmax(logits)``.

    **Interpretation**:
        - ``T > 1.0`` → model was overconfident (softens probabilities).
        - ``T < 1.0`` → model was underconfident (sharpens probabilities).
        - ``T ≈ 1.0`` → model was already well-calibrated.

    Args:
        model: Trained GRU feature extractor (unwrapped, in eval mode).
        classifier: Trained classification head (unwrapped, in eval mode).
        val_loader: Validation data loader yielding ``(x, y)`` batches.
        device: Computation device.
        accelerator: Optional Accelerate accelerator instance.  When
            provided, device placement is handled automatically.

    Returns:
        Optimised temperature value as a Python float.
    """
    model.eval()
    classifier.eval()

    # Collect all logits and labels from the validation set.
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            if accelerator is None:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
            hidden = model(batch_x)
            logits = classifier(hidden)
            all_logits.append(logits.cpu())
            all_labels.append(batch_y.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    n_samples, n_classes = logits_tensor.shape
    logger.info(
        "Temperature scaling: %d val samples, %d classes",
        n_samples,
        n_classes,
    )

    # Optimise the single temperature parameter with LBFGS.
    # This is a 1-D convex problem — LBFGS converges in a few iterations.
    temperature_param = nn.Parameter(torch.ones(1))

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()  # type: ignore[name-defined]  # noqa: F821
        scaled_logits = logits_tensor / temperature_param
        loss = F.cross_entropy(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    optimizer = torch.optim.LBFGS(
        [temperature_param],
        lr=CALIB_LR,
        max_iter=CALIB_MAX_ITER,
        line_search_fn="strong_wolfe",
    )
    optimizer.step(_closure)

    T = float(temperature_param.item())

    # Log pre- and post-calibration NLL and ECE.
    with torch.no_grad():
        pre_probs = torch.softmax(logits_tensor, dim=1)
        post_probs = torch.softmax(logits_tensor / T, dim=1)
        pre_nll = F.cross_entropy(logits_tensor, labels_tensor).item()
        post_nll = F.cross_entropy(logits_tensor / T, labels_tensor).item()
        pre_ece = _compute_ece(pre_probs, labels_tensor)
        post_ece = _compute_ece(post_probs, labels_tensor)

    logger.info(
        "Temperature scaling: T=%.4f, NLL %.4f→%.4f, ECE %.4f→%.4f",
        T,
        pre_nll,
        post_nll,
        pre_ece,
        post_ece,
    )

    return T
