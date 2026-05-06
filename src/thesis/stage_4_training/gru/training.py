"""GRU feature extractor — architecture, dataset, training, and inference.

Provides an attention-pooled GRU that encodes sliding-window price sequences
into fixed-length hidden-state vectors for downstream LightGBM training.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any

from accelerate import Accelerator
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
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.shared.config import Config
from thesis.shared.constants import (
    COSINE_T0,
    COSINE_TMULT,
    GRAD_CLIP_NORM,
    PLATEAU_PATIENCE,
    WARMUP_EPOCHS,
)
from thesis.shared.ui import console
from thesis.stage_4_training.gru.arch import GRUExtractor
from thesis.stage_4_training.gru.calibration import _calibrate_model
from thesis.stage_4_training.gru.data import (
    SequenceDataset,
    _extract_sample_weights,
    prepare_sequences,
)
from thesis.stage_4_training.gru.inference import (
    extract_hidden_states,
)
from thesis.stage_4_training.gru.losses import FocalLoss, _nt_xent_loss

logger = logging.getLogger("thesis.gru")


# Training helpers


def _build_model_and_classifier(
    config: Config,
    input_size: int | None = None,
) -> tuple[GRUExtractor, nn.Linear]:
    """Build GRU model and classification/regression head.

    Device placement is handled externally (Accelerate or manual .to()).

    Args:
        config: Application configuration.
        input_size: Override for number of input features. When ``None``,
            falls back to ``config.gru.input_size``.

    Returns:
        Tuple of (GRUExtractor, nn.Linear) where the linear head outputs
        3 logits for multiclass or 1 value for regression.
    """
    gru_cfg = config.gru
    is_regression = gru_cfg.objective == "regression"

    model = GRUExtractor(
        input_size=input_size or gru_cfg.input_size,
        hidden_size=gru_cfg.hidden_size,
        num_layers=gru_cfg.num_layers,
        dropout=gru_cfg.dropout,
        bidirectional=gru_cfg.bidirectional,
    )

    output_size = 1 if is_regression else config.labels.num_classes
    classifier = nn.Linear(gru_cfg.hidden_size, output_size)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "GRU: %d params, %d layers, hidden=%d, output=%d, objective=%s",
        total_params,
        gru_cfg.num_layers,
        gru_cfg.hidden_size,
        output_size,
        gru_cfg.objective,
    )
    return model, classifier


def _train_epoch(
    model: GRUExtractor,
    classifier: nn.Linear,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accelerator: Any | None = None,
    is_regression: bool = False,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, float]:
    """Run one training epoch.

    When ``accelerator`` is provided, device placement is handled
    automatically by the Accelerate-prepared DataLoader and
    ``accelerator.backward()`` / ``accelerator.clip_grad_norm_()`` are
    used.  Otherwise, falls back to manual ``.to(device)`` and standard
    PyTorch calls.

    Gradient accumulation is used when ``gradient_accumulation_steps > 1``:
    gradients are accumulated over multiple micro-batches before the optimizer
    steps, simulating a larger effective batch size without extra memory.

    Args:
        model: GRU feature extractor.
        classifier: Linear classification/regression head.
        train_loader: Training data loader.
        criterion: Loss function (FocalLoss for multiclass, MSELoss for regression).
        optimizer: Optimizer.
        device: Target device (unused when accelerator handles placement).
        accelerator: Optional Accelerate accelerator instance.
        is_regression: If True, treats target as continuous, reports loss only.
        gradient_accumulation_steps: Number of micro-batches to accumulate
            before stepping the optimizer (default 1 = no accumulation).

    Returns:
        Tuple of (average_loss, accuracy_or_mae). For regression, second value is
        mean absolute error; for classification, it's accuracy.
    """
    model.train()
    classifier.train()
    train_loss = 0.0
    train_metric_sum = 0.0  # classification accuracy or regression MAE
    train_total = 0

    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 3:
            batch_x, batch_y, batch_w = batch
        else:
            batch_x, batch_y = batch
            batch_w = None
        if accelerator is None:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if batch_w is not None:
                batch_w = batch_w.to(device)

        hidden = model(batch_x)
        output = classifier(hidden)

        if is_regression:
            preds = output.squeeze(-1)  # (batch,) continuous
            loss = criterion(preds, batch_y)
            mae = (preds - batch_y).abs().sum().item()
            train_metric_sum += mae
        else:
            loss = criterion(output, batch_y, batch_w)
            train_metric_sum += (output.argmax(dim=1) == batch_y).sum().item()

        # Scale loss for gradient accumulation so effective LR stays constant
        loss = loss / gradient_accumulation_steps

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        # Step only after the accumulation window, or on the last batch
        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_last_batch = batch_idx + 1 == len(train_loader)
        if is_accumulation_step or is_last_batch:
            params = list(model.parameters()) + list(classifier.parameters())
            if accelerator is not None:
                accelerator.clip_grad_norm_(params, max_norm=GRAD_CLIP_NORM)
            else:
                torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()

        # Report loss at original (unscaled) scale for interpretable logging
        train_loss += loss.item() * gradient_accumulation_steps * len(batch_x)
        train_total += len(batch_y)

    return (
        train_loss / train_total,
        train_metric_sum / train_total if train_total > 0 else 0.0,
    )


def _validate_epoch(
    model: GRUExtractor,
    classifier: nn.Linear,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    accelerator: Any | None = None,
    is_regression: bool = False,
) -> tuple[float, float]:
    """Run one validation epoch.

    When ``accelerator`` is provided, device placement is handled
    automatically by the Accelerate-prepared DataLoader.  Otherwise,
    falls back to manual ``.to(device)``.

    Args:
        model: GRU feature extractor.
        classifier: Linear classification/regression head.
        val_loader: Validation data loader.
        criterion: Loss function (FocalLoss or MSELoss).
        device: Target device (unused when accelerator handles placement).
        accelerator: Optional Accelerate accelerator instance.
        is_regression: If True, treats target as continuous.

    Returns:
        Tuple of (average_loss, accuracy_or_mae).
    """
    model.eval()
    classifier.eval()
    val_loss = 0.0
    val_metric_sum = 0.0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            if accelerator is None:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

            hidden = model(batch_x)
            output = classifier(hidden)

            if is_regression:
                preds = output.squeeze(-1)
                loss = criterion(preds, batch_y)
                val_metric_sum += (preds - batch_y).abs().sum().item()
            else:
                loss = criterion(output, batch_y)
                val_metric_sum += (output.argmax(dim=1) == batch_y).sum().item()

            val_loss += loss.item() * len(batch_x)
            val_total += len(batch_y)

    return val_loss / val_total, val_metric_sum / val_total if val_total > 0 else 0.0


def _pretrain_contrastive(
    model: GRUExtractor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Any,
    epochs: int,
    temperature: float = 0.1,
) -> list[float]:
    """Pretrain GRU encoder with contrastive InfoNCE loss.

    Adjacent windows in the batch form positive pairs (temporal
    proximity); all other samples serve as negatives. Uses NT-Xent
    loss with cosine similarity on GRU hidden states. This forces
    the GRU to learn meaningful temporal representations before
    fine-tuning for the downstream classification/regression task.

    Args:
        model: GRU feature extractor (classifier not needed).
        train_loader: Non-shuffled DataLoader preserving temporal
            adjacency for positive pair construction.
        optimizer: Optimizer for GRU parameters only.
        accelerator: Accelerate accelerator instance.
        epochs: Number of contrastive pretraining epochs.
        temperature: NT-Xent temperature (default 0.1).

    Returns:
        List of per-epoch average contrastive losses.

    Raises:
        ValueError: If ``epochs`` is less than 1.
    """
    if epochs < 1:
        raise ValueError("contrastive pretrain epochs must be >= 1")

    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch in train_loader:
            if len(batch) == 3:
                batch_x, _, _ = batch
            else:
                batch_x, _ = batch

            N = batch_x.shape[0]
            # NT-Xent requires even batch size for symmetric pairing
            if N % 2 != 0:
                batch_x = batch_x[:-1]
                N -= 1
            if N < 2:
                continue

            # Get hidden states from GRU (no classifier)
            hidden = model(batch_x)  # (N, hidden_dim)

            # Split into two views: even indices (adjacent anchors)
            # paired with odd indices (temporal neighbours)
            z1 = hidden[0::2]  # (N/2, D)
            z2 = hidden[1::2]  # (N/2, D)

            loss = _nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item() * N
            epoch_samples += N

        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        history.append(avg_loss)
        logger.info(
            "Contrastive pretrain epoch %d/%d: loss=%.4f",
            epoch + 1,
            epochs,
            avg_loss,
        )

    logger.info(
        "Contrastive pretraining finished: losses=%s",
        [round(x, 4) for x in history],
    )
    return history


# Training loop


def train_gru(
    config: Config,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    window_index: int = 0,
) -> tuple[
    GRUExtractor,
    nn.Linear,
    np.ndarray,
    np.ndarray,
    list[dict[str, float]],
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Train a GRU classifier and extract hidden-state features.

    The GRU backbone is trained with a temporary linear head using
    Focal Loss and early stopping on validation loss.  A cosine
    annealing scheduler with linear warmup stabilises early training.
    The best checkpoint is restored before hidden states are extracted
    for downstream LightGBM training.

    Args:
        config: Application configuration containing GRU and labeling settings.
        train_df: Training split as a time-series DataFrame.
        val_df: Validation split as a time-series DataFrame.
        window_index: Walk-forward window index used to diversify the random
            seed per window.  When > 0, the effective seed becomes
            ``config.workflow.random_seed + window_index`` so each window
            starts from a different weight initialisation.

    Returns:
        A tuple containing ``(model, classifier, train_hidden, val_hidden,
        history, mean, std)`` where ``train_hidden`` and ``val_hidden`` are
        extracted GRU embeddings, ``history`` stores per-epoch metrics, and
        ``mean``/``std`` are the per-feature normalization statistics from
        the training dataset.
    """
    gru_cfg = config.gru
    gru_cols = list(config.gru.feature_cols)
    seed = config.workflow.random_seed
    is_regression = gru_cfg.objective == "regression"
    label_col = "regression_target" if is_regression else "label"
    logger.info(
        "GRU objective: %s (model.objective=%s)",
        gru_cfg.objective,
        config.model.objective,
    )

    # Dynamically filter GRU columns to those surviving correlation filtering.
    # ``prepare_sequences`` adds ``log_returns`` on-the-fly via
    # ``_ensure_log_returns``, so treat it as always available.
    _dynamic_cols = {"log_returns"}
    available_cols = set(train_df.columns) | _dynamic_cols
    configured_set = set(gru_cols)
    missing = configured_set - available_cols
    if missing:
        logger.warning(
            "GRU: %d configured features not in data (dropped by correlation "
            "filter): %s — adapting input_size automatically",
            len(missing),
            sorted(missing),
        )
        gru_cols = [c for c in gru_cols if c in available_cols]
    input_size = len(gru_cols)
    logger.info(
        "GRU input: %d features (config had %d)", input_size, len(configured_set)
    )

    # Set seeds for reproducibility — offset by window_index so each
    # walk-forward window starts from a different weight initialisation,
    # increasing ensemble diversity across windows.
    torch.manual_seed(seed + window_index)
    np.random.seed(seed + window_index)

    # Accelerate handles device placement, mixed precision, and multi-GPU.
    # fp16 is useful on CUDA, but on CPU it adds no value and can cause
    # avoidable numerical/runtime issues.
    mixed_precision = "fp16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Prepare sequences
    train_seq, train_labels, _ = prepare_sequences(
        train_df, gru_cols, gru_cfg.sequence_length, label_col=label_col
    )
    val_seq, val_labels, _ = prepare_sequences(
        val_df, gru_cols, gru_cfg.sequence_length, label_col=label_col
    )
    train_sample_weights = _extract_sample_weights(train_df, gru_cfg.sequence_length)

    if train_seq.shape[0] == 0:
        raise ValueError(
            f"GRU training: 0 sequences from train data "
            f"(sequence_length={gru_cfg.sequence_length}, "
            f"train rows={len(train_df)}). "
            "Ensure train_df has >= sequence_length rows."
        )
    if val_seq.shape[0] == 0:
        raise ValueError(
            f"GRU validation: 0 sequences from val data "
            f"(sequence_length={gru_cfg.sequence_length}, "
            f"val rows={len(val_df)}). "
            "Ensure val_df has >= sequence_length rows."
        )

    # Remap labels from {-1, 0, 1} to {0, 1, 2} for PyTorch indexing (multiclass only)
    if is_regression:
        if train_labels is not None:
            train_labels = train_labels.astype(np.float32)
        if val_labels is not None:
            val_labels = val_labels.astype(np.float32)
    else:
        if train_labels is not None:
            train_labels = (train_labels + 1).astype(np.int32)
        if val_labels is not None:
            val_labels = (val_labels + 1).astype(np.int32)

    # Create datasets & loaders — use training statistics for val to prevent leakage
    train_dataset = SequenceDataset(
        train_seq, train_labels, sample_weights=train_sample_weights
    )
    val_dataset = SequenceDataset(
        val_seq, val_labels, mean=train_dataset.mean, std=train_dataset.std
    )

    # Note: shuffle=True shuffles which sequences are batched each epoch,
    # not the internal sequence order.  This is standard mini-batch RNN
    # practice to improve generalisation.  Val loader keeps shuffle=False
    # to preserve temporal order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=gru_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=gru_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Build model (no manual .to(device) — Accelerate handles placement)
    device = accelerator.device
    model, classifier = _build_model_and_classifier(config, input_size=input_size)

    # Skip ``torch.compile`` on CPU. In the current PyTorch build used by this
    # repo, that path imports MKLDNN wrappers backed by deprecated
    # ``torch.jit.script_method`` internals, which turns test runs noisy and
    # unstable under strict warning settings.
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable — skipping torch.compile on CPU")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=gru_cfg.learning_rate,
    )

    # Prepare model, classifier, optimizer, and loaders with Accelerate
    model, classifier, optimizer, train_loader, val_loader = accelerator.prepare(
        model, classifier, optimizer, train_loader, val_loader
    )

    # Contrastive pretraining (opt-in — epochs=0 skips entirely)
    if gru_cfg.contrastive_pretrain_epochs > 0:
        logger.info(
            "Starting contrastive pretraining for %d epoch(s)",
            gru_cfg.contrastive_pretrain_epochs,
        )

        # Non-shuffled loader preserves temporal adjacency for pairing
        pretrain_loader = DataLoader(
            train_dataset,
            batch_size=gru_cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )
        pretrain_loader = accelerator.prepare(pretrain_loader)

        pretrain_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=gru_cfg.learning_rate,
        )

        _pretrain_contrastive(
            model=model,
            train_loader=pretrain_loader,
            optimizer=pretrain_optimizer,
            accelerator=accelerator,
            epochs=gru_cfg.contrastive_pretrain_epochs,
        )

    # LR scheduler: 3-epoch linear warmup → cosine annealing with warm restarts
    # (T_0=10, T_mult=2)
    warmup_epochs = WARMUP_EPOCHS

    def _lr_lambda(epoch: int) -> float:
        """Cosine annealing LR schedule with 3-epoch linear warmup.

        Warmup: linearly scales LR from 0 to 1 over the first 3 epochs.
        After warmup: cosine annealing with warm restarts
        (``T_0=10``, ``T_mult=2``).

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            LR multiplier in [0, 1].
        """
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Cosine annealing with warm restarts (T_0=10, T_mult=2)
        adjusted = epoch - warmup_epochs
        t_0 = COSINE_T0
        t_mult = COSINE_TMULT
        t_i = t_0
        while adjusted >= t_i:
            adjusted -= t_i
            t_i *= t_mult
        return 0.5 * (1.0 + math.cos(math.pi * adjusted / t_i))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Loss function: FocalLoss for multiclass, MSELoss for regression
    if is_regression:
        criterion: nn.Module = nn.MSELoss()
    else:
        # Class-weighted Focal Loss to handle label imbalance. Build a full
        # num_classes-length alpha vector so windows missing a class still train.
        unique_classes = np.unique(train_labels)
        class_weights_arr = compute_class_weight(
            "balanced", classes=unique_classes, y=train_labels
        )
        class_weights_full = np.ones(config.labels.num_classes, dtype=np.float32)
        for cls, weight in zip(unique_classes, class_weights_arr):
            class_weights_full[int(cls)] = float(weight)
        class_weights_tensor = torch.tensor(class_weights_full, dtype=torch.float32).to(
            accelerator.device
        )
        criterion = FocalLoss(
            gamma=gru_cfg.focal_loss_gamma,
            alpha=class_weights_tensor,
            num_classes=config.labels.num_classes,
        )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    patience_counter = 0
    plateau_warned = False
    history: list[dict[str, float]] = []
    stage_start = time.perf_counter()

    metric_label = "mae" if is_regression else "acc"
    t_metric_label = "t_" + metric_label
    v_metric_label = "v_" + metric_label
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]GRU training"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]t_loss={task.fields[t_loss]:.4f}"),
        TextColumn(f"[green]{t_metric_label}={{task.fields[t_metric]:.4f}}"),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TextColumn(f"[green]{v_metric_label}={{task.fields[v_metric]:.4f}}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    )

    with progress:
        task = progress.add_task(
            "epochs",
            total=gru_cfg.epochs,
            t_loss=0.0,
            t_metric=0.0,
            v_loss=0.0,
            v_metric=0.0,
        )

        for epoch in range(gru_cfg.epochs):
            train_loss, train_metric = _train_epoch(
                model,
                classifier,
                train_loader,
                criterion,
                optimizer,
                device,
                accelerator=accelerator,
                is_regression=is_regression,
                gradient_accumulation_steps=gru_cfg.gradient_accumulation_steps,
            )
            val_loss, val_metric = _validate_epoch(
                model,
                classifier,
                val_loader,
                criterion,
                device,
                accelerator=accelerator,
                is_regression=is_regression,
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 4),
                    f"train_{metric_label}": round(train_metric, 4),
                    "val_loss": round(val_loss, 4),
                    f"val_{metric_label}": round(val_metric, 4),
                    "lr": round(current_lr, 6),
                }
            )
            logger.info("Epoch %d: lr=%.6f", epoch + 1, current_lr)

            progress.update(
                task,
                advance=1,
                t_loss=train_loss,
                t_metric=train_metric,
                v_loss=val_loss,
                v_metric=val_metric,
            )

            # Early stopping — enforce min_epochs before allowing patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    "model": copy.deepcopy(
                        accelerator.unwrap_model(model).state_dict()
                    ),
                    "classifier": copy.deepcopy(
                        accelerator.unwrap_model(classifier).state_dict()
                    ),
                }
                patience_counter = 0
                plateau_warned = False
            else:
                patience_counter += 1
                # Plateau detection: warn once when val-loss stalls for
                # PLATEAU_PATIENCE consecutive epochs without improvement.
                if not plateau_warned and patience_counter >= PLATEAU_PATIENCE:
                    plateau_warned = True
                    logger.warning(
                        "Plateau detected: val_loss=%.4f hasn't improved for "
                        "%d epochs (LR=%.6f). Consider reducing learning rate.",
                        val_loss,
                        patience_counter,
                        current_lr,
                    )
                if (
                    patience_counter >= gru_cfg.patience
                    and epoch + 1 >= gru_cfg.min_epochs
                ):
                    if epoch + 1 < gru_cfg.epochs:
                        logger.info(
                            "\nEarly stop at epoch %d (patience=%d, min_epochs=%d)",
                            epoch + 1,
                            gru_cfg.patience,
                            gru_cfg.min_epochs,
                        )
                    break

    # Summary
    total_time = time.perf_counter() - stage_start
    best_metric_key = f"val_{metric_label}"
    best_val_metric = history[best_epoch - 1][best_metric_key] if history else 0.0
    logger.info(
        "GRU done: %d/%d epochs, best=e%d v_loss=%.4f v_%s=%.4f (%.1fs)",
        len(history),
        gru_cfg.epochs,
        best_epoch,
        best_val_loss,
        metric_label,
        best_val_metric,
        total_time,
    )

    # Restore best model (unwrap to load raw state dict)
    if best_state is not None:
        accelerator.unwrap_model(model).load_state_dict(best_state["model"])
        accelerator.unwrap_model(classifier).load_state_dict(best_state["classifier"])

    # Temperature scaling calibration (multiclass only — not applicable to regression)
    temperature: float = 1.0
    if not is_regression and gru_cfg.temperature_scaling:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_classifier = accelerator.unwrap_model(classifier)
        temperature = _calibrate_model(
            unwrapped_model,
            unwrapped_classifier,
            val_loader,
            device,
            accelerator=accelerator,
        )
        unwrapped_model.temperature = temperature  # type: ignore[attr-defined]
        logger.info("Temperature T=%.4f stored on GRU model", temperature)

    # Extract hidden states for LightGBM (use unwrapped model)
    # Apply training-set standardization so inference matches what the model saw
    train_mean = train_dataset.mean
    train_std = train_dataset.std
    train_hidden = extract_hidden_states(
        accelerator.unwrap_model(model),
        train_seq,
        gru_cfg.batch_size,
        device,
        mean=train_mean,
        std=train_std,
    )
    val_hidden = extract_hidden_states(
        accelerator.unwrap_model(model),
        val_seq,
        gru_cfg.batch_size,
        device,
        mean=train_mean,
        std=train_std,
    )

    logger.info(
        "GRU hidden states: train=%s, val=%s", train_hidden.shape, val_hidden.shape
    )

    return (
        model,
        classifier,
        train_hidden,
        val_hidden,
        history,
        train_mean,
        train_std,
        gru_cols,
    )
