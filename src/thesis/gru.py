"""GRU feature extractor — architecture, dataset, training, and inference.

Provides an attention-pooled GRU that encodes sliding-window price sequences
into fixed-length hidden-state vectors for downstream LightGBM training.

Public API::

    from thesis.gru import (
        GRUExtractor,
        SequenceDataset,
        prepare_sequences,
        train_gru,
        extract_hidden_states,
        save_gru_model,
        load_gru_model,
    )
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from thesis.config import Config

logger = logging.getLogger("thesis.gru")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class VariationalDropout(nn.Module):
    """Variational (locked) dropout — same mask across all timesteps.

    Standard dropout draws an independent mask per element, breaking the
    temporal correlation that RNNs rely on.  Variational dropout generates
    *one* mask per sample and broadcasts it over the entire sequence so
    the same features are consistently dropped at every timestep.

    Args:
        p: Dropout probability (0.0 = no dropout).
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply variational dropout.

        Args:
            x: Tensor with shape ``(batch, seq_len, features)``.

        Returns:
            Tensor with identical shape, dropout applied.
        """
        if not self.training or self.p == 0:
            return x
        # One mask per sample, broadcast over all timesteps.
        mask = torch.bernoulli(torch.full_like(x[:, :1, :], 1 - self.p)) / (1 - self.p)
        return x * mask


class GRUExtractor(nn.Module):
    """GRU-based feature extractor with learned attention pooling.

    Encodes a (batch, seq_len, input_size) sequence into a single
    (batch, hidden_size) vector via a 2-layer GRU followed by attention
    weighted-sum over all timesteps.  The attention layer learns which
    positions in the input sequence are most informative, rather than
    blindly using the final hidden state.

    Architecture::

        input → LayerNorm → VariationalDropout → GRU(2-layer) → attention → weighted sum → output

    Args:
        input_size: Number of features per timestep.
        hidden_size: GRU hidden dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (applied only if num_layers > 1).
        variational_dropout: Variational dropout probability applied to the
            GRU input (0.0 = disabled).
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        variational_dropout: float = 0.1,
    ) -> None:
        """Initialize the GRU extractor module.

        Args:
            input_size: Number of expected features at each timestep.
            hidden_size: Dimensionality of the GRU hidden state.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability applied between GRU layers when
                ``num_layers > 1``.
            variational_dropout: Variational dropout probability applied to
                the GRU input before the recurrent computation.

        Returns:
            None.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_size)
        self.var_drop = VariationalDropout(p=variational_dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Learned attention scorer: maps each timestep's hidden state to a scalar score.
        self.attn_scorer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batched sequence into an attention-weighted context vector.

        Args:
            x: Input tensor with shape ``(batch, seq_len, input_size)``.

        Returns:
            Attention-weighted context vector with shape
            ``(batch, hidden_size)``.
        """
        x = self.input_norm(x)
        x = self.var_drop(x)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_size)

        # Compute attention weights over the sequence dimension.
        # scores: (batch, seq_len, 1) → squeeze → softmax over seq_len
        attn_weights = torch.softmax(
            self.attn_scorer(gru_out), dim=1
        )  # (batch, seq_len, 1)

        # Weighted sum of hidden states: (batch, seq_len, hidden) → (batch, hidden)
        context = (gru_out * attn_weights).sum(dim=1)
        return context


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Down-weights easy (well-classified) examples so the model focuses
    on hard, misclassified samples.  This is critical when one class
    dominates (e.g. Hold ≈ 69 % in 3-class financial labeling).

    Formula::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where::

        p_t  = probability of the correct class
        gamma >= 0  (focusing parameter; higher → more emphasis on hard examples)
        alpha_t    = per-class weight (optional)

    Args:
        gamma: Focusing parameter.  ``gamma=0`` reduces to standard
            cross-entropy.  Default ``2.0`` follows Lin et al.
        alpha: Per-class weights as a 1-D tensor of length
            ``num_classes``, or ``None`` for uniform weighting.
            Typically set to inverse class frequencies.
        num_classes: Number of target classes (default 3).

    Shape:
        - logits: ``(N, C)`` — raw scores before softmax.
        - targets: ``(N,)`` — integer class indices.
        - output: scalar tensor.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        if alpha is not None:
            if alpha.dim() != 1 or alpha.size(0) != num_classes:
                msg = (
                    f"alpha must be a 1-D tensor of length {num_classes}, "
                    f"got shape {tuple(alpha.shape)}"
                )
                raise ValueError(msg)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(N, C)`` raw scores.
            targets: ``(N,)`` integer class labels.

        Returns:
            Scalar loss tensor.
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Probability of the correct class per sample  →  (N,)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        p_t = (probs * targets_one_hot).sum(dim=-1).clamp(min=1e-8)

        # Standard cross-entropy (no reduction)  →  (N,)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Focal modulating factor  (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
        else:
            alpha_t = 1.0

        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Dataset & sequence preparation
# ---------------------------------------------------------------------------


class SequenceDataset(Dataset):
    """Sliding-window dataset for GRU input sequences.

    Each sample is a window of (sequence_length, input_size) values
    from the GRU input columns, plus the corresponding label.

    Args:
        sequences: Array of shape ``(n_samples, sequence_length, n_features)``
            containing precomputed GRU sequences.
        labels: Optional array of shape ``(n_samples,)`` containing labels
            aligned with ``sequences``.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        """Initialize the dataset with per-feature standardization.

        When ``mean`` and ``std`` are provided (e.g. from the training
        set), those statistics are used instead of computing new ones
        from ``sequences``.  This prevents data leakage when constructing
        the validation or test datasets.

        Args:
            sequences: 3D array shaped ``(n_samples, sequence_length,
                n_features)``.  Standardized and stored as a float tensor.
            labels: Optional 1D array of labels with length ``n_samples``.
                Stored internally as a long tensor copy when provided.
            mean: Optional per-feature mean array (shape ``(1, 1, n_features)``)
                from the training set.  Computed from ``sequences`` when ``None``.
            std: Optional per-feature std array (shape ``(1, 1, n_features)``)
                from the training set.  Computed from ``sequences`` when ``None``.

        Returns:
            None.
        """
        # Per-feature standardization: use provided stats or compute from data
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = sequences.mean(axis=(0, 1), keepdims=True)
            self.std = sequences.std(axis=(0, 1), keepdims=True) + 1e-8
        standardized = (sequences - self.mean) / self.std
        self.sequences = torch.from_numpy(standardized.copy()).float()
        self.labels = (
            torch.from_numpy(labels.copy()).long() if labels is not None else None
        )

    def __len__(self) -> int:
        """Return the number of available samples.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Retrieve a single sequence sample.

        Args:
            idx: Sample index.

        Returns:
            A tuple of ``(sequence, label)`` where ``label`` is ``None`` when
            labels were not provided.
        """
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx], None


def _sliding_windows(data: np.ndarray, window: int) -> np.ndarray:
    """Construct a 3D sliding-window view over a 2D array.

    Args:
        data: 2D array with shape ``(n_rows, n_features)``.
        window: Length of each sliding window.

    Returns:
        View array with shape ``(n_samples, window, n_features)`` where
        ``n_samples = n_rows - window + 1``.
    """
    n_rows, n_features = data.shape
    n_samples = n_rows - window + 1

    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(
        data,
        shape=(n_samples, window, n_features),
        strides=strides,
    )


def _ensure_log_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure that ``log_returns`` exists in the input DataFrame.

    Args:
        df: Input DataFrame expected to contain a ``close`` column.

    Returns:
        DataFrame that includes a ``log_returns`` column.
    """
    if "log_returns" not in df.columns:
        return df.with_columns(
            pl.col("close")
            .log()
            .diff()
            .fill_null(strategy="forward")
            .fill_null(0.0)
            .alias("log_returns")
        )
    return df


def _validate_gru_cols(df: pl.DataFrame, gru_cols: list[str]) -> None:
    """Validate that all requested GRU input columns are present.

    Args:
        df: DataFrame containing candidate GRU columns.
        gru_cols: Ordered list of required GRU input column names.

    Returns:
        None.

    Raises:
        ValueError: If any requested GRU column is missing.
    """
    for col in gru_cols:
        if col not in df.columns:
            raise ValueError(f"GRU input column '{col}' not found in DataFrame")


def _extract_labels(
    df: pl.DataFrame, label_col: str, sequence_length: int
) -> np.ndarray | None:
    """Extract labels aligned to the end index of each sequence.

    Args:
        df: Input DataFrame.
        label_col: Label column name.
        sequence_length: Sliding-window size.

    Returns:
        Label array aligned with generated windows, or ``None`` when
        ``label_col`` does not exist.
    """
    if label_col not in df.columns:
        return None
    return df[label_col].to_numpy()[sequence_length - 1 :]


def _identify_static_cols(
    df: pl.DataFrame, gru_cols: list[str], exclude_cols: frozenset[str], label_col: str
) -> list[str]:
    """Identify non-GRU feature columns for downstream static features.

    Args:
        df: Input DataFrame.
        gru_cols: GRU input column names.
        exclude_cols: Columns to exclude explicitly.
        label_col: Label column name.

    Returns:
        Ordered list of static feature column names.
    """
    gru_col_set = set(gru_cols)
    return [
        c
        for c in df.columns
        if c not in exclude_cols and c not in gru_col_set and c != label_col
    ]


def prepare_sequences(
    df: pl.DataFrame,
    gru_cols: list[str],
    sequence_length: int,
    label_col: str = "label",
    exclude_cols: frozenset[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Build sliding-window sequences for GRU training.

    Args:
        df: Feature-enriched DataFrame with GRU input columns and labels.
        gru_cols: Column names for GRU input (e.g. ['log_returns', 'rsi_14']).
        sequence_length: Window size for each sequence.
        label_col: Name of the label column.
        exclude_cols: Columns to exclude from static features.

    Returns:
        Tuple of (sequences, labels, static_feature_cols):
        - sequences: np.ndarray of shape (n_samples, seq_len, input_size)
        - labels: np.ndarray of shape (n_samples,) or None if label missing
        - static_feature_cols: list of column names for static features
    """
    if exclude_cols is None:
        exclude_cols = frozenset()

    df = _ensure_log_returns(df)
    _validate_gru_cols(df, gru_cols)

    gru_data = df.select(gru_cols).to_numpy()
    n_rows = len(df)
    n_samples = n_rows - sequence_length + 1
    if n_samples <= 0:
        raise ValueError(
            f"DataFrame has {n_rows} rows, need at least {sequence_length} "
            f"for sequence_length={sequence_length}"
        )

    sequences = _sliding_windows(gru_data, sequence_length)
    labels = _extract_labels(df, label_col, sequence_length)
    static_cols = _identify_static_cols(df, gru_cols, exclude_cols, label_col)

    return sequences, labels, static_cols


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _build_model_and_classifier(
    config: Config,
    input_size: int | None = None,
) -> tuple[GRUExtractor, nn.Linear]:
    """Build GRU model and classification head.

    Device placement is handled externally (Accelerate or manual .to()).

    Args:
        config: Application configuration.
        input_size: Override for number of input features. When ``None``,
            falls back to ``config.gru.input_size``.

    Returns:
        Tuple of (GRUExtractor, nn.Linear classifier).
    """
    gru_cfg = config.gru
    model = GRUExtractor(
        input_size=input_size or gru_cfg.input_size,
        hidden_size=gru_cfg.hidden_size,
        num_layers=gru_cfg.num_layers,
        dropout=gru_cfg.dropout,
    )

    num_classes = config.labels.num_classes
    classifier = nn.Linear(gru_cfg.hidden_size, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "GRU: %d params, %d layers, hidden=%d",
        total_params,
        gru_cfg.num_layers,
        gru_cfg.hidden_size,
    )
    return model, classifier


def _train_epoch(
    model: GRUExtractor,
    classifier: nn.Linear,
    train_loader: DataLoader,
    criterion: FocalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accelerator: Any | None = None,
) -> tuple[float, float]:
    """Run one training epoch.

    When ``accelerator`` is provided, device placement is handled
    automatically by the Accelerate-prepared DataLoader and
    ``accelerator.backward()`` / ``accelerator.clip_grad_norm_()`` are
    used.  Otherwise, falls back to manual ``.to(device)`` and standard
    PyTorch calls.

    Args:
        model: GRU feature extractor.
        classifier: Linear classification head.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Target device (unused when accelerator handles placement).
        accelerator: Optional Accelerate accelerator instance.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    classifier.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:
        if accelerator is None:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

        hidden = model(batch_x)
        logits = classifier(hidden)

        loss = criterion(logits, batch_y)
        optimizer.zero_grad()

        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()

        params = list(model.parameters()) + list(classifier.parameters())
        if accelerator is not None:
            accelerator.clip_grad_norm_(params, max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        train_loss += loss.item() * len(batch_x)
        train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        train_total += len(batch_y)

    return train_loss / train_total, train_correct / train_total


def _validate_epoch(
    model: GRUExtractor,
    classifier: nn.Linear,
    val_loader: DataLoader,
    criterion: FocalLoss,
    device: torch.device,
    accelerator: Any | None = None,
) -> tuple[float, float]:
    """Run one validation epoch.

    When ``accelerator`` is provided, device placement is handled
    automatically by the Accelerate-prepared DataLoader.  Otherwise,
    falls back to manual ``.to(device)``.

    Args:
        model: GRU feature extractor.
        classifier: Linear classification head.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Target device (unused when accelerator handles placement).
        accelerator: Optional Accelerate accelerator instance.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            if accelerator is None:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

            hidden = model(batch_x)
            logits = classifier(hidden)

            loss = criterion(logits, batch_y)
            val_loss += loss.item() * len(batch_x)
            val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            val_total += len(batch_y)

    return val_loss / val_total, val_correct / val_total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_gru(
    config: Config,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
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

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Accelerate handles device placement, mixed precision, and multi-GPU
    accelerator = Accelerator(mixed_precision="fp16")

    # Prepare sequences
    train_seq, train_labels, _ = prepare_sequences(
        train_df, gru_cols, gru_cfg.sequence_length
    )
    val_seq, val_labels, _ = prepare_sequences(
        val_df, gru_cols, gru_cfg.sequence_length
    )

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

    # Remap labels from {-1, 0, 1} to {0, 1, 2} for PyTorch indexing
    if train_labels is not None:
        train_labels = (train_labels + 1).astype(np.int32)
    if val_labels is not None:
        val_labels = (val_labels + 1).astype(np.int32)

    # Create datasets & loaders — use training statistics for val to prevent leakage
    train_dataset = SequenceDataset(train_seq, train_labels)
    val_dataset = SequenceDataset(
        val_seq, val_labels, mean=train_dataset.mean, std=train_dataset.std
    )

    # Note: shuffle=True for training loader shuffles which sequences are processed in each
    # epoch (not the sequence order itself). This is standard practice for mini-batch RNN
    # training to improve generalization. Val loader keeps shuffle=False to preserve order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=gru_cfg.batch_size,
        shuffle=True,
        drop_last=True,
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

    # LR scheduler: 3-epoch linear warmup → cosine annealing with warm restarts
    # (T_0=10, T_mult=2)
    warmup_epochs = 3

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Cosine annealing with warm restarts (T_0=10, T_mult=2)
        adjusted = epoch - warmup_epochs
        t_0 = 10
        t_mult = 2
        t_i = t_0
        while adjusted >= t_i:
            adjusted -= t_i
            t_i *= t_mult
        return 0.5 * (1.0 + math.cos(math.pi * adjusted / t_i))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Class-weighted Focal Loss to handle label imbalance (Hold ~3.5% of samples)
    unique_classes = np.unique(train_labels)
    class_weights_arr = compute_class_weight(
        "balanced", classes=unique_classes, y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32).to(
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
    history: list[dict[str, float]] = []
    stage_start = time.perf_counter()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]GRU training"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[cyan]t_loss={task.fields[t_loss]:.4f}"),
        TextColumn("[green]t_acc={task.fields[t_acc]:.3f}"),
        TextColumn("•"),
        TextColumn("[cyan]v_loss={task.fields[v_loss]:.4f}"),
        TextColumn("[green]v_acc={task.fields[v_acc]:.3f}"),
        TimeElapsedColumn(),
        transient=False,
        console=Console(stderr=True),
    )

    with progress:
        task = progress.add_task(
            "epochs",
            total=gru_cfg.epochs,
            t_loss=0.0,
            t_acc=0.0,
            v_loss=0.0,
            v_acc=0.0,
        )

        for epoch in range(gru_cfg.epochs):
            train_loss, train_acc = _train_epoch(
                model,
                classifier,
                train_loader,
                criterion,
                optimizer,
                device,
                accelerator=accelerator,
            )
            val_loss, val_acc = _validate_epoch(
                model,
                classifier,
                val_loader,
                criterion,
                device,
                accelerator=accelerator,
            )
            scheduler.step()

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                }
            )

            progress.update(
                task,
                advance=1,
                t_loss=train_loss,
                t_acc=train_acc,
                v_loss=val_loss,
                v_acc=val_acc,
            )

            # Early stopping
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
            else:
                patience_counter += 1
                if patience_counter >= gru_cfg.patience:
                    if epoch + 1 < gru_cfg.epochs:
                        logger.info(
                            "Early stop at epoch %d (patience=%d)",
                            epoch + 1,
                            gru_cfg.patience,
                        )
                    break

    # Summary
    total_time = time.perf_counter() - stage_start
    best_val_acc = history[best_epoch - 1]["val_acc"] if history else 0.0
    logger.info(
        "GRU done: %d/%d epochs, best=e%d v_loss=%.4f v_acc=%.3f (%.1fs)",
        len(history),
        gru_cfg.epochs,
        best_epoch,
        best_val_loss,
        best_val_acc,
        total_time,
    )

    # Restore best model (unwrap to load raw state dict)
    if best_state is not None:
        accelerator.unwrap_model(model).load_state_dict(best_state["model"])
        accelerator.unwrap_model(classifier).load_state_dict(best_state["classifier"])

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def extract_hidden_states(
    model: GRUExtractor,
    sequences: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Extract final-layer hidden states for a batch of sequences.

    When ``mean`` and ``std`` are provided, the sequences are standardized
    using the same per-feature statistics that were computed during training.
    This ensures the model receives identically scaled input at inference
    time.

    Args:
        model: Trained GRU extractor used for forward passes.
        sequences: Input array with shape ``(n_samples, seq_len, input_size)``.
        batch_size: Number of samples per inference batch.
        device: Computation device. If ``None``, CUDA is used when available,
            otherwise CPU.
        mean: Per-feature mean with shape broadcastable to ``sequences``
            (typically ``(1, 1, n_features)``).  When ``None``, no
            standardization is applied.
        std: Per-feature standard deviation with the same shape convention
            as ``mean``.  Must be provided when ``mean`` is provided.

    Returns:
        Array of shape ``(n_samples, hidden_size)`` containing one hidden
        state vector per input sequence.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Apply the same standardization used during training
    data = sequences.copy()
    if mean is not None and std is not None:
        data = (data - mean) / std

    all_sequences = torch.from_numpy(data).float()

    hidden_states: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(all_sequences), batch_size):
            batch = all_sequences[i : i + batch_size].to(device)
            hidden = model(batch)
            hidden_states.append(hidden.cpu().numpy())

    return np.concatenate(hidden_states, axis=0)


def predict_gru_proba(
    model: GRUExtractor,
    classifier: nn.Linear,
    sequences: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Predict class probabilities from a trained GRU backbone + classifier.

    Args:
        model: Trained GRU backbone.
        classifier: Classification head trained on top of the GRU hidden state.
        sequences: Input array with shape ``(n_samples, seq_len, input_size)``.
        batch_size: Number of samples per inference batch.
        device: Computation device. If ``None``, CUDA is used when available,
            otherwise CPU.
        mean: Optional training-set feature mean for standardization.
        std: Optional training-set feature std for standardization.

    Returns:
        Array of shape ``(n_samples, n_classes)`` with softmax probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    classifier.eval()

    data = sequences.copy()
    if mean is not None and std is not None:
        data = (data - mean) / std

    all_sequences = torch.from_numpy(data).float()
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(all_sequences), batch_size):
            batch = all_sequences[i : i + batch_size].to(device)
            hidden = model(batch)
            logits = classifier(hidden)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(probabilities, axis=0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_gru_model(
    model: GRUExtractor,
    config: Config,
    path: str | Path,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    classifier: nn.Linear | None = None,
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
            the GRU backbone for stacking inference.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gru_cfg = config.gru
    raw_model = getattr(model, "_orig_mod", model)

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
    }
    if classifier is not None:
        raw_classifier = getattr(classifier, "_orig_mod", classifier)
        checkpoint["classifier_state_dict"] = raw_classifier.state_dict()
        checkpoint["num_classes"] = raw_classifier.out_features
    if mean is not None:
        checkpoint["mean"] = mean
    if std is not None:
        checkpoint["std"] = std
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
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
