"""Dataset and sequence preparation utilities for GRU training.

Provides sliding-window construction, input validation, label extraction,
and the PyTorch Dataset wrapper used by the GRU training loop.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from thesis.shared.constants import STD_EPS

logger = logging.getLogger("thesis.gru")


# ---------------------------------------------------------------------------
# Dataset
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
        sample_weights: np.ndarray | None = None,
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
            sample_weights: Optional 1D array of per-sample training weights.
            mean: Optional per-feature mean array (shape ``(1, 1, n_features)``)
                from the training set.  Computed from ``sequences`` when ``None``.
            std: Optional per-feature std array (shape ``(1, 1, n_features)``)
                from the training set.  Computed from ``sequences`` when ``None``.
        """
        # Per-feature standardization: use provided stats or compute from data
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
        else:
            self.mean = sequences.mean(axis=(0, 1), keepdims=True)
            self.std = sequences.std(axis=(0, 1), keepdims=True) + STD_EPS
        standardized = (sequences - self.mean) / self.std
        self.sequences = torch.from_numpy(standardized.copy()).float()
        if labels is not None:
            if labels.dtype.kind == "f":
                self.labels = torch.from_numpy(labels.copy()).float()
            else:
                self.labels = torch.from_numpy(labels.copy()).long()
        else:
            self.labels = None
        self.sample_weights = (
            torch.from_numpy(sample_weights.copy()).float()
            if sample_weights is not None
            else None
        )

    def __len__(self) -> int:
        """Return the number of available samples.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieve a single sequence sample.

        Args:
            idx: Sample index.

        Returns:
            A tuple of ``(sequence, label)`` where ``label`` is ``None`` when
            labels were not provided.
        """
        if self.labels is not None:
            if self.sample_weights is not None:
                return self.sequences[idx], self.labels[idx], self.sample_weights[idx]
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx], None


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------


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


def _extract_sample_weights(
    df: pl.DataFrame, sequence_length: int
) -> np.ndarray | None:
    """Extract average-uniqueness weights aligned to sequence end indices."""
    if "sample_weight" not in df.columns:
        return None
    return df["sample_weight"].to_numpy()[sequence_length - 1 :].astype(np.float32)


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
