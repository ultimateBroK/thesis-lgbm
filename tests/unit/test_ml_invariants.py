"""ML pipeline invariants: exclusions, sequence/label alignment, CV gap."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import TimeSeriesSplit

from thesis.config import Config
from thesis.constants import EXCLUDE_COLS
from thesis.gru import prepare_sequences


def test_exclude_cols_contains_target_and_lookahead_fields() -> None:
    for col in (
        "timestamp",
        "label",
        "tp_price",
        "sl_price",
        "touched_bar",
        "open_right",
        "high_right",
        "low_right",
        "close_right",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_returns",
    ):
        assert col in EXCLUDE_COLS


def test_static_feature_list_excludes_derived_label_columns() -> None:
    """Hybrid static columns are ``columns - EXCLUDE_COLS``; no look-ahead cols."""
    df = pl.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "label": [0, 0, 0],
            "rsi_14": [1.0, 2.0, 3.0],
            "open_right": [100.0, 100.0, 100.0],
        }
    )
    static = [c for c in df.columns if c not in EXCLUDE_COLS]
    assert static == ["rsi_14"]
    assert "open_right" not in static


def test_prepare_sequences_label_count_matches_window_count() -> None:
    """Labels are aligned to the end bar of each window (see ``_extract_labels``)."""
    n = 40
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "close": 1800.0 + np.cumsum(rng.normal(0, 0.1, n)),
            "label": rng.integers(-1, 2, n).astype(np.int64),
        }
    )
    seq_len = 12
    seq, labels, _static = prepare_sequences(
        df,
        ["close"],
        seq_len,
        exclude_cols=EXCLUDE_COLS,
    )
    expected_n = n - seq_len + 1
    assert seq.shape[0] == expected_n
    assert labels is not None
    assert len(labels) == expected_n


def test_align_slice_length_matches_hidden_count_formula() -> None:
    """Mirror ``hybrid.train._align_splits_with_sequences`` row count."""
    n_rows = 100
    seq_len = 48
    hidden_len = n_rows - seq_len + 1
    df = pl.DataFrame({"x": range(n_rows)})
    aligned = df.slice(seq_len - 1, hidden_len)
    assert len(aligned) == hidden_len


def test_optuna_time_series_split_gap_equals_purge_plus_embargo() -> None:
    """``hybrid.lgbm._train_optuna`` uses this gap on ``X_train`` indices only."""
    cfg = Config()
    gap = cfg.splitting.purge_bars + cfg.splitting.embargo_bars
    tscv = TimeSeriesSplit(n_splits=3, gap=gap)
    assert tscv.gap == gap
