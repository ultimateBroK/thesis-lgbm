"""Single source of truth for feature column lists across pipeline stages."""

from __future__ import annotations

# Type note: the *config* parameter is intentionally left untyped throughout this
# module.  Importing ``Config`` from ``thesis.config`` would create a circular
# dependency because that module may transitively import pipeline helpers that
# depend on the column lists defined here.

# ---------------------------------------------------------------------------
# Constants (no config needed)
# ---------------------------------------------------------------------------

OHLCV_RAW_COLS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]
"""Core OHLCV columns present in the raw data source."""

OHLCV_OPTIONAL_COLS: list[str] = ["tick_count", "avg_spread"]
"""Optional columns that may accompany OHLCV data."""

LABEL_META_COLS: list[str] = [
    "label",
    "upper_barrier",
    "lower_barrier",
    "touched_bar",
    "event_end",
    "sample_weight",
]
"""Metadata columns produced by the triple-barrier labelling stage."""


# ---------------------------------------------------------------------------
# Config-driven helpers
# ---------------------------------------------------------------------------


def get_static_feature_cols(config) -> list[str]:
    """Return the static (non-sequential) feature columns from config."""
    return list(config.features.static_feature_cols)


def get_gru_feature_cols(config) -> list[str]:
    """Return the GRU-specific feature columns from config."""
    return list(config.gru.feature_cols)


def get_label_helper_cols(config) -> list[str]:
    """Return helper columns used during label construction (e.g. ATR)."""
    return [f"atr_{config.features.atr_period}"]


def build_feature_output_cols(config) -> list[str]:
    """All columns that ``features.parquet`` must contain.

    Combines OHLCV raw columns, label helpers, static features, and GRU
    features into a sorted, deduplicated list.
    """
    return sorted(
        set(
            OHLCV_RAW_COLS
            + get_label_helper_cols(config)
            + get_static_feature_cols(config)
            + get_gru_feature_cols(config)
        )
    )


def build_label_output_cols(config) -> list[str]:
    """All columns that ``labels.parquet`` must contain.

    Superset of feature output columns plus label metadata columns.
    """
    return sorted(set(build_feature_output_cols(config) + LABEL_META_COLS))


def build_exclude_cols(config) -> frozenset[str]:
    """Columns excluded from model training — the minimal non-feature set.

    These columns are either identifiers (timestamp), raw price/volume data,
    labelling artefacts, or GRU sequence inputs that are not valid static
    features for the LightGBM base model.
    """
    return frozenset(
        OHLCV_RAW_COLS
        + OHLCV_OPTIONAL_COLS
        + get_label_helper_cols(config)
        + LABEL_META_COLS
        + ["log_returns"]  # GRU sequence input, not static LightGBM feature
    )
