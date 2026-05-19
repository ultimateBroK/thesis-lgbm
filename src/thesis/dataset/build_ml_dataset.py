"""Build ML-ready dataset by merging features with triple-barrier labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from thesis.shared.config import Config
from thesis.shared.constants import (
    LABEL_META_COLS,
    build_feature_output_cols,
    get_static_feature_cols,
)
from thesis.shared.utils import console

logger = logging.getLogger("thesis.dataset.build_ml_dataset")


def _load_parquet(path: Path, name: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    logger.info("Loading %s: %s", name.lower(), path)
    return pl.read_parquet(path)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Wrote %s", path)


def _join_features_labels(features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Timestamp join preferred.

    Positional concat when timestamps absent but lengths match.
    """
    if "timestamp" in features.columns and "timestamp" in labels.columns:
        return features.join(labels, on="timestamp", how="inner")
    if len(features) == len(labels):
        return pl.concat(
            [
                features,
                labels.drop([c for c in labels.columns if c in features.columns]),
            ],
            how="horizontal",
        )

    raise ValueError(
        f"Cannot join: features={len(features)} rows, labels={len(labels)} rows, "
        "and no shared timestamp column."
    )


def _drop_null_labels(df: pl.DataFrame) -> pl.DataFrame:
    if "label" not in df.columns:
        return df

    n_before = len(df)
    df = df.filter(pl.col("label").is_not_null())
    df = df.filter(~pl.col("label").is_nan())
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Dropped %d rows with NaN labels", n_dropped)
    return df


def _model_feature_cols(df: pl.DataFrame, config: Config) -> list[str]:
    return sorted(c for c in df.columns if c in set(get_static_feature_cols(config)))


def _label_distribution(df: pl.DataFrame) -> dict[str, int]:
    if "label" not in df.columns:
        return {}
    dist = df["label"].value_counts().sort("label")
    return {str(row["label"]): int(row["count"]) for row in dist.iter_rows(named=True)}


def _validate_ml_dataset(df: pl.DataFrame, config: Config) -> None:
    if df.is_empty():
        raise ValueError("ML dataset is empty after dropping NaN labels")

    expected = set(build_feature_output_cols(config) + LABEL_META_COLS)
    missing = expected - set(df.columns)
    if missing:
        logger.warning("Missing expected columns (non-fatal): %s", sorted(missing))


def _write_ml_artifacts(
    df: pl.DataFrame,
    out_path: Path,
    model_cols: list[str],
    dist: dict[str, int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info(
        "Saved ml_dataset: %s (%d rows, %d cols)", out_path, len(df), len(df.columns)
    )

    reports_dir = out_path.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if dist:
        _write_json(reports_dir / "label_distribution.json", dist)
        logger.info("Label distribution: %s", dist)
    _write_json(
        reports_dir / "feature_list.json",
        {"features": model_cols, "count": len(model_cols)},
    )
    logger.info("Feature columns (%d): %s", len(model_cols), model_cols)


def _print_summary(
    df: pl.DataFrame, model_cols: list[str], dist: dict[str, int]
) -> None:
    console.rule("ML Dataset Summary")
    console.print(f"  Rows: {len(df)}")
    console.print(f"  Columns: {len(df.columns)}")
    console.print(f"  Feature columns: {len(model_cols)}")
    if dist:
        console.print(f"  Label distribution: {dist}")


def build_ml_dataset(config: Config) -> None:
    """Merge features + labels → validated ML-ready parquet + metadata artifacts."""
    features_path = Path(config.paths.features)
    labels_path = Path(config.paths.labels)
    out_path = Path(config.paths.ml_dataset)

    features = _load_parquet(features_path, "Features")
    labels = _load_parquet(labels_path, "Labels")
    logger.info("Features: %d rows, %d cols", len(features), len(features.columns))
    logger.info("Labels:   %d rows, %d cols", len(labels), len(labels.columns))

    df = _join_features_labels(features, labels)
    df = _drop_null_labels(df)
    _validate_ml_dataset(df, config)

    model_cols = _model_feature_cols(df, config)
    dist = _label_distribution(df)
    _write_ml_artifacts(df, out_path, model_cols, dist)
    _print_summary(df, model_cols, dist)
