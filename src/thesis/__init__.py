"""Thesis ML pipeline — top-level public API surface.

This module re-exports the most commonly used symbols so that callers can use
``import thesis; thesis.run_pipeline(...)`` without reaching into sub-packages.
"""

from thesis.pipeline import run_pipeline
from thesis.shared.config import Config, get_config, load_config
from thesis.stage_1_data import prepare_data
from thesis.stage_2_features import generate_features
from thesis.stage_3_labels import generate_labels
from thesis.stage_4_training import WalkForwardWindow, generate_windows, train_model
from thesis.stage_5_backtest import run_backtest
from thesis.stage_6_reporting import generate_report

__all__ = [
    "Config",
    "get_config",
    "generate_features",
    "generate_labels",
    "generate_report",
    "generate_windows",
    "load_config",
    "prepare_data",
    "run_backtest",
    "run_pipeline",
    "train_model",
    "WalkForwardWindow",
]
