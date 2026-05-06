"""Session-scoped artifact paths — single place for CLI and dashboard.

``config.toml`` defines global defaults; ``main.py`` and the Streamlit app
rewrite ``paths.session_dir``, model, predictions, backtest, and report for a
given session directory. This module centralizes that logic.
"""

from __future__ import annotations

from pathlib import Path

from thesis.shared.config import Config, load_config


def configure_session_paths(config: Config, session_dir: str | Path) -> None:
    """Point session-owned artifact paths at a session directory.

    Updates ``paths.session_dir``, ``paths.model``, ``paths.gru_model``,
    ``paths.predictions``, ``paths.backtest_results``, and ``paths.report``.

    Args:
        config: Loaded configuration (from repo root or snapshot TOML).
        session_dir: Timestamped session folder under ``results/``.
    """
    sd = Path(session_dir)
    config.paths.session_dir = str(sd)
    config.paths.model = str(sd / "models" / "lightgbm_model.pkl")
    config.paths.gru_model = str(sd / "models" / "gru_model.pt")
    config.paths.predictions = str(sd / "predictions" / "final_predictions.parquet")
    config.paths.backtest_results = str(sd / "backtest" / "backtest_results.json")
    config.paths.report = str(sd / "reports" / "thesis_report.md")


def load_config_for_session(
    session_dir: str | Path,
    *,
    base_config_path: str | Path = "config.toml",
) -> Config:
    """Load session config from a snapshot when available.

    Applies :func:`configure_session_paths` so model/predictions/report paths
    match the CLI resume behavior.

    Args:
        session_dir: Session directory (e.g. ``results/XAUUSD_1H_20260418_143052``).
        base_config_path: Fallback TOML when no ``config/config_snapshot.toml``.

    Returns:
        Config with session paths wired.
    """
    sd = Path(session_dir)
    snapshot = sd / "config" / "config_snapshot.toml"
    if snapshot.exists():
        config = load_config(snapshot)
    else:
        config = load_config(base_config_path)
    configure_session_paths(config, sd)
    return config
