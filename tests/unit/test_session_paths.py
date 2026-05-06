"""Tests for centralized session path wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.shared.config import Config
from thesis.shared.session_paths import (
    configure_session_paths,
    load_config_for_session,
)


def test_configure_session_paths_sets_model_gru_predictions_backtest_report(
    tmp_path: Path,
) -> None:
    cfg = Config()
    session = tmp_path / "XAUUSD_1H_20200101_120000"
    configure_session_paths(cfg, session)

    assert cfg.paths.session_dir == str(session)
    assert cfg.paths.model == str(session / "models" / "lightgbm_model.pkl")
    assert cfg.paths.gru_model == str(session / "models" / "gru_model.pt")
    assert cfg.paths.predictions == str(
        session / "predictions" / "final_predictions.parquet"
    )
    assert cfg.paths.backtest_results == str(
        session / "backtest" / "backtest_results.json"
    )
    assert cfg.paths.report == str(session / "reports" / "thesis_report.md")


@pytest.mark.unit
def test_load_config_for_session_without_snapshot_uses_repo_toml(
    tmp_path: Path,
) -> None:
    """When no snapshot exists, load base ``config.toml`` from repo root."""
    session = tmp_path / "empty_session"
    session.mkdir()
    # No config/ under session — falls back to repo config.toml if present
    cfg = load_config_for_session(session, base_config_path="config.toml")
    assert cfg.paths.session_dir == str(session)
    assert "models" in cfg.paths.model


@pytest.mark.unit
def test_load_config_for_session_prefers_snapshot_over_base(tmp_path: Path) -> None:
    """When ``config/config_snapshot.toml`` exists, merge its sections then apply paths."""
    session = tmp_path / "sess_with_snap"
    (session / "config").mkdir(parents=True)
    snap = session / "config" / "config_snapshot.toml"
    snap.write_text(
        "[workflow]\nrandom_seed = 424242\n",
        encoding="utf-8",
    )
    cfg = load_config_for_session(session, base_config_path="config.toml")
    assert cfg.workflow.random_seed == 424242
    assert cfg.paths.session_dir == str(session)
