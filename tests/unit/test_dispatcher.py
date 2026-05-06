"""Tests for walk-forward dispatcher — architecture routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.dispatcher import _run_walk_forward


@pytest.mark.unit
class TestDispatcher:
    @patch("thesis.stage_4_training.walk_forward.dispatcher._run_walk_forward_static")
    def test_static_architecture_routes_correctly(self, mock_static) -> None:
        config = Config()
        config.model.architecture = "static"
        _run_walk_forward(config)
        mock_static.assert_called_once()

    @patch("thesis.stage_4_training.walk_forward.dispatcher._run_walk_forward_hybrid")
    def test_hybrid_architecture_routes_correctly(self, mock_hybrid) -> None:
        config = Config()
        config.model.architecture = "hybrid"
        _run_walk_forward(config)
        mock_hybrid.assert_called_once()

    def test_unsupported_architecture_raises(self) -> None:
        config = Config()
        config.model.architecture = "unknown_arch"
        with pytest.raises(ValueError, match="Unsupported model.architecture"):
            _run_walk_forward(config)
