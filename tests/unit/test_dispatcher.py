"""Tests for walk-forward dispatcher — architecture routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.dispatcher import train_walk_forward


@pytest.mark.unit
class TestDispatcher:
    @patch("thesis.stage_4_training.walk_forward.dispatcher.train_lgbm_walk_forward")
    def test_lgbm_architecture_routes_correctly(self, mock_lgbm) -> None:
        config = Config()
        config.model.lgbm_expanded_features = False
        train_walk_forward(config)
        mock_lgbm.assert_called_once_with(config, expanded_features=False)

    def test_unsupported_architecture_raises(self) -> None:
        config = Config()
        config.model.architecture = "unknown_arch"
        with pytest.raises(ValueError, match="This thesis version only supports LightGBM architecture"):
            train_walk_forward(config)