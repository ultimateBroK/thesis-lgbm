"""Tests for data download script.

Tests output directory derivation and config integration.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import pytest


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    from thesis.shared.config import (
        Config,
        DataConfig,
        PathsConfig,
        SplittingConfig,
        FeaturesConfig,
        LabelsConfig,
        LGBMConfig,
        WorkflowConfig,
        BacktestConfig,
    )

    cfg = Config(
        data=DataConfig(
            symbol="XAUUSD",
            timeframe="1H",
            market_tz="America/New_York",
            start_date="2018-01-01",
            end_date="2026-04-30",
            tick_size=0.01,
            contract_size=100,
            symbol_download="XAUUSD",
            asset_class="fx",
            download_concurrency=20,
            download_max_retries=7,
            download_force=False,
            download_skip_current_month=True,
        ),
        splitting=SplittingConfig(),
        features=FeaturesConfig(),
        labels=LabelsConfig(),
        model=LGBMConfig(),
        backtest=BacktestConfig(),
        workflow=WorkflowConfig(),
        paths=PathsConfig(data_raw="data/raw/XAUUSD"),
    )
    return cfg


def test_download_max_retries_in_config(mock_config):
    """Test that DataConfig has download_max_retries attribute."""
    assert hasattr(mock_config.data, "download_max_retries")
    assert mock_config.data.download_max_retries == 7


def test_output_dir_derived_from_instrument(mock_config, tmp_path):
    """Test that output_dir is derived from instrument when not explicitly provided."""
    from data_download import run_download

    # Mock download_month to avoid actual network calls
    with patch("data_download.download_month") as mock_download:
        mock_download.return_value = (
            1000,
            0,
            0,
        )  # rows, missing_hours, confirmed_absent

        # Run with XAG/USD instrument but no explicit output_dir
        run_download(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
            instrument="XAG/USD",
            asset_class="fx",
            workers=1,
        )

        # Check that download_month was called
        assert mock_download.called

        # Get the output_dir from the call
        call_kwargs = mock_download.call_args
        output_dir = call_kwargs.kwargs.get("output_dir") or call_kwargs[0][2]

        # Verify output_dir contains XAGUSD not XAUUSD
        assert "XAGUSD" in str(output_dir) or "XAG" in str(output_dir)


def test_output_dir_respects_explicit_override(mock_config, tmp_path):
    """Test that explicit output_dir is respected even with different instrument."""
    from data_download import run_download

    custom_output = tmp_path / "custom_output"

    with patch("data_download.download_month") as mock_download:
        mock_download.return_value = (1000, 0, 0)

        run_download(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
            output_dir=custom_output,
            instrument="XAG/USD",
            asset_class="fx",
            workers=1,
        )

        call_kwargs = mock_download.call_args
        output_dir = call_kwargs.kwargs.get("output_dir") or call_kwargs[0][2]

        # Verify explicit output_dir was used
        assert output_dir == custom_output


def test_instrument_xagusd_creates_correct_directory():
    """Test that XAG/USD creates data/raw/XAGUSD directory."""
    from data_download import run_download

    with patch("data_download.download_month") as mock_download:
        mock_download.return_value = (1000, 0, 0)

        run_download(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
            instrument="XAG/USD",
            asset_class="fx",
            workers=1,
        )

        call_kwargs = mock_download.call_args
        output_dir = call_kwargs.kwargs.get("output_dir") or call_kwargs[0][2]

        # XAG/USD should become data/raw/XAGUSD
        expected_suffix = Path("data/raw/XAGUSD")
        assert str(output_dir).endswith(str(expected_suffix)) or "XAGUSD" in str(
            output_dir
        )


def test_instrument_xauusd_default_uses_config_path():
    """Test that default XAUUSD uses config.toml data_raw path."""
    from data_download import run_download

    with patch("data_download.download_month") as mock_download:
        mock_download.return_value = (1000, 0, 0)

        # Run without specifying instrument (should use config default XAUUSD)
        run_download(
            start_year=2024,
            start_month=1,
            end_year=2024,
            end_month=1,
            workers=1,
        )

        call_kwargs = mock_download.call_args
        output_dir = call_kwargs.kwargs.get("output_dir") or call_kwargs[0][2]

        # Default should use XAUUSD path
        assert "XAUUSD" in str(output_dir)
