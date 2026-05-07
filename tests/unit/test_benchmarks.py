"""Tests for reporting benchmarks — naive strategies and model label helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from thesis.shared.config import Config
from thesis.stage_6_reporting.benchmarks import (
    _annualized_sharpe,
    _compute_random_strategy,
    _equity_curve_from_bar_returns,
    _load_close_prices_for_benchmark,
    _max_drawdown_pct,
    _model_label,
    compute_benchmark_comparison,
)


# ---------------------------------------------------------------------------
# _model_label
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelLabel:
    def test_lgbm(self) -> None:
        config = Config()
        config.model.architecture = "lgbm"
        assert _model_label(config) == "LightGBM"

    def test_hybrid(self) -> None:
        config = Config()
        config.model.architecture = "hybrid"
        assert _model_label(config) == "Hybrid GRU + LightGBM"

    def test_gru_only(self) -> None:
        config = Config()
        config.model.architecture = "gru"
        assert _model_label(config) == "GRU-only"

    def test_unknown(self) -> None:
        config = Config()
        config.model.architecture = "gru_only"
        assert _model_label(config) == "Gru_Only Model"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnnualizedSharpe:
    def test_positive_returns(self) -> None:
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
        sharpe = _annualized_sharpe(returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_zero_std(self) -> None:
        returns = np.array([0.01, 0.01, 0.01])
        assert _annualized_sharpe(returns) == 0.0

    def test_empty(self) -> None:
        returns = np.array([])
        assert _annualized_sharpe(returns) == 0.0


@pytest.mark.unit
class TestMaxDrawdownPct:
    def test_no_drawdown(self) -> None:
        equity = np.array([100, 110, 120, 130])
        assert _max_drawdown_pct(equity) == 0.0

    def test_with_drawdown(self) -> None:
        equity = np.array([100, 120, 90, 110])
        dd = _max_drawdown_pct(equity)
        assert dd > 0
        assert dd == pytest.approx(25.0, abs=0.1)  # 90/120 = 75%, dd = 25%

    def test_single_point(self) -> None:
        assert _max_drawdown_pct(np.array([100])) == 0.0


@pytest.mark.unit
class TestBuildEquityCurve:
    def test_basic(self) -> None:
        returns = np.array([0.1, -0.05, 0.15])
        equity = _equity_curve_from_bar_returns(returns, 1000)
        assert len(equity) == 4
        assert equity[0] == 1000
        assert equity[1] == pytest.approx(1100)
        assert equity[2] == pytest.approx(1045)
        assert equity[3] == pytest.approx(1201.75)


@pytest.mark.unit
class TestComputeRandomStrategy:
    def test_returns_expected_keys(self) -> None:
        returns = np.array([0.01, -0.02, 0.005, -0.01, 0.02] * 10)
        result = _compute_random_strategy(returns, 10000, 100, seed=42)
        assert "return_pct" in result
        assert "sharpe" in result
        assert "max_dd_pct" in result
        assert "win_rate_pct" in result
        assert "num_trades" in result

    def test_deterministic_with_seed(self) -> None:
        returns = np.random.randn(100) * 0.01
        r1 = _compute_random_strategy(returns, 10000, 100, seed=42)
        r2 = _compute_random_strategy(returns, 10000, 100, seed=42)
        assert r1["return_pct"] == r2["return_pct"]


# ---------------------------------------------------------------------------
# _load_close_prices_for_benchmark
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadClosePricesForBenchmark:
    def test_lgbm_mode_loads_test_data(self, tmp_path) -> None:
        config = Config()
        config.validation.method = "static"

        test_path = tmp_path / "test.parquet"
        pl.DataFrame({"close": [100.0, 101.0, 102.0]}).write_parquet(test_path)

        result = _load_close_prices_for_benchmark(test_path, {}, config)
        assert result is not None
        assert len(result) == 3

    def test_walkforward_uses_ohlcv_fallback(self, tmp_path) -> None:
        config = Config()
        config.validation.method = "sliding"
        config.paths.ohlcv = str(tmp_path / "ohlcv.parquet")

        ohlcv_path = tmp_path / "ohlcv.parquet"
        ts = pl.Series("timestamp", [1704067200000, 1704153600000, 1704240000000]).cast(
            pl.Datetime("ms")
        )
        pl.DataFrame({"timestamp": ts, "close": [100.0, 101.0, 102.0]}).write_parquet(
            ohlcv_path
        )

        result = _load_close_prices_for_benchmark(
            tmp_path / "nonexistent.parquet",
            {},
            config,
        )
        assert result is not None
        assert len(result) == 3

    def test_no_data_returns_none(self, tmp_path) -> None:
        config = Config()
        config.validation.method = "sliding"
        config.paths.ohlcv = str(tmp_path / "nonexistent.parquet")
        result = _load_close_prices_for_benchmark(
            tmp_path / "nonexistent.parquet", {}, config
        )
        assert result is None


# ---------------------------------------------------------------------------
# compute_benchmark_comparison
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeBenchmarkComparison:
    def test_returns_four_strategies(self, tmp_path) -> None:
        config = Config()
        config.validation.method = "static"

        test_path = tmp_path / "test.parquet"
        close_prices = np.cumsum(np.random.randn(50) * 0.5 + 100)
        pl.DataFrame({"close": close_prices}).write_parquet(test_path)

        result = compute_benchmark_comparison(
            test_path,
            {
                "return_pct": 15.0,
                "sharpe_ratio": 1.2,
                "max_drawdown_pct": -8.0,
                "win_rate_pct": 55.0,
                "num_trades": 100,
            },
            config,
        )
        assert len(result) == 4
        strategies = [r["strategy"] for r in result]
        assert "Buy & Hold" in strategies
        assert "Always Long" in strategies
        assert "Random Signal" in strategies

    def test_no_data_returns_empty(self, tmp_path) -> None:
        config = Config()
        config.validation.method = "static"
        config.paths.ohlcv = str(tmp_path / "nonexistent.parquet")

        result = compute_benchmark_comparison(
            tmp_path / "nonexistent.parquet", {}, config
        )
        assert result == []
