"""Smoke tests for chart builder module."""

import numpy as np
import polars as pl
import pytest

pytest.importorskip("pyecharts")
from pyecharts.charts import Bar, Grid, HeatMap, Line, Pie, Scatter, Tab

from thesis.charts import (
    EXCLUDED_FEATURE_COLS,
    build_candlestick_chart,
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_correlation_heatmap,
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_feature_distributions_chart,
    build_feature_importance_chart,
    build_label_distribution_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
)
from thesis.shared.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Minimal OHLCV data (5 rows)."""
    return pl.DataFrame(
        {
            "timestamp": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "open": [2000.0, 2010.0, 2005.0, 2015.0, 2008.0],
            "high": [2015.0, 2020.0, 2010.0, 2025.0, 2018.0],
            "low": [1995.0, 2005.0, 2000.0, 2010.0, 2000.0],
            "close": [2010.0, 2005.0, 2008.0, 2012.0, 2015.0],
            "volume": [1000, 1200, 900, 1100, 1300],
        }
    )


@pytest.fixture
def sample_features() -> pl.DataFrame:
    """Minimal features data."""
    return pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "rsi_14": [55.0, 60.0, 45.0],
            "atr_14": [5.0, 5.5, 4.8],
            "macd": [0.5, -0.3, 0.1],
            "bb_width": [10.0, 12.0, 11.0],
        }
    )


@pytest.fixture
def sample_labels() -> pl.DataFrame:
    """Minimal labels data."""
    return pl.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "label": [1, -1, 0, 1],
        }
    )


@pytest.fixture
def sample_preds() -> pl.DataFrame:
    """Minimal prediction data."""
    return pl.DataFrame(
        {
            "true_label": [1, -1, 0, 1, -1],
            "pred_label": [1, -1, 0, -1, -1],
            "pred_proba_class_minus1": [0.1, 0.8, 0.2, 0.4, 0.7],
            "pred_proba_class_0": [0.2, 0.1, 0.7, 0.3, 0.1],
            "pred_proba_class_1": [0.7, 0.1, 0.1, 0.3, 0.2],
        }
    )


@pytest.fixture
def sample_trades() -> list[dict]:
    """Minimal trade list (35 trades for rolling Sharpe)."""
    trades = []
    for i in range(35):
        pnl = 100.0 if i % 3 != 0 else -50.0
        trades.append(
            {
                "pnl": pnl,
                "entry_time": f"2024-01-{i + 1:02d}T10:00:00Z",
                "exit_time": f"2024-01-{i + 1:02d}T14:00:00Z",
            }
        )
    return trades


@pytest.fixture
def sample_config() -> Config:
    """Minimal config."""
    return Config()


@pytest.fixture
def sample_fi() -> dict[str, float]:
    """Minimal feature importance."""
    return {
        "rsi_14": 0.15,
        "atr_14": 0.12,
        "gru_0": 0.10,
        "macd": 0.08,
        "gru_1": 0.07,
        "bb_width": 0.05,
    }


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_excluded_feature_cols_is_frozenset() -> None:
    assert isinstance(EXCLUDED_FEATURE_COLS, frozenset)
    assert "timestamp" in EXCLUDED_FEATURE_COLS
    assert "label" in EXCLUDED_FEATURE_COLS
    assert "rsi_14" not in EXCLUDED_FEATURE_COLS


# ---------------------------------------------------------------------------
# Test: Data Exploration Charts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_candlestick_chart(
    sample_ohlcv: pl.DataFrame, sample_config: Config
) -> None:
    chart, info = build_candlestick_chart(sample_ohlcv, sample_config)
    assert isinstance(chart, Grid)
    assert isinstance(info, dict)
    assert "total_bars" in info
    assert "displayed_bars" in info
    assert "downsampled" in info
    opts = chart.dump_options()
    assert isinstance(opts, str)
    assert "candlestick" in opts or "kline" in opts.lower() or "series" in opts


@pytest.mark.unit
def test_build_candlestick_downsamples(sample_config: Config) -> None:
    """Large dataset triggers downsampling."""
    n = 5000
    df = pl.DataFrame(
        {
            "timestamp": [
                f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00" for i in range(n)
            ],
            "open": [1800.0 + i * 0.01 for i in range(n)],
            "high": [1802.0 + i * 0.01 for i in range(n)],
            "low": [1798.0 + i * 0.01 for i in range(n)],
            "close": [1801.0 + i * 0.01 for i in range(n)],
            "volume": [100.0] * n,
        }
    )
    chart, info = build_candlestick_chart(df, sample_config, max_bars=1000)
    assert isinstance(chart, Grid)
    assert info["downsampled"] is True
    assert info["total_bars"] == n
    assert info["displayed_bars"] <= 1000


@pytest.mark.unit
def test_build_correlation_heatmap(sample_features: pl.DataFrame) -> None:
    chart = build_correlation_heatmap(sample_features)
    assert isinstance(chart, HeatMap)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_label_distribution_chart(sample_labels: pl.DataFrame) -> None:
    chart = build_label_distribution_chart(sample_labels)
    assert isinstance(chart, Pie)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_feature_distributions_chart(sample_features: pl.DataFrame) -> None:
    chart = build_feature_distributions_chart(sample_features)
    assert isinstance(chart, Tab)


# ---------------------------------------------------------------------------
# Test: Model Performance Charts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_confusion_matrix_chart() -> None:
    true = np.array([1, -1, 0, 1, -1])
    pred = np.array([1, -1, 0, -1, -1])
    chart = build_confusion_matrix_chart(true, pred)
    assert isinstance(chart, HeatMap)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_confidence_distribution_chart(sample_preds: pl.DataFrame) -> None:
    chart = build_confidence_distribution_chart(sample_preds)
    assert isinstance(chart, Bar)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_feature_importance_chart(sample_fi: dict) -> None:
    chart = build_feature_importance_chart(sample_fi, top_n=6)
    assert isinstance(chart, Bar)
    opts = chart.dump_options()
    assert isinstance(opts, str)


# ---------------------------------------------------------------------------
# Test: Backtest Charts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_equity_drawdown_chart(sample_trades: list) -> None:
    metrics = {"total_trades": 35, "total_return_pct": 150.0}
    chart = build_equity_drawdown_chart(sample_trades, metrics)
    assert isinstance(chart, Grid)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_pnl_histogram_chart(sample_trades: list) -> None:
    metrics = {"avg_win": 100.0, "avg_loss": -50.0}
    chart = build_pnl_histogram_chart(sample_trades, metrics)
    assert isinstance(chart, Bar)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_monthly_returns_heatmap(sample_trades: list) -> None:
    chart = build_monthly_returns_heatmap(sample_trades)
    assert isinstance(chart, HeatMap)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_rolling_sharpe_chart(sample_trades: list) -> None:
    chart = build_rolling_sharpe_chart(sample_trades, window=10)
    assert isinstance(chart, Line)
    opts = chart.dump_options()
    assert isinstance(opts, str)


@pytest.mark.unit
def test_build_duration_pnl_scatter(sample_trades: list) -> None:
    chart = build_duration_pnl_scatter(sample_trades)
    assert isinstance(chart, Scatter)
    opts = chart.dump_options()
    assert isinstance(opts, str)


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_candlestick_empty(sample_config: Config) -> None:
    """Empty DataFrame should not crash (Grid is still returned)."""
    df = pl.DataFrame(
        {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        },
        schema={
            "timestamp": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    chart, info = build_candlestick_chart(df, sample_config)
    assert isinstance(chart, Grid)
    assert isinstance(info, dict)
    assert info["total_bars"] == 0


@pytest.mark.unit
def test_build_rolling_sharpe_too_few_trades() -> None:
    """Fewer trades than window should return empty Line."""
    trades = [
        {
            "pnl": 100.0,
            "entry_time": "2024-01-01T10:00:00Z",
            "exit_time": "2024-01-01T14:00:00Z",
        }
    ]
    chart = build_rolling_sharpe_chart(trades, window=30)
    assert isinstance(chart, Line)


@pytest.mark.unit
def test_build_equity_drawdown_empty() -> None:
    """Empty trades should return empty Grid."""
    chart = build_equity_drawdown_chart([], {})
    assert isinstance(chart, Grid)


# ---------------------------------------------------------------------------
# load_session_data from charts.loader
# ---------------------------------------------------------------------------

from thesis.charts.loader import load_session_data


@pytest.mark.unit
class TestLoadSessionData:
    def test_returns_dict_with_missing_files(self, tmp_path) -> None:
        config = Config()
        config.paths.session_dir = str(tmp_path)
        config.paths.ohlcv = str(tmp_path / "ohlcv.parquet")
        config.paths.features = str(tmp_path / "features.parquet")
        config.paths.test_data = str(tmp_path / "test.parquet")
        config.paths.labels = str(tmp_path / "labels.parquet")
        config.paths.predictions = str(tmp_path / "preds.parquet")
        config.paths.backtest_results = str(tmp_path / "bt.json")

        result = load_session_data(config)
        assert isinstance(result, dict)
        assert result["ohlcv"] is None
        assert result["predictions"] is None
        assert result["backtest_results"] is None
        assert result["trades"] == []
        assert result["metrics"] == {}

    def test_loads_existing_files(self, tmp_path) -> None:
        config = Config()
        config.paths.session_dir = str(tmp_path)
        config.paths.ohlcv = str(tmp_path / "ohlcv.parquet")
        config.paths.features = str(tmp_path / "features.parquet")
        config.paths.test_data = str(tmp_path / "test.parquet")
        config.paths.labels = str(tmp_path / "labels.parquet")
        config.paths.predictions = str(tmp_path / "preds.parquet")
        config.paths.backtest_results = str(tmp_path / "bt.json")

        # Create test parquet files
        df = pl.DataFrame({"a": [1, 2, 3]})
        df.write_parquet(tmp_path / "ohlcv.parquet")
        df.write_parquet(tmp_path / "features.parquet")
        df.write_parquet(tmp_path / "test.parquet")
        df.write_parquet(tmp_path / "labels.parquet")

        result = load_session_data(config)
        assert result["ohlcv"] is not None
        assert result["features"] is not None
        assert result["test"] is not None
        assert result["labels"] is not None

    def test_loads_backtest_results_json(self, tmp_path) -> None:
        config = Config()
        config.paths.session_dir = str(tmp_path)
        config.paths.ohlcv = str(tmp_path / "nonexistent.parquet")
        config.paths.features = str(tmp_path / "nonexistent.parquet")
        config.paths.test_data = str(tmp_path / "nonexistent.parquet")
        config.paths.labels = str(tmp_path / "nonexistent.parquet")
        config.paths.predictions = str(tmp_path / "preds.parquet")
        config.paths.backtest_results = str(tmp_path / "bt.json")

        bt_dir = tmp_path / "backtest"
        bt_dir.mkdir()
        bt_data = {"metrics": {"total_return_pct": 10.0}, "trades": [{"pnl": 100}]}
        import json

        (bt_dir / "backtest_results.json").write_text(json.dumps(bt_data))

        result = load_session_data(config)
        assert result["metrics"]["total_return_pct"] == 10.0
        assert len(result["trades"]) == 1

    def test_session_dir_predictions(self, tmp_path) -> None:
        config = Config()
        config.paths.session_dir = str(tmp_path)
        config.paths.ohlcv = str(tmp_path / "nonexistent.parquet")
        config.paths.features = str(tmp_path / "nonexistent.parquet")
        config.paths.test_data = str(tmp_path / "nonexistent.parquet")
        config.paths.labels = str(tmp_path / "nonexistent.parquet")
        config.paths.backtest_results = str(tmp_path / "nonexistent.json")

        preds_dir = tmp_path / "predictions"
        preds_dir.mkdir()
        preds_path = preds_dir / "final_predictions.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(preds_path)

        result = load_session_data(config)
        assert result["predictions"] is not None

    def test_feature_importance_json(self, tmp_path) -> None:
        config = Config()
        config.paths.session_dir = str(tmp_path)
        config.paths.ohlcv = str(tmp_path / "nonexistent.parquet")
        config.paths.features = str(tmp_path / "nonexistent.parquet")
        config.paths.test_data = str(tmp_path / "nonexistent.parquet")
        config.paths.labels = str(tmp_path / "nonexistent.parquet")
        config.paths.predictions = str(tmp_path / "nonexistent.parquet")
        config.paths.backtest_results = str(tmp_path / "nonexistent.json")

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        import json

        (reports_dir / "feature_importance.json").write_text(json.dumps({"rsi": 10}))

        result = load_session_data(config)
        assert result["feature_importance"]["rsi"] == 10
