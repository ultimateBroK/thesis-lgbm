"""Unit tests for _data_quality — data quality reporting functions."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from thesis.stage_6_reporting.data_quality import (
    compute_data_quality_report,
    compute_label_distribution,
    compute_missing_bar_stats,
    compute_ohlcv_consistency,
    compute_outlier_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(
    n: int = 100,
    start: datetime = datetime(2023, 1, 1),
    interval_hours: int = 1,
) -> pl.DataFrame:
    """Generate a simple OHLCV DataFrame with valid relationships."""
    timestamps = [start + timedelta(hours=interval_hours * i) for i in range(n)]
    close = np.cumsum(np.random.randn(n) * 0.5) + 100.0
    open_ = close + np.random.randn(n) * 0.1
    high = np.maximum(open_, close) + abs(np.random.randn(n)) * 0.2
    low = np.minimum(open_, close) - abs(np.random.randn(n)) * 0.2
    volume = np.abs(np.random.randn(n)) * 1000 + 100
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


# ---------------------------------------------------------------------------
# compute_ohlcv_consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOhlcvConsistency:
    def test_valid_data_consistent(self) -> None:
        df = _make_ohlcv_df(50)
        result = compute_ohlcv_consistency(df)
        assert result["is_consistent"] is True
        assert result["ohlc_violations"] == 0
        assert result["price_negative_count"] == 0

    def test_violation_detected(self) -> None:
        df = _make_ohlcv_df(10)
        # Force a violation: set low > high on first row
        df = (
            df.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 0)
                .then(pl.col("low") + 1000)  # low way above high
                .otherwise(pl.col("low"))
                .alias("low")
            )
            .drop("index")
        )
        result = compute_ohlcv_consistency(df)
        assert result["ohlc_violations"] > 0
        assert result["is_consistent"] is False

    def test_negative_price_detected(self) -> None:
        df = _make_ohlcv_df(10)
        df = (
            df.with_row_index()
            .with_columns(
                pl.when(pl.col("index") == 0)
                .then(-1.0)
                .otherwise(pl.col("close"))
                .alias("close")
            )
            .drop("index")
        )
        result = compute_ohlcv_consistency(df)
        assert result["price_negative_count"] > 0
        assert result["is_consistent"] is False


# ---------------------------------------------------------------------------
# compute_label_distribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLabelDistribution:
    def test_balanced_labels(self) -> None:
        labels = np.array([-1, -1, 0, 0, 1, 1])
        result = compute_label_distribution(labels)
        assert result["total"] == 6
        assert result["counts"] == {-1: 2, 0: 2, 1: 2}
        assert result["imbalance_ratio"] == 1.0
        for pct in result["percentages"].values():
            assert abs(pct - 33.33) < 0.1

    def test_imbalanced_labels(self) -> None:
        labels = np.array([-1] * 10 + [0] * 80 + [1] * 10)
        result = compute_label_distribution(labels)
        assert result["counts"][0] == 80
        assert result["imbalance_ratio"] == 8.0

    def test_single_class(self) -> None:
        labels = np.array([0, 0, 0])
        result = compute_label_distribution(labels)
        assert result["counts"][0] == 3
        assert result["imbalance_ratio"] == 0.0


# ---------------------------------------------------------------------------
# compute_outlier_returns
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOutlierReturns:
    def test_no_outliers_in_normal_data(self) -> None:
        np.random.seed(42)
        df = _make_ohlcv_df(200)
        result = compute_outlier_returns(df, z_threshold=5.0)
        assert result["outlier_count"] == 0
        assert result["outlier_ratio"] == 0.0

    def test_outlier_detected(self) -> None:
        df = _make_ohlcv_df(100)
        # Inject a massive price jump at row 50
        close = df["close"].to_list()
        close[50] = close[49] * 1.5  # 50% jump
        df = df.with_columns(pl.Series("close", close))
        result = compute_outlier_returns(df, z_threshold=4.0)
        assert result["outlier_count"] >= 1

    def test_empty_df(self) -> None:
        df = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        result = compute_outlier_returns(df)
        assert result["outlier_count"] == 0


# ---------------------------------------------------------------------------
# compute_missing_bar_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingBarStats:
    def test_no_gaps_regular_data(self) -> None:
        df = _make_ohlcv_df(50, interval_hours=1)
        result = compute_missing_bar_stats(df, "1h")
        assert result["total_bars"] == 50
        assert result["gaps_found"] == 0

    def test_gap_detected(self) -> None:
        start = datetime(2023, 1, 2, 0, 0)  # Monday
        timestamps = [start + timedelta(hours=i) for i in range(5)]
        # Insert a big gap between bar 2 and 3
        timestamps[3] = timestamps[2] + timedelta(hours=10)
        timestamps[4] = timestamps[3] + timedelta(hours=1)
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
            }
        )
        result = compute_missing_bar_stats(df, "1h")
        assert result["gaps_found"] >= 1

    def test_weekend_gap_uses_scalar_datetime_weekday(self) -> None:
        timestamps = [
            datetime(2023, 1, 6, 21, 0),  # Friday
            datetime(2023, 1, 9, 0, 0),  # Monday
            datetime(2023, 1, 9, 1, 0),
        ]
        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * 3,
                "high": [101.0] * 3,
                "low": [99.0] * 3,
                "close": [100.5] * 3,
            }
        )

        result = compute_missing_bar_stats(df, "1h")

        assert result["gaps_found"] == 1
        assert result["weekend_gaps"] == 1
        assert result["missing_ratio"] == 0.0


# ---------------------------------------------------------------------------
# compute_data_quality_report (integration of all checks)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeDataQualityReport:
    def test_returns_all_keys(self) -> None:
        df = _make_ohlcv_df(50)
        labels = np.array([-1, 0, 1] * 16 + [0, 1])
        result = compute_data_quality_report(df, labels)
        assert "ohlcv_consistency" in result
        assert "missing_bars" in result
        assert "outlier_returns" in result
        assert "label_distribution" in result
        assert "markdown" in result

    def test_without_labels(self) -> None:
        df = _make_ohlcv_df(20)
        result = compute_data_quality_report(df)
        assert "ohlcv_consistency" in result
        assert "label_distribution" not in result

    def test_markdown_rendered(self) -> None:
        df = _make_ohlcv_df(10)
        result = compute_data_quality_report(df)
        md = result["markdown"]
        assert "## Data Quality Report" in md
        assert "### OHLCV Consistency" in md
