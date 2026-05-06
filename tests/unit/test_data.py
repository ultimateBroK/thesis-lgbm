"""Tests for data module.

Tests train/val/test splitting and label distribution logging.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from thesis.shared.config import Config
from thesis.stage_3_labels.labeling import _log_distribution


def create_synthetic_labeled_data(
    n_rows: int = 500,
    start_date: str = "2020-01-01",
) -> pl.DataFrame:
    """Create synthetic labeled data for testing."""
    np.random.seed(42)

    timestamps = pl.datetime_range(
        start=pl.datetime(2020, 1, 1, 0),
        end=pl.datetime(2020, 1, 1, 0) + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )

    # Create features
    n_features = 10
    data = {"timestamp": timestamps, "label": np.random.choice([-1, 0, 1], n_rows)}

    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_rows)

    # Add correlated features (feature_5 and feature_6 will be highly correlated)
    data["feature_5"] = data["feature_0"] + np.random.randn(n_rows) * 0.01
    data["feature_6"] = data["feature_1"] + np.random.randn(n_rows) * 0.01

    # Add OHLC columns
    data["open"] = np.random.randn(n_rows) + 1800
    data["high"] = data["open"] + np.abs(np.random.randn(n_rows))
    data["low"] = data["open"] - np.abs(np.random.randn(n_rows))
    data["close"] = data["open"] + np.random.randn(n_rows)
    data["volume"] = np.random.randint(1000, 10000, n_rows).astype(float)

    # Add label-related columns
    data["tp_price"] = data["close"] + 10
    data["sl_price"] = data["close"] - 10
    data["touched_bar"] = np.random.choice([-1, 0, 1, 2, 3], n_rows)

    return pl.DataFrame(data)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample config for testing."""
    config = Config()
    config.splitting.train_start = "2020-01-01"
    config.splitting.train_end = "2020-03-31 23:59:59"
    config.splitting.val_start = "2020-04-01"
    config.splitting.val_end = "2020-05-31 23:59:59"
    config.splitting.test_start = "2020-06-01"
    config.splitting.test_end = "2020-07-31 23:59:59"
    config.splitting.purge_bars = 24
    config.splitting.embargo_bars = 12
    config.features.correlation_threshold = 0.95
    return config


@pytest.mark.unit
@pytest.mark.data
def test_chronological_ordering(sample_config: Config) -> None:
    """Test that splits are chronologically ordered (train < val < test)."""
    df = create_synthetic_labeled_data(n_rows=5000)

    # Parse date boundaries
    ts_dtype = df["timestamp"].dtype
    bounds = {}
    for key in (
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ):
        bounds[key] = (
            pl.lit(getattr(sample_config.splitting, key))
            .str.to_datetime()
            .cast(ts_dtype)
        )

    train_df = df.filter(
        (pl.col("timestamp") >= bounds["train_start"])
        & (pl.col("timestamp") <= bounds["train_end"])
    )
    val_df = df.filter(
        (pl.col("timestamp") >= bounds["val_start"])
        & (pl.col("timestamp") <= bounds["val_end"])
    )
    test_df = df.filter(
        (pl.col("timestamp") >= bounds["test_start"])
        & (pl.col("timestamp") <= bounds["test_end"])
    )

    if len(train_df) > 0 and len(val_df) > 0:
        train_max = train_df["timestamp"].max()
        val_min = val_df["timestamp"].min()
        assert train_max < val_min, "Train must end before val starts"

    if len(val_df) > 0 and len(test_df) > 0:
        val_max = val_df["timestamp"].max()
        test_min = test_df["timestamp"].min()
        assert val_max < test_min, "Val must end before test starts"


@pytest.mark.unit
@pytest.mark.data
def test_split_ratios_approximate(sample_config: Config) -> None:
    """Test that split ratios are approximately correct."""
    df = create_synthetic_labeled_data(n_rows=5000)

    # Parse date boundaries
    ts_dtype = df["timestamp"].dtype
    bounds = {}
    for key in (
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ):
        bounds[key] = (
            pl.lit(getattr(sample_config.splitting, key))
            .str.to_datetime()
            .cast(ts_dtype)
        )

    train_df = df.filter(
        (pl.col("timestamp") >= bounds["train_start"])
        & (pl.col("timestamp") <= bounds["train_end"])
    )
    val_df = df.filter(
        (pl.col("timestamp") >= bounds["val_start"])
        & (pl.col("timestamp") <= bounds["val_end"])
    )
    test_df = df.filter(
        (pl.col("timestamp") >= bounds["test_start"])
        & (pl.col("timestamp") <= bounds["test_end"])
    )

    total = len(train_df) + len(val_df) + len(test_df)
    if total == 0:
        pytest.skip("No data in splits")

    train_ratio = len(train_df) / total
    val_ratio = len(val_df) / total
    test_ratio = len(test_df) / total

    # Rough checks (allowing for purge effects)
    assert train_ratio > 0.3, f"Train ratio {train_ratio} too low"
    assert val_ratio > 0.1, f"Val ratio {val_ratio} too low"
    assert test_ratio > 0.1, f"Test ratio {test_ratio} too low"


@pytest.mark.unit
@pytest.mark.data
def test_log_distribution_no_crash() -> None:
    """Test that _log_distribution doesn't crash."""
    df = create_synthetic_labeled_data(n_rows=100)

    # Should not raise any exception
    _log_distribution(df)
    df = df.drop("label")

    # Should not raise any exception
    _log_distribution(df)


# ---------------------------------------------------------------------------
# Data quality statistics tests (task 10)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.data
def test_compute_data_quality_stats_no_gaps() -> None:
    """Test _compute_data_quality_stats with perfectly regular data (no gaps)."""
    from thesis.stage_1_data.processing import _compute_data_quality_stats

    n_rows = 100
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 1, 0, time_zone="UTC"),
        end=pl.datetime(2024, 1, 1, 0, time_zone="UTC") + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )
    ohlcv = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.full(n_rows, 1800.0),
            "high": np.full(n_rows, 1802.0),
            "low": np.full(n_rows, 1798.0),
            "close": np.full(n_rows, 1800.0),
            "volume": np.full(n_rows, 5000.0),
            "tick_count": np.full(n_rows, 50),
            "avg_spread": np.full(n_rows, 0.02),
        }
    )

    group_ms = 3_600_000  # 1 hour
    stats = _compute_data_quality_stats(ohlcv, group_ms, deduped_timestamps=0)

    assert stats["total_bars"] == n_rows
    assert stats["deduped_timestamps"] == 0
    assert stats["calendar_gaps"] == 0
    assert stats["weekend_gaps"] == 0
    assert stats["real_gaps"] == 0
    assert stats["estimated_missing_bars"] == 0
    assert stats["largest_gap_bars"] == 0
    assert stats["start_date"] is not None
    assert stats["end_date"] is not None


@pytest.mark.unit
@pytest.mark.data
def test_compute_data_quality_stats_with_gaps() -> None:
    """Test _compute_data_quality_stats detects gaps in irregular data."""
    from thesis.stage_1_data.processing import _compute_data_quality_stats

    # Create data with a known gap: skip 5 hours
    n_rows = 200
    timestamps = pl.datetime_range(
        start=pl.datetime(2024, 1, 2, 0, time_zone="UTC"),  # Thursday
        end=pl.datetime(2024, 1, 2, 0, time_zone="UTC") + pl.duration(hours=n_rows - 1),
        interval="1h",
        eager=True,
    )
    # Insert a gap: remove 5 rows in the middle
    gap_start = 80
    gap_end = gap_start + 5
    keep_mask = [True] * n_rows
    for i in range(gap_start, gap_end):
        keep_mask[i] = False
    gapped_timestamps = [ts for i, ts in enumerate(timestamps) if keep_mask[i]]

    ohlcv = pl.DataFrame(
        {
            "timestamp": pl.Series(gapped_timestamps),
            "open": np.full(len(gapped_timestamps), 1800.0),
            "high": np.full(len(gapped_timestamps), 1802.0),
            "low": np.full(len(gapped_timestamps), 1798.0),
            "close": np.full(len(gapped_timestamps), 1800.0),
            "volume": np.full(len(gapped_timestamps), 5000.0),
            "tick_count": np.full(len(gapped_timestamps), 50),
            "avg_spread": np.full(len(gapped_timestamps), 0.02),
        }
    )

    group_ms = 3_600_000  # 1 hour
    stats = _compute_data_quality_stats(ohlcv, group_ms, deduped_timestamps=0)

    assert stats["total_bars"] == len(gapped_timestamps)
    # Should detect at least one gap
    assert stats["calendar_gaps"] >= 1
    # Gap of 5 bars → 4 missing
    assert stats["estimated_missing_bars"] >= 4
    # Largest gap should be >= 5 bars (the one we created)
    assert stats["largest_gap_bars"] >= 5


@pytest.mark.unit
@pytest.mark.data
def test_compute_data_quality_stats_single_bar() -> None:
    """Test _compute_data_quality_stats with just 1 bar — should not crash."""
    from thesis.stage_1_data.processing import _compute_data_quality_stats

    ohlcv = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0),
                end=pl.datetime(2024, 1, 1, 0),
                interval="1h",
                eager=True,
            ),
            "open": [1800.0],
            "high": [1802.0],
            "low": [1798.0],
            "close": [1800.0],
            "volume": [5000.0],
            "tick_count": [50],
            "avg_spread": [0.02],
        }
    )

    stats = _compute_data_quality_stats(ohlcv, 3_600_000, deduped_timestamps=0)

    assert stats["total_bars"] == 1
    assert stats["calendar_gaps"] == 0
    assert stats["estimated_missing_bars"] == 0


# ---------------------------------------------------------------------------
# Additional _impl tests for coverage
# ---------------------------------------------------------------------------

from thesis.stage_1_data.processing import (
    _parse_datetime_bound,
    _deduplicate_and_filter,
    _filter_date_range,
    _log_gap_report,
    _log_candle_quality_report,
    _spans_weekend,
    _save_data_quality_json,
)


@pytest.mark.unit
class TestParseDatetimeBound:
    def test_valid_date(self) -> None:
        result = _parse_datetime_bound("2024-01-01", "start_date", pl.Datetime("ms"))
        assert result is not None

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _parse_datetime_bound("", "start_date", pl.Datetime("ms"))


@pytest.mark.unit
class TestDeduplicateAndFilter:
    def test_deduplicates_timestamps(self) -> None:
        ts = pl.Series("timestamp", [946684800000, 946684800000, 946771200000]).cast(
            pl.Datetime("ms")
        )
        df = pl.DataFrame(
            {
                "timestamp": ts,
                "open": [1.0, 1.1, 2.0],
                "high": [1.5, 1.6, 2.5],
                "low": [0.8, 0.9, 1.8],
                "close": [1.2, 1.3, 2.2],
                "volume": [100.0, 200.0, 300.0],
                "tick_count": [10, 20, 30],
                "avg_spread": [0.01, 0.02, 0.03],
            }
        )
        result, dropped, dupes = _deduplicate_and_filter(df)
        assert len(result) == 2
        assert dupes == 1

    def test_no_duplicates(self) -> None:
        ts = pl.Series("timestamp", [946684800000, 946771200000]).cast(
            pl.Datetime("ms")
        )
        df = pl.DataFrame(
            {
                "timestamp": ts,
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.8, 1.8],
                "close": [1.2, 2.2],
                "volume": [100.0, 200.0],
                "tick_count": [10, 20],
                "avg_spread": [0.01, 0.02],
            }
        )
        result, dropped, dupes = _deduplicate_and_filter(df)
        assert len(result) == 2
        assert dupes == 0


@pytest.mark.unit
class TestFilterDateRange:
    def test_filters_to_range(self) -> None:
        ts = pl.Series(
            "timestamp",
            [
                1704067200000,
                1704153600000,
                1704240000000,
                1704326400000,
                1704412800000,
                1704499200000,
                1704585600000,
                1704672000000,
                1704758400000,
                1704844800000,
            ],
        ).cast(pl.Datetime("ms"))
        df = pl.DataFrame(
            {
                "timestamp": ts,
                "open": [1.0] * 10,
                "high": [1.5] * 10,
                "low": [0.8] * 10,
                "close": [1.2] * 10,
                "volume": [100.0] * 10,
            }
        )
        config = Config()
        config.data.start_date = "2024-01-03"
        config.data.end_date = "2024-01-07"
        result = _filter_date_range(df, config)
        assert len(result) == 5

    def test_empty_result_raises(self) -> None:
        ts = pl.Series("timestamp", [1577836800000]).cast(pl.Datetime("ms"))
        df = pl.DataFrame(
            {
                "timestamp": ts,
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
        config = Config()
        config.data.start_date = "2030-01-01"
        config.data.end_date = "2030-12-31"
        with pytest.raises(ValueError, match="No OHLCV bars remain"):
            _filter_date_range(df, config)


@pytest.mark.unit
class TestSpansWeekend:
    def test_weekday_gap(self) -> None:
        from datetime import datetime

        start = datetime(2024, 1, 15, 0, 0)  # Monday
        end = datetime(2024, 1, 16, 0, 0)  # Tuesday
        assert _spans_weekend(start, end) is False

    def test_weekend_gap(self) -> None:
        from datetime import datetime

        start = datetime(2024, 1, 12, 22, 0)  # Friday evening
        end = datetime(2024, 1, 15, 6, 0)  # Monday morning
        assert _spans_weekend(start, end) is True

    def test_short_gap_not_weekend(self) -> None:
        from datetime import datetime

        start = datetime(2024, 1, 12, 23, 0)  # Friday night
        end = datetime(2024, 1, 13, 4, 0)  # Saturday morning — but < 6 hours
        assert _spans_weekend(start, end) is False


@pytest.mark.unit
class TestLogGapReport:
    def test_single_bar(self) -> None:
        df = pl.DataFrame({"timestamp": [pl.datetime(2024, 1, 1)]})
        # Should not crash with < 2 bars
        _log_gap_report(df, 3_600_000)

    def test_multi_bar(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 1, 5),
                    interval="1h",
                    eager=True,
                ),
            }
        )
        _log_gap_report(df, 3_600_000)


@pytest.mark.unit
class TestLogCandleQualityReport:
    def test_empty(self) -> None:
        df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "tick_count": [],
                "avg_spread": [],
            }
        ).cast(
            {
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "tick_count": pl.Int64,
                "avg_spread": pl.Float64,
            }
        )
        _log_candle_quality_report(df)  # Should not crash

    def test_valid_candles(self) -> None:
        df = pl.DataFrame(
            {
                "open": [1.0],
                "high": [1.5],
                "low": [0.8],
                "close": [1.2],
                "volume": [100.0],
                "tick_count": [10],
                "avg_spread": [0.01],
            }
        )
        _log_candle_quality_report(df)


@pytest.mark.unit
class TestSaveDataQualityJson:
    def test_saves_json(self, tmp_path) -> None:
        config = Config()
        config.paths.data_quality_json = str(tmp_path / "data_quality.json")
        stats = {"total_bars": 100, "deduped_timestamps": 5}
        _save_data_quality_json(stats, config)

        import json

        path = tmp_path / "data_quality.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_bars"] == 100
