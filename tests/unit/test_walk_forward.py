"""Tests for walk-forward validation helpers.

Tests for confidence column enrichment added to OOF predictions (task 13).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from thesis.shared.config import Config
from thesis.stage_4_training.walk_forward.utils import (
    _add_confidence_columns,
    _validate_predictions,
    _write_prediction_manifest,
)


def _make_oof_df(n_rows: int = 20) -> pl.DataFrame:
    """Create a synthetic OOF predictions DataFrame with probability columns."""
    rng = np.random.RandomState(42)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0),
                end=pl.datetime(2024, 1, 1, 0) + pl.duration(hours=n_rows - 1),
                interval="1h",
                eager=True,
            ),
            "true_label": rng.choice([-1, 0, 1], n_rows),
            "pred_label": rng.choice([-1, 0, 1], n_rows),
            "pred_proba_class_minus1": rng.uniform(0.1, 0.5, n_rows),
            "pred_proba_class_0": rng.uniform(0.1, 0.6, n_rows),
            "pred_proba_class_1": rng.uniform(0.1, 0.5, n_rows),
        }
    )


# ---------------------------------------------------------------------------
# Confidence columns tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_add_confidence_columns_produces_expected_columns() -> None:
    """_add_confidence_columns should add max_confidence and confidence_bin."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    assert "max_confidence" in result.columns
    assert "confidence_bin" in result.columns
    # Original rows should be preserved
    assert len(result) == len(df)


@pytest.mark.unit
def test_add_confidence_columns_max_is_max_of_probas() -> None:
    """max_confidence should be the row-wise max of the three proba columns."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    expected_max = df.select(
        pl.max_horizontal(
            [
                pl.col("pred_proba_class_minus1"),
                pl.col("pred_proba_class_0"),
                pl.col("pred_proba_class_1"),
            ]
        )
    ).to_series()

    actual_max = result["max_confidence"]
    for i in range(len(df)):
        assert actual_max[i] == pytest.approx(expected_max[i])


@pytest.mark.unit
def test_add_confidence_columns_bin_logic() -> None:
    """Confidence bins should follow: >= 0.6 → high, >= 0.4 → medium, else low."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    max_conf = result["max_confidence"].to_numpy()
    bins = result["confidence_bin"].to_list()

    for i in range(len(df)):
        if max_conf[i] >= 0.6:
            assert bins[i] == "high", f"Row {i}: conf={max_conf[i]:.3f}, bin={bins[i]}"
        elif max_conf[i] >= 0.4:
            assert bins[i] == "medium", (
                f"Row {i}: conf={max_conf[i]:.3f}, bin={bins[i]}"
            )
        else:
            assert bins[i] == "low", f"Row {i}: conf={max_conf[i]:.3f}, bin={bins[i]}"


@pytest.mark.unit
def test_add_confidence_columns_missing_proba_noop() -> None:
    """When probability columns are missing, the function returns the DF unchanged."""
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0),
                end=pl.datetime(2024, 1, 1, 3),
                interval="1h",
                eager=True,
            ),
            "pred_label": [1, -1, 0, 1],
        }
    )
    result = _add_confidence_columns(df)

    # Should be unchanged — no new columns
    assert result.columns == df.columns
    assert len(result) == len(df)


@pytest.mark.unit
def test_add_confidence_columns_partial_proba_noop() -> None:
    """When only some probability columns exist, the function is a no-op."""
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0),
                end=pl.datetime(2024, 1, 1, 3),
                interval="1h",
                eager=True,
            ),
            "pred_proba_class_minus1": [0.5, 0.1, 0.2, 0.3],
            "pred_proba_class_0": [0.3, 0.7, 0.3, 0.3],
            # pred_proba_class_1 is missing
        }
    )
    result = _add_confidence_columns(df)

    # Should be unchanged
    assert "max_confidence" not in result.columns
    assert "confidence_bin" not in result.columns
    assert len(result) == len(df)


@pytest.mark.unit
def test_add_confidence_columns_bins_categorical() -> None:
    """Confidence bins should be string type (not null)."""
    df = _make_oof_df(20)
    result = _add_confidence_columns(df)

    assert result["confidence_bin"].dtype == pl.Utf8
    assert result["confidence_bin"].null_count() == 0


@pytest.mark.unit
def test_validate_predictions_rejects_duplicate_timestamps(tmp_path) -> None:
    """OOF predictions must have unique timestamps (no window overlap)."""
    df = _add_confidence_columns(_make_oof_df(4))
    bad = df.with_columns(pl.lit(df["timestamp"][0]).alias("timestamp"))

    with pytest.raises(ValueError, match="duplicate timestamps"):
        _validate_predictions(bad, tmp_path / "final_predictions.parquet")


@pytest.mark.unit
def test_validate_predictions_rejects_invalid_label(tmp_path) -> None:
    """pred_label must stay in {-1, 0, 1}."""
    df = _add_confidence_columns(_make_oof_df(4)).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(2)
        .otherwise(pl.col("pred_label"))
        .alias("pred_label")
    )

    with pytest.raises(ValueError, match="Invalid pred_label"):
        _validate_predictions(df, tmp_path / "final_predictions.parquet")


@pytest.mark.unit
def test_write_prediction_manifest(tmp_path) -> None:
    """Prediction manifest should summarize final_predictions.parquet."""
    df = _add_confidence_columns(_make_oof_df(5))
    preds_path = tmp_path / "final_predictions.parquet"

    _validate_predictions(df, preds_path)
    _write_prediction_manifest(df, preds_path, windows_count=2)

    with open(tmp_path / "prediction_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["row_count"] == 5
    assert manifest["windows_count"] == 2
    assert manifest["start"] == str(df["timestamp"][0])
    assert manifest["end"] == str(df["timestamp"][-1])
    assert manifest["mean_confidence"] is not None


# ---------------------------------------------------------------------------
# Additional walk-forward utility function tests
# ---------------------------------------------------------------------------

from thesis.stage_4_training.walk_forward.utils import (
    _select_static_feature_cols,
    _counts_dict,
    _pct_dict,
    _window_dates,
    _window_diagnostics,
    _compute_per_class_metrics,
    _add_prediction_diagnostics,
    _label_suffix,
    _one_hot_proba_columns,
    _align_probability_matrix,
    _probability_columns,
    _log_gru_signal_quality,
)


@pytest.mark.unit
class TestSelectStaticFeatureCols:
    def test_uses_config_whitelist(self) -> None:
        from thesis.shared.config import Config

        config = Config()
        config.features.static_feature_cols = ["rsi_14", "adx_14"]
        df = pl.DataFrame({"rsi_14": [1.0], "adx_14": [2.0], "extra": [3.0]})
        result = _select_static_feature_cols(config, df, ["extra"])
        assert result == ["rsi_14", "adx_14"]

    def test_fallback_when_config_cols_missing(self) -> None:
        from thesis.shared.config import Config

        config = Config()
        config.features.static_feature_cols = ["nonexistent"]
        df = pl.DataFrame({"a": [1.0], "b": [2.0]})
        result = _select_static_feature_cols(config, df, ["a", "b"])
        assert result == ["a", "b"]

    def test_fallback_filters_to_available(self) -> None:
        from thesis.shared.config import Config

        config = Config()
        config.features.static_feature_cols = ["nonexistent"]
        df = pl.DataFrame({"a": [1.0], "c": [3.0]})
        result = _select_static_feature_cols(config, df, ["a", "b", "c"])
        assert "a" in result
        assert "c" in result
        assert "b" not in result


@pytest.mark.unit
class TestCountsDict:
    def test_basic(self) -> None:
        counts = _counts_dict(np.array([-1, 0, 1, -1, 0, 0]))
        assert counts == {"-1": 2, "0": 3, "1": 1}

    def test_empty(self) -> None:
        assert _counts_dict(np.array([], dtype=np.int32)) == {}

    def test_single_class(self) -> None:
        assert _counts_dict(np.array([1, 1, 1])) == {"1": 3}


@pytest.mark.unit
class TestPctDict:
    def test_basic(self) -> None:
        result = _pct_dict({"1": 3, "0": 1})
        assert result == {"1": 75.0, "0": 25.0}

    def test_empty(self) -> None:
        assert _pct_dict({}) == {}


@pytest.mark.unit
class TestWindowDates:
    def test_basic(self) -> None:
        df = pl.DataFrame(
            {"timestamp": [pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 2)]}
        )
        result = _window_dates(df)
        assert result["start"] == str(pl.datetime(2024, 1, 1))
        assert result["end"] == str(pl.datetime(2024, 1, 2))

    def test_empty(self) -> None:
        df = pl.DataFrame({"timestamp": []}).cast({"timestamp": pl.Datetime})
        result = _window_dates(df)
        assert result == {"start": "", "end": ""}

    def test_no_timestamp_col(self) -> None:
        df = pl.DataFrame({"a": [1]})
        result = _window_dates(df)
        assert result == {"start": "", "end": ""}


@pytest.mark.unit
class TestWindowDiagnostics:
    def test_returns_expected_keys(self) -> None:
        rng = np.random.RandomState(42)
        n = 30
        train_df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2023, 1, 1),
                    pl.datetime(2023, 1, 1) + pl.duration(hours=n - 1),
                    interval="1h",
                    eager=True,
                )
            }
        )
        test_df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2023, 2, 1),
                    pl.datetime(2023, 2, 1) + pl.duration(hours=9),
                    interval="1h",
                    eager=True,
                )
            }
        )
        y_train = rng.choice([-1, 0, 1], n)
        y_test = rng.choice([-1, 0, 1], 10)

        diag = _window_diagnostics(0, train_df, test_df, y_train, y_test)
        assert diag["window"] == 0
        assert diag["train_rows"] == n
        assert diag["test_rows"] == 10
        assert "train_label_counts" in diag
        assert "test_label_counts" in diag
        assert "train_dates" in diag
        assert "test_dates" in diag


@pytest.mark.unit
class TestComputePerClassMetrics:
    def test_perfect_predictions(self) -> None:
        y = np.array([-1, 0, 1, -1, 0, 1])
        result = _compute_per_class_metrics(y, y)
        for cls_str in ["-1", "0", "1"]:
            assert result[cls_str]["precision"] == 1.0
            assert result[cls_str]["recall"] == 1.0
            assert result[cls_str]["f1"] == 1.0

    def test_partial_misses(self) -> None:
        preds = np.array([-1, 0, 1, 0, -1, 1])
        y = np.array([-1, 0, 1, -1, 0, 1])
        result = _compute_per_class_metrics(preds, y)
        assert isinstance(result, dict)
        assert "-1" in result


@pytest.mark.unit
class TestLabelSuffix:
    def test_positive(self) -> None:
        assert _label_suffix(0) == "0"
        assert _label_suffix(1) == "1"

    def test_negative(self) -> None:
        assert _label_suffix(-1) == "minus1"


@pytest.mark.unit
class TestOneHotProbaColumns:
    def test_basic(self) -> None:
        preds = np.array([-1, 0, 1])
        result = _one_hot_proba_columns(preds)
        assert "pred_proba_class_minus1" in result
        assert "pred_proba_class_0" in result
        assert "pred_proba_class_1" in result
        np.testing.assert_array_equal(result["pred_proba_class_minus1"], [1, 0, 0])
        np.testing.assert_array_equal(result["pred_proba_class_0"], [0, 1, 0])
        np.testing.assert_array_equal(result["pred_proba_class_1"], [0, 0, 1])

    def test_custom_prefix(self) -> None:
        preds = np.array([0, 1])
        result = _one_hot_proba_columns(preds, prefix="proba_")
        assert "proba_0" in result
        assert "proba_1" in result


@pytest.mark.unit
class TestAlignProbabilityMatrix:
    def test_reorders_columns(self) -> None:
        # Model outputs in order [0, 1, -1]
        proba = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        class_order = [0, 1, -1]
        result = _align_probability_matrix(proba, class_order)
        assert result.shape == (2, 3)
        # Column 0 should be class -1 (index 2 in original) = 0.7, 0.3
        np.testing.assert_allclose(result[:, 0], [0.7, 0.3])
        # Column 1 should be class 0 (index 0 in original) = 0.1, 0.3
        np.testing.assert_allclose(result[:, 1], [0.1, 0.3])

    def test_missing_class_zero_filled(self) -> None:
        # Binary model — only classes [0, 1]
        proba = np.array([[0.6, 0.4], [0.3, 0.7]])
        class_order = [0, 1]
        result = _align_probability_matrix(proba, class_order)
        assert result.shape == (2, 3)
        # Column 0 is class -1 → should be zeros
        np.testing.assert_allclose(result[:, 0], [0.0, 0.0])


@pytest.mark.unit
class TestProbabilityColumns:
    def test_basic(self) -> None:
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        class_order = [-1, 0, 1]
        result = _probability_columns(proba, class_order)
        assert "pred_proba_class_minus1" in result
        assert "pred_proba_class_0" in result
        assert "pred_proba_class_1" in result
        np.testing.assert_allclose(result["pred_proba_class_minus1"], [0.7, 0.1])


@pytest.mark.unit
class TestAddPredictionDiagnostics:
    def test_adds_keys(self) -> None:
        diag = {"window": 0}
        preds = np.array([-1, 0, 1])
        y_test = np.array([-1, 0, 1])
        proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        _add_prediction_diagnostics(diag, preds, y_test, proba)
        assert "prediction_counts" in diag
        assert "accuracy" in diag
        assert "mean_confidence" in diag
        assert diag["accuracy"] == 1.0
        assert diag["mean_confidence"] == pytest.approx(0.8, abs=0.05)

    def test_empty_predictions(self) -> None:
        diag = {"window": 0}
        preds = np.array([], dtype=np.int32)
        y_test = np.array([], dtype=np.int32)
        proba = np.empty((0, 3))
        _add_prediction_diagnostics(diag, preds, y_test, proba)
        assert diag["accuracy"] is None
        assert diag["mean_confidence"] is None


@pytest.mark.unit
class TestLogGruSignalQuality:
    def test_empty_hidden_states(self) -> None:
        """Empty hidden states should warn and return."""
        _log_gru_signal_quality(np.array([]), np.array([]), Config())

    def test_none_inputs(self) -> None:
        _log_gru_signal_quality(None, np.array([0, 1]), Config())
        _log_gru_signal_quality(np.array([[1, 2]]), None, Config())

    def test_shape_mismatch(self) -> None:
        _log_gru_signal_quality(np.array([[1, 2]]), np.array([0, 1]), Config())

    def test_single_class(self) -> None:
        hidden = np.random.randn(20, 5)
        labels = np.zeros(20, dtype=np.int32)
        _log_gru_signal_quality(hidden, labels, Config())

    def test_insufficient_samples_per_class(self) -> None:
        hidden = np.random.randn(3, 5)
        labels = np.array([-1, 0, 1])  # Only 1 sample per class
        _log_gru_signal_quality(hidden, labels, Config())

    def test_valid_signal(self) -> None:
        rng = np.random.RandomState(42)
        hidden = rng.randn(100, 8)
        # Create labels with some signal — first dim correlates with label
        labels = np.where(
            hidden[:, 0] > 0, 1, np.where(hidden[:, 0] < -0.5, -1, 0)
        ).astype(np.int32)
        _log_gru_signal_quality(hidden, labels, Config())


# ---------------------------------------------------------------------------
# Walk-forward artifacts tests
# ---------------------------------------------------------------------------

from thesis.stage_4_training.walk_forward.artifacts import _build_wf_history


class MockWindow:
    def __init__(self, train_start, train_end, test_start, test_end):
        self.train_start_idx = train_start
        self.train_end_idx = train_end
        self.test_start_idx = test_start
        self.test_end_idx = test_end


@pytest.mark.unit
class TestBuildWfHistory:
    def test_basic(self) -> None:
        windows = [MockWindow(0, 100, 101, 130), MockWindow(50, 150, 151, 180)]
        diags = [
            {"window": 1, "accuracy": 0.8},
            {"window": 2, "accuracy": 0.85},
        ]
        history = _build_wf_history(windows, diags, oof_len=60)
        assert history["num_windows"] == 2
        assert history["total_oof_predictions"] == 60
        assert len(history["window_details"]) == 2
        assert history["window_details"][0]["train_start_idx"] == 0
        assert history["window_details"][0]["accuracy"] == 0.8

    def test_empty(self) -> None:
        history = _build_wf_history([], [], oof_len=0)
        assert history["num_windows"] == 0
        assert history["window_details"] == []

    def test_missing_diagnostics_handled(self) -> None:
        windows = [MockWindow(0, 50, 51, 70)]
        history = _build_wf_history(windows, [], oof_len=20)
        assert len(history["window_details"]) == 1
        # Should have basic keys even without matching diagnostics
        assert "train_start_idx" in history["window_details"][0]
