"""Tests for report rendering helpers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from thesis.shared.config import Config
from thesis.stage_6_reporting.benchmarks import _model_label
from thesis.stage_6_reporting.generation import (
    _benchmark_comparison_table,
    _config_table,
    _exec_verdict,
)
from thesis.stage_6_reporting.sections import (
    _assess_model_quality,
    _assess_trading_edge,
    _derive_recommendation,
    _identify_primary_issue,
    _render_data_quality_section,
    _render_metric_zones_section,
    _render_oof_vs_oos_section,
)


@pytest.mark.unit
def test_config_table_shows_walk_forward_for_sliding_validation() -> None:
    """Sliding validation should not render stale static split ranges."""
    cfg = Config()
    cfg.validation.method = "sliding"
    lines: list[str] = []

    _config_table(lines, cfg)
    rendered = "\n".join(lines)

    assert "bar-based walk-forward" in rendered
    assert "train/test/step bars" in rendered
    assert cfg.splitting.train_start not in rendered
    assert cfg.splitting.test_end not in rendered


@pytest.mark.unit
def test_config_table_shows_static_ranges_for_static_validation() -> None:
    """Static validation should keep explicit train/val/test ranges."""
    cfg = Config()
    cfg.validation.method = "static"
    lines: list[str] = []

    _config_table(lines, cfg)
    rendered = "\n".join(lines)

    assert cfg.splitting.train_start in rendered
    assert cfg.splitting.val_start in rendered
    assert cfg.splitting.test_start in rendered
    assert "bar-based walk-forward" not in rendered


@pytest.mark.unit
def test_model_label_matches_architecture() -> None:
    """Report title should reflect the configured architecture."""
    cfg = Config()

    cfg.model.architecture = "lgbm"
    assert _model_label(cfg) == "LightGBM"

    cfg.model.architecture = "hybrid"
    assert _model_label(cfg) == "Hybrid GRU + LightGBM"


@pytest.mark.unit
def test_benchmark_table_discloses_not_cost_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark section should state benchmark cost assumptions explicitly."""
    cfg = Config()
    cfg.model.architecture = "hybrid"
    metrics = {"return_pct": 1.0, "sharpe_ratio": 0.5, "max_drawdown_pct": -1.0}

    def fake_benchmarks(_test_path, _metrics, _config):
        return [
            {
                "strategy": "Hybrid GRU+LGBM",
                "return_pct": 1.0,
                "sharpe": 0.5,
                "max_dd_pct": 1.0,
                "win_rate_pct": 50.0,
                "num_trades": 1,
            }
        ]

    monkeypatch.setattr(
        "thesis.stage_6_reporting.tables.compute_benchmark_comparison", fake_benchmarks
    )
    lines: list[str] = []

    _benchmark_comparison_table(lines, metrics, cfg)
    rendered = "\n".join(lines)

    assert "not trading-cost-equivalent" in rendered
    assert "Hybrid GRU + LightGBM model" in rendered


# ---------------------------------------------------------------------------
# Executive verdict tests (zone-based recommendation)
# ---------------------------------------------------------------------------


class TestAssessModelQuality:
    """Unit tests for _assess_model_quality."""

    def test_poor_acc_below_baseline(self) -> None:
        ps = {
            "accuracy": 0.50,
            "majority_baseline": 0.55,
            "directional_accuracy": 0.48,
            "per_class": {
                "Short": {"f1": 0.3},
                "Hold": {"f1": 0.3},
                "Long": {"f1": 0.3},
            },
        }
        quality, reason = _assess_model_quality(ps)
        assert quality == "POOR"
        assert "below" in reason.lower()

    def test_good_above_baseline_with_edge(self) -> None:
        ps = {
            "accuracy": 0.62,
            "majority_baseline": 0.50,
            "directional_accuracy": 0.58,
            "per_class": {
                "Short": {"f1": 0.5},
                "Hold": {"f1": 0.4},
                "Long": {"f1": 0.5},
            },
        }
        quality, reason = _assess_model_quality(ps)
        assert quality == "GOOD"
        assert "directional edge" in reason.lower()

    def test_fair_marginal_edge(self) -> None:
        ps = {
            "accuracy": 0.52,
            "majority_baseline": 0.50,
            "directional_accuracy": 0.51,
            "per_class": {
                "Short": {"f1": 0.3},
                "Hold": {"f1": 0.3},
                "Long": {"f1": 0.3},
            },
        }
        quality, reason = _assess_model_quality(ps)
        assert quality == "FAIR"


class TestAssessTradingEdge:
    """Unit tests for _assess_trading_edge."""

    def test_negative_profit_factor_below_one(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 0.8, "sharpe_ratio": 0.5, "return_pct": 10}
        )
        assert edge == "NEGATIVE"

    def test_negative_sharpe(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 1.5, "sharpe_ratio": -0.2, "return_pct": 5}
        )
        assert edge == "NEGATIVE"

    def test_marginal_low_sharpe(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 1.3, "sharpe_ratio": 0.7, "return_pct": 5}
        )
        assert edge == "MARGINAL"

    def test_positive(self) -> None:
        edge, reason = _assess_trading_edge(
            {"profit_factor": 2.0, "sharpe_ratio": 2.0, "return_pct": 15}
        )
        assert edge == "POSITIVE"


class TestDeriveRecommendation:
    """Unit tests for _derive_recommendation."""

    def test_not_deployable_poor_model(self) -> None:
        rec = _derive_recommendation(
            "POOR", "POSITIVE", {"num_trades": 100, "return_pct": 10}
        )
        assert "NOT DEPLOYABLE" in rec

    def test_not_deployable_negative_edge(self) -> None:
        rec = _derive_recommendation(
            "GOOD", "NEGATIVE", {"num_trades": 100, "return_pct": 10}
        )
        assert "NOT DEPLOYABLE" in rec

    def test_dep_insufficient_trades(self) -> None:
        rec = _derive_recommendation(
            "GOOD", "POSITIVE", {"num_trades": 10, "return_pct": 10}
        )
        assert "NOT DEPLOYABLE" in rec and "insufficient" in rec.lower()

    def test_deployable_with_caution(self) -> None:
        rec = _derive_recommendation(
            "FAIR", "MARGINAL", {"num_trades": 100, "return_pct": 10}
        )
        assert "caution" in rec.lower()

    def test_deployable(self) -> None:
        rec = _derive_recommendation(
            "GOOD", "POSITIVE", {"num_trades": 150, "return_pct": 10}
        )
        assert rec == "DEPLOYABLE"


class TestIdentifyPrimaryIssue:
    """Unit tests for _identify_primary_issue."""

    def test_zero_trades(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 0,
                "sharpe_ratio": 0,
                "profit_factor": 0,
                "max_drawdown_pct": 0,
                "return_pct": 0,
                "win_rate_pct": 0,
            },
            None,
        )
        assert result is not None
        assert "Zero trades" in result

    def test_negative_sharpe(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 50,
                "sharpe_ratio": -0.3,
                "profit_factor": 0.9,
                "max_drawdown_pct": 10,
                "return_pct": -5,
                "win_rate_pct": 30,
            },
            None,
        )
        assert result is not None
        assert "negative" in result.lower()

    def test_drawdown_catastrophic(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 80,
                "sharpe_ratio": 0.6,
                "profit_factor": 1.1,
                "max_drawdown_pct": 55,
                "return_pct": 5,
                "win_rate_pct": 45,
            },
            None,
        )
        assert result is not None
        assert "catastrophic" in result.lower()

    def test_none_when_all_ok(self) -> None:
        result = _identify_primary_issue(
            {
                "num_trades": 200,
                "sharpe_ratio": 2.0,
                "profit_factor": 2.5,
                "max_drawdown_pct": 10,
                "return_pct": 30,
                "win_rate_pct": 55,
            },
            {"directional_accuracy": 0.60},
        )
        assert result is None


class TestExecVerdict:
    """Integration tests for the extended _exec_verdict function."""

    def _make_pred_stats(self, **overrides) -> dict:
        base = {
            "accuracy": 0.52,
            "majority_baseline": 0.50,
            "directional_accuracy": 0.51,
            "per_class": {
                "Short": {"f1": 0.3},
                "Hold": {"f1": 0.3},
                "Long": {"f1": 0.3},
            },
        }
        base.update(overrides)
        return base

    def test_verdict_line_present_with_metrics(self) -> None:
        """Verdict line must appear when both pred_stats and metrics exist."""
        L: list[str] = []
        metrics = {
            "profit_factor": 1.4,
            "sharpe_ratio": 0.8,
            "return_pct": -2,
            "num_trades": 80,
            "max_drawdown_pct": 15,
            "win_rate_pct": 40,
        }
        _exec_verdict(L, metrics, self._make_pred_stats())
        rendered = "\n".join(L)
        assert "**Verdict:**" in rendered
        assert "Primary issue" in rendered

    def test_no_metrics_still_shows_model_quality(self) -> None:
        """Verdict line shows model quality even without backtest metrics."""
        L: list[str] = []
        _exec_verdict(L, {}, self._make_pred_stats())
        rendered = "\n".join(L)
        assert "**Verdict:**" in rendered
        assert "no backtest metrics available" in rendered.lower()
        assert "Primary issue" in rendered

    def test_no_pred_stats_no_metrics(self) -> None:
        """Returns early with no output lines."""
        L: list[str] = []
        _exec_verdict(L, {}, None)
        # Should have no lines appended
        assert len(L) == 0

    def test_no_pred_stats_with_metrics(self) -> None:
        """Fallback message when only the demo ran."""
        L: list[str] = []
        _exec_verdict(L, {"return_pct": 5, "num_trades": 10}, None)
        rendered = "\n".join(L)
        assert "unavailable" in rendered.lower()


# ---------------------------------------------------------------------------
# OOF vs OOS comparison tests
# ---------------------------------------------------------------------------


def _make_wf_history_json(path: Path, window_details: list[dict]) -> None:
    """Write a synthetic walk-forward history JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "num_windows": len(window_details),
                "total_oof_predictions": sum(
                    w.get("test_rows", 0) for w in window_details
                ),
                "window_details": window_details,
            },
            indent=2,
        )
    )


def _make_predictions_parquet(path: Path, rows: list[dict]) -> None:
    """Write a synthetic predictions parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


@pytest.mark.unit
class TestOofVsOosSection:
    """Tests for _render_oof_vs_oos_section."""

    def test_missing_session_dir_shows_unavailable(self) -> None:
        """Graceful message when session_dir is empty."""
        cfg = Config()
        cfg.paths.session_dir = ""
        L: list[str] = []
        _render_oof_vs_oos_section(L, cfg)
        rendered = "\n".join(L)
        assert "unavailable" in rendered.lower()
        assert "session directory" in rendered.lower()

    def test_missing_wf_history_shows_unavailable(self, tmp_path: Path) -> None:
        """Graceful message when walk_forward_history.json is missing."""
        cfg = Config()
        cfg.paths.session_dir = str(tmp_path)
        L: list[str] = []
        _render_oof_vs_oos_section(L, cfg)
        rendered = "\n".join(L)
        assert "unavailable" in rendered.lower()
        assert "walk-forward history" in rendered.lower()

    def test_empty_window_details_shows_unavailable(self, tmp_path: Path) -> None:
        """Graceful message when window_details is empty."""
        cfg = Config()
        cfg.paths.session_dir = str(tmp_path)
        wf_path = tmp_path / "reports" / "walk_forward_history.json"
        _make_wf_history_json(wf_path, [])
        L: list[str] = []
        _render_oof_vs_oos_section(L, cfg)
        rendered = "\n".join(L)
        assert "unavailable" in rendered.lower()
        assert "no window details" in rendered.lower()

    def test_oof_only_with_no_oos_predictions(self, tmp_path: Path) -> None:
        """Renders OOF metrics with N/A for OOS when predictions missing."""
        cfg = Config()
        cfg.paths.session_dir = str(tmp_path)
        # Point predictions to non-existent file
        cfg.paths.predictions = str(tmp_path / "nonexistent.parquet")

        wf_path = tmp_path / "reports" / "walk_forward_history.json"
        _make_wf_history_json(
            wf_path,
            [
                {
                    "window": 1,
                    "test_rows": 100,
                    "accuracy": 0.55,
                    "per_class": {
                        "-1": {"f1": 0.45, "support": 30},
                        "0": {"f1": 0.60, "support": 40},
                        "1": {"f1": 0.50, "support": 30},
                    },
                },
                {
                    "window": 2,
                    "test_rows": 100,
                    "accuracy": 0.57,
                    "per_class": {
                        "-1": {"f1": 0.47, "support": 25},
                        "0": {"f1": 0.62, "support": 45},
                        "1": {"f1": 0.52, "support": 30},
                    },
                },
            ],
        )

        L: list[str] = []
        _render_oof_vs_oos_section(L, cfg)
        rendered = "\n".join(L)

        assert "OOF vs OOS Generalization Check" in rendered
        assert "OOF (Walk-Forward)" in rendered
        assert "OOS (2024-2026)" in rendered
        # OOF accuracy: (0.55*100 + 0.57*100) / 200 = 0.56 = 56.0%
        assert "56.0%" in rendered
        # OOS should be N/A
        assert "N/A" in rendered

    def test_oof_and_oos_full_comparison(self, tmp_path: Path) -> None:
        """Renders full side-by-side table when both OOF and OOS available."""
        cfg = Config()
        cfg.paths.session_dir = str(tmp_path)
        # Set OOS date range
        cfg.backtest.oob_start_date = "2024-01-01"
        cfg.backtest.oob_end_date = "2026-03-31"

        # Create walk-forward history
        wf_path = tmp_path / "reports" / "walk_forward_history.json"
        _make_wf_history_json(
            wf_path,
            [
                {
                    "window": 1,
                    "test_rows": 50,
                    "accuracy": 0.55,
                    "per_class": {
                        "-1": {"f1": 0.45, "support": 15},
                        "0": {"f1": 0.60, "support": 20},
                        "1": {"f1": 0.50, "support": 15},
                    },
                },
            ],
        )

        # Create predictions parquet with OOS date range rows
        preds_path = tmp_path / "predictions" / "final_predictions.parquet"
        cfg.paths.predictions = str(preds_path)
        _make_predictions_parquet(
            preds_path,
            [
                {
                    "timestamp": "2024-06-15T10:00:00",
                    "true_label": -1,
                    "pred_label": -1,
                },
                {
                    "timestamp": "2024-06-15T11:00:00",
                    "true_label": 0,
                    "pred_label": -1,
                },
                {
                    "timestamp": "2024-06-15T12:00:00",
                    "true_label": 1,
                    "pred_label": 1,
                },
                {
                    "timestamp": "2024-06-15T13:00:00",
                    "true_label": 0,
                    "pred_label": 0,
                },
                {
                    "timestamp": "2025-12-31T23:00:00",
                    "true_label": 0,
                    "pred_label": 0,
                },
                # An out-of-range row that should be excluded
                {
                    "timestamp": "2023-06-15T10:00:00",
                    "true_label": -1,
                    "pred_label": -1,
                },
            ],
        )

        L: list[str] = []
        _render_oof_vs_oos_section(L, cfg)
        rendered = "\n".join(L)

        assert "OOF vs OOS Generalization Check" in rendered
        assert "OOF (Walk-Forward)" in rendered
        assert "OOS (2024-2026)" in rendered
        assert "Delta" in rendered
        # OOF accuracy: 55.0%
        assert "55.0%" in rendered
        # OOS: 5 rows in range, 4 correct (first 5 rows, last 4 correct)
        # true vs pred: row0 (-1==-1)✓, row1 (0!=-1)✗, row2 (1==1)✓, row3 (0==0)✓, row4 (0==0)✓
        # accuracy = 4/5 = 0.80 = 80.0%
        assert "80.0%" in rendered
        assert "Short" in rendered
        assert "Flat" in rendered
        assert "Long" in rendered
        # Macro F1 row present
        assert "Macro F1" in rendered

    def test_no_windows_with_positive_test_rows_returns_none(
        self, tmp_path: Path
    ) -> None:
        """All windows have test_rows=0, returns N/A for OOF."""
        cfg = Config()
        cfg.paths.session_dir = str(tmp_path)

        wf_path = tmp_path / "reports" / "walk_forward_history.json"
        _make_wf_history_json(
            wf_path,
            [{"window": 1, "test_rows": 0, "accuracy": None, "per_class": {}}],
        )

        L: list[str] = []
        _render_oof_vs_oos_section(L, cfg)
        rendered = "\n".join(L)

        assert "OOF vs OOS Generalization Check" in rendered
        # All metric values should be N/A since total_test_rows == 0
        assert rendered.count("N/A") >= 5  # Accuracy + Macro F1 + 3 class F1s


# ---------------------------------------------------------------------------
# Data quality section tests (task 10)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderDataQualitySection:
    """Tests for _render_data_quality_section."""

    def test_missing_json_shows_unavailable_message(self) -> None:
        """Graceful message when data quality JSON is missing."""
        cfg = Config()
        cfg.paths.session_dir = "/nonexistent"
        cfg.paths.data_quality_json = "/nonexistent/data_quality.json"
        L: list[str] = []
        _render_data_quality_section(L, cfg)
        rendered = "\n".join(L)
        assert "not found" in rendered.lower()

    def test_valid_json_renders_table(self, tmp_path: Path) -> None:
        """Valid JSON renders a complete data quality table."""
        import json

        cfg = Config()
        dq_path = tmp_path / "data_quality.json"
        dq_data = {
            "total_bars": 50000,
            "deduped_timestamps": 3,
            "calendar_gaps": 150,
            "weekend_gaps": 140,
            "real_gaps": 10,
            "estimated_missing_bars": 42,
            "largest_gap_bars": 24,
            "start_date": "2013-01-01 00:00:00",
            "end_date": "2026-03-31 23:00:00",
        }
        dq_path.write_text(json.dumps(dq_data))
        cfg.paths.data_quality_json = str(dq_path)

        L: list[str] = []
        _render_data_quality_section(L, cfg)
        rendered = "\n".join(L)

        assert "## Data Quality" in rendered
        assert "50,000" in rendered  # total bars
        assert "42" in rendered  # estimated missing
        assert "24 bars" in rendered  # largest gap
        assert "2013-01-01" in rendered
        assert "2026-03-31" in rendered

    def test_corrupt_json_shows_error_message(self, tmp_path: Path) -> None:
        """Corrupt JSON shows an error message instead of crashing."""
        cfg = Config()
        dq_path = tmp_path / "data_quality.json"
        dq_path.write_text("{invalid json")
        cfg.paths.data_quality_json = str(dq_path)

        L: list[str] = []
        _render_data_quality_section(L, cfg)
        rendered = "\n".join(L)

        assert "could not be read" in rendered.lower()


# ---------------------------------------------------------------------------
# Metric zones section tests (task 12)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderMetricZonesSection:
    """Tests for _render_metric_zones_section."""

    def test_renders_all_expected_metrics(self) -> None:
        """All configured metrics appear in the output."""
        metrics = {
            "return_pct": 15.0,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": -10.0,
            "win_rate_pct": 55.0,
            "profit_factor": 2.0,
            "calmar_ratio": 1.2,
            "sortino_ratio": 1.8,
            "expectancy_pct": 0.5,
        }
        L: list[str] = []
        _render_metric_zones_section(L, metrics)
        rendered = "\n".join(L)

        assert "## Metric Quality Zones" in rendered
        for label in (
            "Total Return",
            "Sharpe Ratio",
            "Max Drawdown",
            "Win Rate",
            "Profit Factor",
            "Calmar Ratio",
            "Sortino Ratio",
            "Expectancy",
        ):
            assert label in rendered, f"Missing metric: {label}"

    def test_good_metrics_show_green_zone(self) -> None:
        """Good metric values should show green emoji indicators."""
        metrics = {
            "return_pct": 20.0,
            "sharpe_ratio": 2.0,
            "max_drawdown_pct": -5.0,
            "win_rate_pct": 60.0,
            "profit_factor": 2.0,
        }
        L: list[str] = []
        _render_metric_zones_section(L, metrics)
        rendered = "\n".join(L)

        # Green indicators should appear for good metrics
        assert "🟢" in rendered

    def test_poor_metrics_show_red_zone(self) -> None:
        """Poor metric values should show red emoji indicators."""
        metrics = {
            "return_pct": -5.0,
            "sharpe_ratio": -0.5,
            "max_drawdown_pct": -40.0,
            "win_rate_pct": 30.0,
            "profit_factor": 0.8,
        }
        L: list[str] = []
        _render_metric_zones_section(L, metrics)
        rendered = "\n".join(L)

        # Red indicators should appear for poor metrics
        assert "🔴" in rendered

    def test_none_metrics_show_na(self) -> None:
        """None values should render as N/A."""
        metrics = {
            "return_pct": None,
            "sharpe_ratio": None,
        }
        L: list[str] = []
        _render_metric_zones_section(L, metrics)
        rendered = "\n".join(L)

        # Each None metric has 2 "N/A" occurrences (value column + zone column)
        assert rendered.count("N/A") >= 2

    def test_with_trades_computes_win_loss_ratio(self) -> None:
        """When trades are provided, Avg Win/Avg Loss row is rendered."""
        trades = [
            {"pnl": 100.0},
            {"pnl": 50.0},
            {"pnl": -30.0},
            {"pnl": -20.0},
        ]
        metrics = {"return_pct": 5.0}
        L: list[str] = []
        _render_metric_zones_section(L, metrics, trades=trades)
        rendered = "\n".join(L)

        assert "Avg Win / Avg Loss" in rendered
        # win/loss = (75) / (25) = 3.0
        assert "3.0" in rendered or "3.00" in rendered
