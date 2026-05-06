"""Unit tests for metric zone classification helpers."""

from __future__ import annotations

import math

import pytest

from thesis.shared.zones import _get_metric_zone, _is_extreme_value


@pytest.mark.unit
def test_extreme_value_thresholds() -> None:
    assert _is_extreme_value("profit_factor", 11.0) == (True, 10.0)
    assert _is_extreme_value("profit_factor", 2.0) == (False, 10.0)
    assert _is_extreme_value("unknown", 999.0) == (False, float("inf"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("metric", "value", "expected_color", "expected_label"),
    [
        ("sharpe_ratio", -0.1, "dangerous", "Negative"),
        ("sharpe_ratio", 0.7, "moderate", "Acceptable"),
        ("sharpe_ratio", 1.5, "good", "Good"),
        ("sharpe_ratio", 2.5, "excellent", "Excellent"),
        ("sharpe_ratio", 3.5, "dangerous", "Suspicious"),
        ("sortino_ratio", -0.1, "dangerous", "Negative"),
        ("sortino_ratio", 1.0, "moderate", "Acceptable"),
        ("sortino_ratio", 2.0, "good", "Good"),
        ("sortino_ratio", 3.0, "excellent", "Excellent"),
        ("max_drawdown_pct", -5.0, "excellent", "Excellent"),
        ("max_drawdown_pct", -15.0, "good", "Good"),
        ("max_drawdown_pct", -30.0, "moderate", "Moderate"),
        ("max_drawdown_pct", -40.0, "poor", "Significant"),
        ("max_drawdown_pct", -55.0, "dangerous", "Critical"),
        ("profit_factor", 0.9, "dangerous", "Losing"),
        ("profit_factor", 1.1, "poor", "Marginal"),
        ("profit_factor", 1.3, "moderate", "Acceptable"),
        ("profit_factor", 1.7, "good", "Good"),
        ("profit_factor", 2.5, "excellent", "Excellent"),
        ("win_rate_pct", 30.0, "poor", "Low"),
        ("win_rate_pct", 40.0, "moderate", "Acceptable"),
        ("win_rate_pct", 50.0, "good", "Good"),
        ("win_rate_pct", 60.0, "excellent", "Excellent"),
        ("win_rate_pct", 70.0, "dangerous", "Suspicious"),
        ("cagr_pct", -1.0, "dangerous", "Negative"),
        ("cagr_pct", 3.0, "poor", "Very Low"),
        ("cagr_pct", 10.0, "moderate", "Conservative"),
        ("cagr_pct", 20.0, "good", "Strong"),
        ("cagr_pct", 40.0, "excellent", "Excellent"),
        ("return_pct", -1.0, "dangerous", "Loss"),
        ("return_pct", 25.0, "poor", "Low"),
        ("return_pct", 75.0, "moderate", "Moderate"),
        ("return_pct", 150.0, "good", "Good"),
        ("return_pct", 300.0, "excellent", "Strong"),
        ("calmar_ratio", -0.1, "dangerous", "Negative"),
        ("calmar_ratio", 0.3, "poor", "Weak"),
        ("calmar_ratio", 0.7, "moderate", "Acceptable"),
        ("calmar_ratio", 1.5, "good", "Good"),
        ("calmar_ratio", 2.5, "excellent", "Excellent"),
        ("sqn", 0.5, "poor", "Poor"),
        ("sqn", 1.2, "moderate", "Average"),
        ("sqn", 2.5, "good", "Good"),
        ("sqn", 3.5, "excellent", "Excellent"),
        ("exposure_time_pct", 10.0, "poor", "Too Selective"),
        ("exposure_time_pct", 20.0, "moderate", "Low"),
        ("exposure_time_pct", 45.0, "good", "Good"),
        ("exposure_time_pct", 70.0, "moderate", "High"),
        ("exposure_time_pct", 90.0, "poor", "Overexposed"),
        ("kelly_criterion", 0.0, "dangerous", "Invalid"),
        ("kelly_criterion", 0.1, "moderate", "Conservative"),
        ("kelly_criterion", 0.2, "good", "Optimal"),
        ("kelly_criterion", 0.3, "moderate", "Aggressive"),
        ("kelly_criterion", 0.5, "dangerous", "Very Aggressive"),
        ("recovery_factor", 0.5, "dangerous", "Bad"),
        ("recovery_factor", 1.5, "poor", "Weak"),
        ("recovery_factor", 3.0, "good", "Good"),
        ("recovery_factor", 5.0, "excellent", "Excellent"),
        ("volatility_ann_pct", 5.0, "excellent", "Low"),
        ("volatility_ann_pct", 15.0, "good", "Moderate"),
        ("volatility_ann_pct", 25.0, "moderate", "High"),
        ("volatility_ann_pct", 40.0, "poor", "Very High"),
        ("avg_win", 25.0, "poor", "Low"),
        ("avg_win", 100.0, "moderate", "Moderate"),
        ("avg_win", 300.0, "good", "Good"),
        ("avg_win", 700.0, "excellent", "High"),
        ("avg_loss", -25.0, "excellent", "Low"),
        ("avg_loss", -100.0, "good", "Moderate"),
        ("avg_loss", -300.0, "moderate", "High"),
        ("avg_loss", -700.0, "poor", "Severe"),
        ("equity_final", 10_000.0, "moderate", "Absolute"),
        ("equity_peak", 11_000.0, "moderate", "Peak"),
        ("commissions", 50.0, "moderate", "Cost"),
        ("avg_trade_pct", 1.5, "excellent", "Excellent"),
        ("avg_trade_pct", 0.5, "good", "Good"),
        ("avg_trade_pct", 0.1, "moderate", "Low"),
        ("avg_trade_pct", -0.1, "poor", "Negative"),
        ("best_trade_pct", 0.2, "poor", "Weak"),
        ("best_trade_pct", 1.0, "moderate", "Moderate"),
        ("best_trade_pct", 2.0, "good", "Strong"),
        ("best_trade_pct", 4.0, "excellent", "Excellent"),
        ("best_trade_pct", 6.0, "dangerous", "Suspicious"),
        ("worst_trade_pct", -0.5, "good", "Good"),
        ("worst_trade_pct", -2.0, "moderate", "Moderate"),
        ("worst_trade_pct", -4.0, "poor", "Poor"),
        ("worst_trade_pct", -6.0, "dangerous", "Dangerous"),
        ("risk_reward_ratio", 2.0, "excellent", "Excellent"),
        ("risk_reward_ratio", 1.5, "good", "Good"),
        ("risk_reward_ratio", 1.0, "moderate", "Fair"),
        ("accuracy", 0.56, "excellent", "Excellent"),
        ("accuracy", 0.52, "moderate", "Moderate"),
        ("accuracy", 0.49, "poor", "Poor"),
        ("directional_accuracy", 0.61, "excellent", "Excellent"),
        ("directional_accuracy", 0.57, "moderate", "Moderate"),
        ("directional_accuracy", 0.54, "poor", "Poor"),
        ("expectancy_pct", 1.1, "excellent", "Excellent"),
        ("expectancy_pct", 0.7, "good", "Good"),
        ("expectancy_pct", 0.2, "moderate", "Low"),
        ("expectancy_pct", -0.1, "poor", "Negative"),
        ("avg_drawdown_pct", -4.0, "excellent", "Excellent"),
        ("avg_drawdown_pct", -8.0, "good", "Good"),
        ("avg_drawdown_pct", -12.0, "moderate", "Moderate"),
        ("avg_drawdown_pct", -20.0, "poor", "High"),
        ("unknown_metric", 1.0, "moderate", "Neutral"),
    ],
)
def test_metric_zone_boundaries(
    metric: str, value: float, expected_color: str, expected_label: str
) -> None:
    color, label, recommendation = _get_metric_zone(metric, value)
    assert color == expected_color
    assert label == expected_label
    assert recommendation


@pytest.mark.unit
def test_metric_zone_missing_and_extreme_values() -> None:
    assert _get_metric_zone("sharpe_ratio", None) == (
        "moderate",
        "N/A",
        "No data available",
    )
    assert _get_metric_zone("sharpe_ratio", math.nan) == (
        "moderate",
        "N/A",
        "No data available",
    )
    color, label, recommendation = _get_metric_zone("profit_factor", 11.0)
    assert color == "dangerous"
    assert label == "Extreme"
    assert "exceeds threshold" in recommendation
