"""Metric zone definitions for CFD backtest benchmarks.

Provides zone classification (excellent/good/moderate/poor/dangerous) for
backtest metrics based on industry benchmarks.

This module has no Streamlit dependency — all functions are pure Python
and can be unit-tested independently.
"""

import math


def _is_extreme_value(metric_name: str, value: float) -> tuple[bool, float]:
    """Check if a metric value is extreme and return threshold info.

    Args:
        metric_name: Name of the metric (e.g., 'recovery_factor', 'sharpe_ratio').
        value: Original metric value.

    Returns:
        Tuple of (is_extreme: bool, threshold: float).
    """
    extreme_thresholds = {
        "recovery_factor": 20.0,
        "sharpe_ratio": 10.0,
        "sortino_ratio": 20.0,
        "calmar_ratio": 15.0,
        "profit_factor": 10.0,
        "sqn": 5.0,
        "kelly_criterion": 0.8,
        "return_pct": 1000.0,
        "cagr_pct": 500.0,
        "return_ann_pct": 500.0,
    }

    threshold = extreme_thresholds.get(metric_name, float("inf"))
    is_extreme = value > threshold

    return is_extreme, threshold


def _get_metric_zone(metric_name: str, value: float) -> tuple[str, str, str]:
    """Return (color_name, zone_label, recommendation) for a given metric.

    Zone labels range from excellent, good, and moderate to poor and
    dangerous. All zones are optimized based on industry benchmarks.

    Args:
        metric_name: The metric key (e.g., 'sharpe_ratio', 'max_drawdown_pct').
        value: The metric value.

    Returns:
        Tuple of (color, zone_label, recommendation_text).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ("moderate", "N/A", "No data available")

    is_extreme, threshold = _is_extreme_value(metric_name, value)

    if is_extreme:
        return (
            "dangerous",
            "Extreme",
            f"Value {value:.1f} exceeds threshold {threshold:.1f}"
            " — verify for overfitting/data issues",
        )

    # ========== Sharpe Ratio ==========
    if metric_name == "sharpe_ratio":
        if value < 0:
            return ("dangerous", "Negative", "Below risk-free rate — review strategy")
        if value < 0.5:
            return ("dangerous", "Poor", "<0.5 — high risk-adjusted cost")
        if value < 1.0:
            return (
                "moderate",
                "Acceptable",
                "0.5-1.0 — acceptable risk-adjusted returns",
            )
        if value < 2.0:
            return ("good", "Good", "1.0-2.0 — solid risk-adjusted returns")
        if value < 3.0:
            return (
                "excellent",
                "Excellent",
                "2.0-3.0 — hedge fund target (verify no overfitting)",
            )
        return ("dangerous", "Suspicious", ">3.0 — verify no overfitting")

    # ========== Sortino Ratio ==========
    if metric_name == "sortino_ratio":
        if value < 0:
            return ("dangerous", "Negative", "Negative — below risk-free rate")
        if value < 0.5:
            return ("dangerous", "Poor", "<0.5 — excessive downside risk")
        if value < 1.5:
            return (
                "moderate",
                "Acceptable",
                "0.5-1.5 — acceptable downside-adjusted returns",
            )
        if value < 2.5:
            return ("good", "Good", "1.5-2.5 — solid downside-adjusted returns")
        if value < 4.0:
            return ("excellent", "Excellent", "2.5-4.0 — very good")
        return ("excellent", "Exceptional", ">4.0 — exceptional downside protection")

    # ========== Max Drawdown ==========
    if metric_name == "max_drawdown_pct":
        if value > -10:
            return ("excellent", "Excellent", "<10% — exceptional capital preservation")
        if value > -20:
            return ("good", "Good", "10-20% — conservative drawdown")
        if value > -35:
            return ("moderate", "Moderate", "20-35% — typical for volatile instruments")
        if value > -50:
            return ("poor", "Significant", "35-50% — high, assess suitability")
        return ("dangerous", "Critical", ">50% — aggressive, question viability")

    # ========== Profit Factor ==========
    if metric_name == "profit_factor":
        if value < 1.0:
            return ("dangerous", "Losing", "<1.0 — strategy loses money")
        if value < 1.2:
            return ("poor", "Marginal", "1.0-1.2 — barely covers costs")
        if value < 1.5:
            return ("moderate", "Acceptable", "1.2-1.5 — covers costs with margin")
        if value < 2.0:
            return ("good", "Good", "1.5-2.0 — strong profitability")
        if value < 3.0:
            return ("excellent", "Excellent", "2.0-3.0 — very efficient")
        return ("dangerous", "Suspicious", ">3.0 — verify no overfitting")

    # ========== Win Rate ==========
    if metric_name == "win_rate_pct":
        if value < 35:
            return ("poor", "Low", "<35% — requires large risk/reward ratio")
        if value < 45:
            return ("moderate", "Acceptable", "35-45% — typical for trend-following")
        if value < 55:
            return ("good", "Good", "45-55% — solid win rate")
        if value < 65:
            return ("excellent", "Excellent", "55-65% — strong (verify if >65%)")
        return ("dangerous", "Suspicious", ">65% — verify no overfitting")

    # ========== CAGR / Annual Return ==========
    if metric_name in ("cagr_pct", "return_ann_pct"):
        if value < 0:
            return ("dangerous", "Negative", "Negative returns — strategy losing money")
        if value < 5:
            return ("poor", "Very Low", "<5% — underperforms inflation")
        if value < 15:
            return ("moderate", "Conservative", "5-15% — conservative but acceptable")
        if value < 30:
            return ("good", "Strong", "15-30% — strong risk-adjusted returns")
        if value < 50:
            return ("excellent", "Excellent", "30-50% — exceptional performance")
        return ("dangerous", "Suspicious", ">50% — verify for overfitting")

    # Total Return
    if metric_name == "return_pct":
        if value < 0:
            return ("dangerous", "Loss", "Negative returns — capital loss")
        if value < 50:
            return ("poor", "Low", "<50% — minimal growth over period")
        if value < 100:
            return ("moderate", "Moderate", "50-100% — doubled capital at best")
        if value < 200:
            return ("good", "Good", "100-200% — solid growth")
        if value < 500:
            return ("excellent", "Strong", "200-500% — strong performance")
        return ("dangerous", "Extreme", ">500% — verify for data issues")

    # Trade count / sample size
    if metric_name == "num_trades":
        if value < 30:
            return ("poor", "Small Sample", "<30 trades — statistically weak")
        if value < 100:
            return ("moderate", "Limited", "30-100 trades — use caution")
        if value < 500:
            return ("good", "Useful", "100-500 trades — useful sample size")
        return ("excellent", "Robust", "≥500 trades — robust backtest sample")

    # ========== Calmar Ratio ==========
    if metric_name == "calmar_ratio":
        if value < 0:
            return ("dangerous", "Negative", "Negative — losses exceed returns")
        if value < 0.5:
            return ("poor", "Weak", "<0.5 — risk outweighs reward")
        if value < 1.0:
            return ("moderate", "Acceptable", "0.5-1.0 — minimum acceptable threshold")
        if value < 2.0:
            return ("good", "Good", "1.0-2.0 — healthy risk/reward balance")
        if value < 3.0:
            return (
                "excellent",
                "Excellent",
                "2.0-3.0 — very strong risk-adjusted returns",
            )
        return ("excellent", "Exceptional", ">3.0 — exceptional risk/reward")

    # ========== SQN ==========
    if metric_name == "sqn":
        if value < 1.0:
            return ("poor", "Poor", "<1.0 — system has no edge")
        if value < 1.5:
            return ("moderate", "Average", "1.0-1.5 — acceptable system quality")
        if value < 2.0:
            return ("moderate", "Average", "1.5-2.0 — acceptable system")
        if value < 3.0:
            return ("good", "Good", "2.0-3.0 — good system quality")
        return ("excellent", "Excellent", ">3.0 — excellent system")

    # ========== Exposure Time ==========
    if metric_name == "exposure_time_pct":
        if value < 15:
            return ("poor", "Too Selective", "<15% — may miss opportunities")
        if value < 30:
            return ("moderate", "Low", "15-30% — conservative exposure")
        if value < 60:
            return ("good", "Good", "30-60% — typical market exposure")
        if value < 80:
            return ("moderate", "High", "60-80% — significant market commitment")
        return ("poor", "Overexposed", ">80% — almost always in trade")

    # ========== Kelly Criterion ==========
    if metric_name == "kelly_criterion":
        if value <= 0:
            return ("dangerous", "Invalid", "0 or negative — no edge")
        if value < 0.15:
            return ("moderate", "Conservative", "<15% — conservative position sizing")
        if value < 0.25:
            return ("good", "Optimal", "15-25% — textbook optimal sizing")
        if value < 0.4:
            return ("moderate", "Aggressive", "25-40% — aggressive, high variance")
        return ("dangerous", "Very Aggressive", ">40% — very aggressive, high risk")

    # ========== Recovery Factor ==========
    if metric_name == "recovery_factor":
        if value < 1.0:
            return ("dangerous", "Bad", "<1.0 — never recovered worst loss")
        if value < 2.0:
            return ("poor", "Weak", "1.0-2.0 — slow recovery")
        if value < 4.0:
            return ("good", "Good", "2.0-4.0 — reasonable recovery")
        return ("excellent", "Excellent", ">4.0 — quick recovery from drawdowns")

    # ========== Volatility (Ann.) ==========
    if metric_name == "volatility_ann_pct":
        if value < 10:
            return ("excellent", "Low", "<10% — very stable")
        if value < 20:
            return ("good", "Moderate", "10-20% — acceptable range")
        if value < 35:
            return ("moderate", "High", "20-35% — elevated risk")
        return ("poor", "Very High", ">35% — excessive volatility")

    # ========== Avg Win / Avg Loss ==========
    if metric_name == "avg_win":
        if value < 50:
            return (
                "poor",
                "Low",
                "<1% of initial capital — small wins, may not cover costs",
            )
        if value < 200:
            return ("moderate", "Moderate", "1-4% of initial capital — decent win size")
        if value < 500:
            return ("good", "Good", "4-10% of initial capital — strong average wins")
        return ("excellent", "High", ">10% of initial capital — excellent win size")

    if metric_name == "avg_loss":
        value = abs(value)
        if value < 50:
            return (
                "excellent",
                "Low",
                "<1% of initial capital — excellent risk control",
            )
        if value < 200:
            return ("good", "Moderate", "1-4% of initial capital — reasonable losses")
        if value < 500:
            return (
                "moderate",
                "High",
                "4-10% of initial capital — large average losses",
            )
        return ("poor", "Severe", ">10% of initial capital — concerning loss size")

    # ========== Equity Final ==========
    if metric_name == "equity_final":
        return ("moderate", "Absolute", "Absolute value — compare to initial capital")

    # ========== Equity Peak ==========
    if metric_name == "equity_peak":
        return ("moderate", "Peak", "Peak equity reached")

    # ========== Commissions ==========
    if metric_name == "commissions":
        return (
            "moderate",
            "Cost",
            "Compare to total return — should be <5% of profits",
        )

    # ========== Avg Trade % ==========
    if metric_name == "avg_trade_pct":
        if value > 1.0:
            return ("excellent", "Excellent", ">1% — strong per-trade returns")
        if value > 0.3:
            return ("good", "Good", "0.3-1% — solid average")
        if value > 0:
            return ("moderate", "Low", "0-0.3% — small per-trade edge")
        return ("poor", "Negative", "<0% — average trade loses money")

    # ========== Best / Worst Trade ==========
    if metric_name == "best_trade_pct":
        if value < 0.5:
            return ("poor", "Weak", "<0.5% — small best trade, limited upside")
        if value < 1.5:
            return ("moderate", "Moderate", "0.5-1.5% — decent single trade")
        if value < 3.0:
            return ("good", "Strong", "1.5-3.0% — strong best trade")
        if value < 5.0:
            return ("excellent", "Excellent", "3.0-5.0% — exceptional single trade")
        return ("dangerous", "Suspicious", ">5.0% — verify for data errors")

    if metric_name == "worst_trade_pct":
        if value > -1.0:
            return ("good", "Good", ">-1% — manageable worst case")
        if value > -3.0:
            return ("moderate", "Moderate", "-1% to -3% — acceptable")
        if value > -5.0:
            return ("poor", "Poor", "-3% to -5% — large single loss")
        return ("dangerous", "Dangerous", "<-5% — catastrophic risk management")

    # ========== Risk/Reward Ratio ==========
    if metric_name == "risk_reward_ratio":
        if value >= 2.0:
            return ("excellent", "Excellent", "≥2.0 — strong R/R")
        if value >= 1.5:
            return ("good", "Good", "1.5-2.0 — solid R/R")
        if value >= 1.0:
            return ("moderate", "Fair", "1.0-1.5 — marginal edge")
    # ========== Model Accuracy ==========
    if metric_name == "accuracy":
        if value >= 0.55:
            return ("excellent", "Excellent", "≥55% — very accurate predictions")
        if value >= 0.50:
            return ("moderate", "Moderate", "50-55% — near random, marginal edge")
        return ("poor", "Poor", "<50% — no predictive edge")

    if metric_name == "directional_accuracy":
        if value >= 0.60:
            return ("excellent", "Excellent", "≥60% — strong directional accuracy")
        if value >= 0.55:
            return ("moderate", "Moderate", "55-60% — decent directional accuracy")
        return ("poor", "Poor", "<55% — weak directional predictions")

    # ========== Expectancy ==========
    if metric_name == "expectancy_pct":
        if value >= 1.0:
            return ("excellent", "Excellent", "≥1% — strong per-trade edge")
        if value >= 0.5:
            return ("good", "Good", "0.5-1% — solid per-trade edge")
        if value >= 0:
            return ("moderate", "Low", "0-0.5% — small per-trade edge")
        return ("poor", "Negative", "<0% — negative expected value per trade")

    # ========== Avg Drawdown ==========
    if metric_name == "avg_drawdown_pct":
        if value > -5:
            return ("excellent", "Excellent", ">-5% — very stable")
        if value > -10:
            return ("good", "Good", "-5% to -10% — conservative")
        if value > -15:
            return ("moderate", "Moderate", "-10% to -15% — typical average drawdown")
        return ("poor", "High", "≤-15% — high average drawdown")

    return ("moderate", "Neutral", "No benchmark available")


_ZONE_COLORS = {
    "excellent": "#22c55e",
    "good": "#84cc16",
    "moderate": "#eab308",
    "poor": "#f97316",
    "dangerous": "#ef4444",
}
