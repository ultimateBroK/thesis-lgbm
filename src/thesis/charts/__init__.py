"""Interactive ECharts chart builders for thesis visualization.

The public API is kept compatible with the former ``thesis.charts`` module,
while implementations live in smaller domain modules.
"""

from thesis.charts.backtest import (
    _compute_monthly_returns,
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
)
from thesis.charts.data import (
    _get_feature_cols,
    build_candlestick_chart,
    build_correlation_heatmap,
    build_feature_distribution_chart,
    build_feature_distributions_chart,
    build_label_distribution_chart,
)
from thesis.charts.loader import load_session_data
from thesis.charts.model import (
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_prediction_distribution_chart,
)
from thesis.charts.shared import COLORS, EXCLUDED_FEATURE_COLS

__all__ = [
    "COLORS",
    "EXCLUDED_FEATURE_COLS",
    "load_session_data",
    "build_candlestick_chart",
    "build_correlation_heatmap",
    "build_label_distribution_chart",
    "build_feature_distribution_chart",
    "build_feature_distributions_chart",
    "build_confusion_matrix_chart",
    "build_confidence_distribution_chart",
    "build_feature_importance_chart",
    "build_prediction_distribution_chart",
    "build_equity_drawdown_chart",
    "build_pnl_histogram_chart",
    "build_monthly_returns_heatmap",
    "build_rolling_sharpe_chart",
    "build_duration_pnl_scatter",
    "_compute_monthly_returns",
    "_get_feature_cols",
]
