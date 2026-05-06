"""Markdown section renderers for the thesis report.

Re-exports from focused sub-modules so that existing consumers
(``impl.py``, ``tables.py``) continue to work without import changes.
"""

from __future__ import annotations

from thesis.stage_6_reporting.sections.assess import (
    _EDGE_PF_NEGATIVE,
    _ISSUE_DD_CATASTROPHIC,
    _MIN_TRADES_DEPLOYABLE,
    _QUALITY_DIR_ACC_FAIR,
    _QUALITY_DIR_ACC_GOOD,
    _assess_model_quality,
    _assess_trading_edge,
    _derive_recommendation,
    _get_zone_info,
    _identify_primary_issue,
)
from thesis.stage_6_reporting.sections.backtest import (
    _compute_avg_win_loss_ratio,
    _render_baseline_comparison_section,
    _render_issues,
    _render_metric_zones_section,
    _render_ml_quality_paragraph,
    _render_primary_issue,
    _render_synthesized_verdict,
)
from thesis.stage_6_reporting.sections.data import (
    _fmt_f2,
    _fmt_pct,
    _load_label_distribution,
    _render_auxiliary_regression_section,
    _render_data_quality_section,
    _render_label_design_section,
    _render_validation_methodology_section,
    _tbl_row,
)
from thesis.stage_6_reporting.sections.oof import (
    _render_oof_vs_oos_section,
)

__all__ = [
    "_EDGE_PF_NEGATIVE",
    "_ISSUE_DD_CATASTROPHIC",
    "_MIN_TRADES_DEPLOYABLE",
    "_QUALITY_DIR_ACC_FAIR",
    "_QUALITY_DIR_ACC_GOOD",
    "_assess_model_quality",
    "_assess_trading_edge",
    "_compute_avg_win_loss_ratio",
    "_derive_recommendation",
    "_fmt_f2",
    "_fmt_pct",
    "_get_zone_info",
    "_identify_primary_issue",
    "_load_label_distribution",
    "_render_auxiliary_regression_section",
    "_render_baseline_comparison_section",
    "_render_data_quality_section",
    "_render_issues",
    "_render_label_design_section",
    "_render_metric_zones_section",
    "_render_ml_quality_paragraph",
    "_render_oof_vs_oos_section",
    "_render_primary_issue",
    "_render_synthesized_verdict",
    "_render_validation_methodology_section",
    "_tbl_row",
]
