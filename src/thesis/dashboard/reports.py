"""Reports section renderer."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from thesis.charts import build_feature_importance_chart
from thesis.dashboard.cards import render_metric_card
from thesis.dashboard.shared import render_chart, trim_generated_visual_sections


def render_reports_section(session_dir: str) -> None:
    """Render the reports page with thesis markdown and artifact visuals."""
    st.markdown("> 🏠 Dashboard > **Reports**")

    session_path = Path(session_dir)
    reports_dir = session_path / "reports"

    # --- Thesis Report ---
    report_md_path = reports_dir / "thesis_report.md"
    if report_md_path.exists():
        content = trim_generated_visual_sections(report_md_path.read_text())
        st.markdown(content)
    else:
        st.info("No thesis report available.")

    st.divider()

    # --- Equity Curve ---
    equity_png = reports_dir / "equity_curve.png"
    if equity_png.exists():
        st.subheader("Equity Curve")
        st.image(str(equity_png), width="stretch")

    st.divider()

    # --- Walk-Forward History ---
    wf_path = reports_dir / "walk_forward_history.json"
    if wf_path.exists():
        with open(wf_path) as f:
            wf_data = json.load(f)
        with st.container(border=True):
            st.subheader("Walk-Forward History")
            st.caption("Sliding-window validation summary")
            summary_cols = st.columns(3, gap="small")
            render_metric_card(
                summary_cols[0],
                "Windows",
                str(wf_data.get("num_windows", "?")),
                "Total walk-forward windows",
                "#3b82f6",
            )
            render_metric_card(
                summary_cols[1],
                "OOF Predictions",
                f"{wf_data.get('total_oof_predictions', 0):,}",
                "Out-of-fold prediction count",
                "#10b981",
            )
            render_metric_card(
                summary_cols[2],
                "Architecture",
                str(wf_data.get("architecture", "hybrid")),
                "Model architecture used",
                "#8b5cf6",
            )

            details = wf_data.get("window_details", [])
            if details:
                with st.expander("Window Details", expanded=False):
                    st.dataframe(
                        [
                            {
                                "Window": d["window"],
                                "Train Start": d["train_start_idx"],
                                "Train End": d["train_end_idx"],
                                "Test Start": d["test_start_idx"],
                                "Test End": d["test_end_idx"],
                            }
                            for d in details
                        ],
                        width="stretch",
                        hide_index=True,
                    )

    # --- Feature Importance ---
    fi_path = reports_dir / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi_data = json.load(f)
        if fi_data:
            st.divider()
            st.subheader("Feature Importance (Hybrid)")
            chart = build_feature_importance_chart(fi_data)
            render_chart(chart, height="600px")
