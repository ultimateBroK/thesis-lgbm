"""Dashboard entry point — sidebar, navigation, and section dispatch."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from thesis.dashboard.backtest import render_backtest_section
from thesis.dashboard.data import render_data_section
from thesis.dashboard.model import render_model_section
from thesis.dashboard.reports import render_reports_section
from thesis.dashboard.session import load_config, session_selector_fragment
from thesis.dashboard.shared import render_config_summary
from thesis.dashboard.training import render_training_section


def main() -> None:
    """Render the Streamlit dashboard with session selection and navigation."""
    st.set_page_config(
        page_title="Thesis Dashboard — XAU/USD",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        /* AMOLED glass effect for metric cards */
        .stMetric {
            background: linear-gradient(135deg,
                rgba(255,255,255,0.05) 0%,
                rgba(255,255,255,0.02) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.3),
                        inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .stMetric label {
            font-size: 0.8rem;
            color: rgba(255,255,255,0.6);
            letter-spacing: 0.02em;
        }
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e2e8f0;
        }
        .stMetric div[data-testid="stMetricDelta"] {
            font-size: 0.85rem;
        }
        /* Subtle glow on hover */
        .stMetric:hover {
            border-color: rgba(255,255,255,0.15);
            box-shadow: 0 4px 30px rgba(0,0,0,0.4),
                        inset 0 1px 0 rgba(255,255,255,0.1),
                        0 0 20px rgba(37,99,235,0.05);
            transition: all 0.2s ease;
        }
        /* Compact sidebar spacing */
        .stSidebar .stExpander details summary {
            font-weight: 600;
            font-size: 0.9rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Sidebar Header ──
    st.sidebar.markdown("### 📈 Thesis Dashboard")
    st.sidebar.caption("Hybrid GRU + LightGBM — XAU/USD")

    # ── Session Selector ──
    with st.sidebar.expander("📁 Session", expanded=True):
        selected = session_selector_fragment()

    if selected is None:
        st.error("No session results found. Run `pixi run workflow` first.")
        return

    # ── Navigation ──
    sections = ["📊 Data", "🧠 Model", "🏃 Training", "💰 Backtest", "📝 Reports"]
    section_map = {
        "📊 Data": "Data Exploration",
        "🧠 Model": "Model Performance",
        "🏃 Training": "Training",
        "💰 Backtest": "Backtest Results",
        "📝 Reports": "Reports",
    }

    current_section = st.session_state.get("nav_section", "📊 Data")

    left_spacer, nav_center, right_spacer = st.columns([0.2, 0.6, 0.2])
    with nav_center:
        nav_cols = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
        for i, sec in enumerate(sections):
            with nav_cols[i]:
                btn_type = "primary" if sec == current_section else "secondary"
                if st.button(sec, key=f"nav_{sec}", type=btn_type, width="stretch"):
                    st.session_state.nav_section = sec
                    st.rerun()

    section = st.session_state.get("nav_section", "📊 Data")

    # ── Load data ──
    session_path = str(Path("results") / selected)
    loaded = load_config(session_path)
    config = loaded["config"]
    data = loaded["data"]
    metrics = data.get("metrics", {})

    # ── Configuration sidebar ──
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        render_config_summary(config)

    # ── Quick Stats sidebar ──
    if metrics:
        with st.sidebar.expander("📊 Quick Stats", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Return", f"{metrics.get('return_pct', 0):.2f}%")
            c2.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
            c1.metric("Trades", f"{metrics.get('num_trades', 0)}")
            c2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")

    # ── Render selected section ──
    section_name = section_map[section]
    if section_name == "Data Exploration":
        render_data_section(data, config)
    elif section_name == "Model Performance":
        render_model_section(data, session_path)
    elif section_name == "Training":
        render_training_section(data, session_path)
    elif section_name == "Reports":
        render_reports_section(session_path)
    else:
        render_backtest_section(data, config)


main()
