"""Backtest Results section renderer."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from thesis.charts import (
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
)
from thesis.dashboard.cards import render_zoned_metric
from thesis.dashboard.shared import render_chart, render_trade_direction_summary


def render_backtest_section(data: dict, config: object) -> None:
    """Render backtest metrics, charts, analysis panels, and CSV downloads."""
    st.markdown("> 🏠 Dashboard > **Backtest Results**")
    st.header("Backtest Results")

    bt = data.get("backtest_results")
    trades = data.get("trades", [])
    metrics = data.get("metrics", {})

    if not bt:
        st.info("No backtest results available.")
        return

    with st.container(border=True):
        st.subheader("Performance Overview")
        st.caption(
            "Zone indicators based on industry benchmarks for XAU/USD CFD trading"
        )

        st.markdown("**📊 Core Financial Metrics**")
        st.caption(
            "Kept intentionally small: return, risk, edge,"
            " consistency, and sample size."
        )
        kpi_cols = st.columns(3, gap="small")
        render_zoned_metric(
            kpi_cols[0],
            "Total Return",
            metrics.get("return_pct", 0),
            "return_pct",
            "{:.2f}",
            "%",
        )
        render_zoned_metric(
            kpi_cols[1],
            "Max Drawdown",
            metrics.get("max_drawdown_pct", 0),
            "max_drawdown_pct",
            "{:.1f}",
            "%",
        )
        render_zoned_metric(
            kpi_cols[2],
            "Profit Factor",
            metrics.get("profit_factor", 0),
            "profit_factor",
            "{:.2f}",
        )

        kpi_cols = st.columns(3, gap="small")
        render_zoned_metric(
            kpi_cols[0],
            "Sharpe Ratio",
            metrics.get("sharpe_ratio", 0),
            "sharpe_ratio",
            "{:.2f}",
        )
        render_zoned_metric(
            kpi_cols[1],
            "Win Rate",
            metrics.get("win_rate_pct", 0),
            "win_rate_pct",
            "{:.1f}",
            "%",
        )
        render_zoned_metric(
            kpi_cols[2], "Trades", metrics.get("num_trades", 0), "num_trades", "{:.0f}"
        )
        st.caption(
            f"Period: {metrics.get('start', 'N/A')[:10]} → "
            f"{metrics.get('end', 'N/A')[:10]}"
        )
        st.caption(f"Initial balance: ${config.backtest.initial_capital:,.0f}")
        st.caption(f"Final equity: ${metrics.get('equity_final', 0):,.0f}")
        st.caption(
            "🟢 Excellent  🟡 Good  🟠 Moderate"
            "  🔴 Poor/Dangerous  ⚪ N/A (context-dependent)"
        )

    st.divider()

    st.subheader("Equity Curve & Drawdown")
    chart = build_equity_drawdown_chart(
        trades, metrics, initial_capital=config.backtest.initial_capital
    )
    render_chart(chart, height="600px")

    st.divider()

    pnl_col, duration_col = st.columns(2)
    with pnl_col:
        st.subheader("Trade PnL Distribution")
        chart = build_pnl_histogram_chart(trades, metrics)
        render_chart(chart, height="500px")
    with duration_col:
        st.subheader("Trade Duration vs PnL")
        chart = build_duration_pnl_scatter(trades)
        render_chart(chart, height="500px")

    st.divider()

    monthly_col, rolling_col = st.columns(2)
    with monthly_col:
        st.subheader("Monthly Returns")
        chart = build_monthly_returns_heatmap(
            trades, initial_capital=config.backtest.initial_capital
        )
        render_chart(chart, height="400px")
    with rolling_col:
        if len(trades) > 30:
            st.subheader("Rolling Metrics")
            chart = build_rolling_sharpe_chart(trades)
            render_chart(chart, height="400px")
        else:
            st.info("Need more than 30 trades for rolling metrics.")

    render_trade_direction_summary(trades)

    st.divider()
    st.subheader("Download Data")
    session_dir = data.get("session_dir")
    if session_dir:
        bt_dir = Path(session_dir) / "backtest"
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            csv_path = bt_dir / "trades_detail.csv"
            if csv_path.exists():
                st.download_button(
                    "📄 Trades Detail CSV",
                    data=csv_path.read_text(),
                    file_name="trades_detail.csv",
                    mime="text/csv",
                )

        with dl_col2:
            eq_path = bt_dir / "equity_curve.csv"
            if eq_path.exists():
                st.download_button(
                    "📈 Equity Curve CSV",
                    data=eq_path.read_text(),
                    file_name="equity_curve.csv",
                    mime="text/csv",
                )

        with dl_col3:
            preds_dir = Path(session_dir) / "predictions"
            preds_csv = preds_dir / "final_predictions.csv"
            if preds_csv.exists():
                st.download_button(
                    "🎯 Predictions CSV",
                    data=preds_csv.read_text(),
                    file_name="final_predictions.csv",
                    mime="text/csv",
                )
    else:
        st.info("No session directory available for downloads.")
