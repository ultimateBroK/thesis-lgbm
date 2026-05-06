"""Shared chart renderer and config/trade summary helpers."""

from __future__ import annotations

import re

import streamlit as st

from thesis.charts import COLORS
from thesis.dashboard.cards import render_metric_card


def render_chart(chart: object, height: str = "500px") -> None:
    """Render a pyecharts chart into the Streamlit app."""
    try:
        from streamlit_echarts import st_pyecharts

        st_pyecharts(chart, height=height)
    except ImportError as e:
        st.warning(f"Chart render failed: {e}")


def date_only(value: str) -> str:
    """Return the date part of a config timestamp string."""
    return str(value).split()[0]


def trim_generated_visual_sections(content: str) -> str:
    """Hide report sections duplicated by dashboard-native charts."""
    marker_pattern = re.compile(r"^##\s+\d*\.?\s*Visual Evidence", re.MULTILINE)
    marker = marker_pattern.search(content)
    return content[: marker.start()].rstrip() if marker else content


def render_config_summary(config: object) -> None:
    """Render compact current experiment settings in the sidebar."""
    train_span = (
        f"{date_only(config.splitting.train_start)}→"
        f"{date_only(config.splitting.train_end)}"
    )
    val_span = (
        f"{date_only(config.splitting.val_start)}→{date_only(config.splitting.val_end)}"
    )
    test_span = (
        f"{date_only(config.splitting.test_start)}→"
        f"{date_only(config.splitting.test_end)}"
    )
    st.markdown(
        f"**Data**: {config.data.symbol} {config.data.timeframe}  "
        f"{date_only(config.data.start_date)}→{date_only(config.data.end_date)}"
    )
    st.markdown(f"**Split**: train {train_span}  \nval {val_span}  \ntest {test_span}")
    st.markdown(
        f"**Walk-forward**: {config.validation.method}, "
        f"train={config.validation.train_window_bars:,}, "
        f"test={config.validation.test_window_bars:,}, "
        f"purge={config.validation.purge_bars} bars"
    )
    st.markdown(
        f"**GRU**: multiclass, inputs={config.gru.input_size}, "
        f"hidden={config.gru.hidden_size}, seq={config.gru.sequence_length}, "
        f"epochs={config.gru.epochs}"
    )
    st.markdown(
        f"**LightGBM**: {config.model.architecture}, "
        f"objective={config.model.objective}, leaves={config.model.num_leaves}, "
        f"lr={config.model.learning_rate}"
    )
    st.markdown(
        f"**Labels**: horizon={config.labels.horizon_bars}, "
        f"TP={config.labels.atr_tp_multiplier}×ATR, "
        f"SL={config.labels.atr_sl_multiplier}×ATR"
    )
    st.markdown(
        f"**Backtest**: capital=${config.backtest.initial_capital:,.0f}, "
        f"spread={config.backtest.spread_ticks:g} ticks, "
        f"conf≥{config.backtest.confidence_threshold:.2f}"
    )


def render_trade_direction_summary(trades: list[dict]) -> None:
    """Render compact direction counts and PnL without low-value charts."""
    if not trades:
        return

    long_trades = [t for t in trades if t.get("direction") == "long"]
    short_trades = [t for t in trades if t.get("direction") == "short"]
    long_pnl = sum(float(t.get("pnl", 0)) for t in long_trades)
    short_pnl = sum(float(t.get("pnl", 0)) for t in short_trades)

    with st.expander("Direction summary", expanded=False):
        cols = st.columns(4, gap="small")
        render_metric_card(
            cols[0], "Long Trades", f"{len(long_trades):,}", None, COLORS["long"]
        )
        render_metric_card(
            cols[1], "Short Trades", f"{len(short_trades):,}", None, COLORS["short"]
        )
        render_metric_card(
            cols[2], "Long PnL", f"${long_pnl:,.0f}", None, COLORS["long"]
        )
        render_metric_card(
            cols[3], "Short PnL", f"${short_pnl:,.0f}", None, COLORS["short"]
        )
