"""Interactive Streamlit dashboard for thesis visualization.

Launch: ``pixi run streamlit``

Combines session discovery, metric cards, zone classification, and all
five dashboard sections (Data, Model, Training, Backtest, Reports) into
a single module.  Zone colour helpers are re-exported from
``thesis.zones`` so that nothing in this file depends on the removed
``dashboard/`` package.
"""

from __future__ import annotations

import html
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie

# ---------------------------------------------------------------------------
# Zone helpers (re-exported from thesis.zones – pure Python, no Streamlit)
# ---------------------------------------------------------------------------
from thesis.zones import (
    _ZONE_COLORS,
    _get_metric_zone,
    _is_extreme_value,
)

# ---------------------------------------------------------------------------
# Session management & chart builders
# ---------------------------------------------------------------------------
from thesis.charts import (
    COLORS,
    EXCLUDED_FEATURE_COLS,
    build_candlestick_chart,
    build_confidence_distribution_chart,
    build_confusion_matrix_chart,
    build_correlation_heatmap,
    build_duration_pnl_scatter,
    build_equity_drawdown_chart,
    build_feature_importance_chart,
    build_label_distribution_chart,
    build_monthly_returns_heatmap,
    build_pnl_histogram_chart,
    build_rolling_sharpe_chart,
    build_shap_chart,
    load_session_data,
)
from thesis.session_paths import load_config_for_session

logger = logging.getLogger("thesis.app_streamlit")

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path for sibling imports
# ---------------------------------------------------------------------------
_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)


# ===================================================================
# Metric card helpers
# ===================================================================


def _render_zoned_metric(
    col: object,
    label: str,
    value: float,
    metric_key: str,
    format_str: str = "{:.2f}",
    unit: str = "",
) -> None:
    """Render a metric card with colour-coded zone indicator."""
    is_extreme, _ = _is_extreme_value(metric_key, value)
    color, zone_label, recommendation = _get_metric_zone(metric_key, value)

    hex_color = _ZONE_COLORS.get(color, "#6b7280")
    display_suffix = " ⚠️" if is_extreme else ""
    safe_label = html.escape(label)
    safe_value = html.escape(format_str.format(value))
    safe_unit = html.escape(unit)
    safe_zone = html.escape(zone_label)
    safe_rec = html.escape(recommendation)

    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {hex_color}22 0%, {hex_color}11 100%);
            border-left: 3px solid {hex_color};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 4px 0;
            min-height: 110px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        ">
            <div>
                <div style="font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">{safe_label}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;">
                    {safe_value}{safe_unit}{display_suffix}
                </div>
            </div>
            <div style="margin-top: 8px;">
                <span style="
                    background: {hex_color}33;
                    color: {hex_color};
                    padding: 2px 10px;
                    border-radius: 12px;
                    font-size: 0.65rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0.03em;
                ">{safe_zone}</span>
                <div style="font-size: 0.65rem; color: inherit; opacity: 0.6; margin-top: 4px; line-height: 1.3;">{safe_rec}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(
    col: object,
    label: str,
    value: str,
    caption: str | None,
    color: str,
) -> None:
    """Render a styled metric card with gradient background and accent border."""
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    caption_html = (
        f'<div style="font-size: 0.65rem; color: inherit; opacity: 0.6; margin-top: 4px; line-height: 1.3;">{html.escape(caption)}</div>'
        if caption
        else ""
    )
    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 12px 14px;
            margin: 4px 0;
            min-height: 90px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-sizing: border-box;
        ">
            <div>
                <div style="font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">{safe_label}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;">{safe_value}</div>
            </div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================================================================
# Session discovery & loading
# ===================================================================


def _find_sessions() -> list[Path]:
    """Discover available session directories under ``results/``."""
    results = Path("results")
    if not results.exists():
        return []

    def parse_session_timestamp(path: Path) -> datetime | None:
        m = re.search(r"(\d{8})_(\d{6})$", path.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: parse_session_timestamp(p) or datetime.min,
        reverse=True,
    )
    return sessions


def _parse_session_meta(name: str) -> dict[str, str]:
    """Parse a session directory name into metadata fields."""
    parts = name.split("_")
    if len(parts) >= 4:
        return {
            "symbol": parts[0],
            "timeframe": parts[1],
            "date": f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}",
            "time": f"{parts[3][:2]}:{parts[3][2:4]}:{parts[3][4:6]}",
        }
    return {"symbol": "?", "timeframe": "?", "date": "?", "time": "?"}


@st.cache_resource(ttl=60)
def _load_config(session_dir: str) -> dict:
    """Load configuration and session data for *session_dir*."""
    config = load_config_for_session(session_dir)
    data = load_session_data(config)
    return {"config": config, "data": data}


@st.fragment(run_every=30)
def _session_selector_fragment() -> str | None:
    """Render a sidebar session selector and return the chosen session name."""
    sessions = _find_sessions()
    if not sessions:
        return None

    session_names = [s.name for s in sessions]

    known = st.session_state.get("known_sessions", set())
    current_set = set(session_names)
    new_sessions = current_set - known
    if new_sessions and known:
        for ns in sorted(new_sessions):
            meta = _parse_session_meta(ns)
            st.toast(f"🆕 New session: {meta['date']} {meta['time']}", icon="📈")
    st.session_state.known_sessions = current_set

    session_labels = []
    for name in session_names:
        meta = _parse_session_meta(name)
        session_labels.append(
            f"{meta['date']} {meta['time']} ({meta['symbol']} {meta['timeframe']})"
        )

    current = st.session_state.get("selected_session")
    if current in session_names:
        idx = session_names.index(current)
    else:
        idx = 0
        st.session_state.selected_session = session_names[0]

    selected_label = st.selectbox(
        "Select session",
        options=session_labels,
        index=idx,
        key="_session_selectbox",
    )
    selected = session_names[session_labels.index(selected_label)]
    st.session_state.selected_session = selected

    if st.button("🔄 Refresh", width="stretch", key="_refresh_btn"):
        st.rerun()

    st.caption("Run `pixi run workflow` to generate new sessions")
    return selected


# ===================================================================
# Shared chart renderer
# ===================================================================


def _render_chart(chart: object, height: str = "500px") -> None:
    """Render a pyecharts chart into the Streamlit app."""
    try:
        from streamlit_echarts import st_pyecharts

        st_pyecharts(chart, height=height)
    except Exception as e:
        st.warning(f"Chart render failed: {e}")


# ===================================================================
# Section: Data Exploration
# ===================================================================


def _render_data_section(data: dict, config: object) -> None:
    """Render the Data Exploration section with charts and controls."""
    st.markdown("> 🏠 Dashboard > **Data Exploration**")
    st.header("Data Exploration")

    ohlcv = data.get("ohlcv")
    if ohlcv is not None:
        st.caption(
            f"{len(ohlcv):,} bars | "
            f"{ohlcv['timestamp'].cast(pl.Utf8).min()} → {ohlcv['timestamp'].cast(pl.Utf8).max()}"
        )
    features = data.get("features")
    labels = data.get("labels")

    if ohlcv is not None and len(ohlcv) > 0:
        st.subheader("Candlestick Chart")

        ts_col = ohlcv["timestamp"]
        if ts_col.dtype == pl.Utf8:
            ts_parsed = ts_col.str.to_datetime()
        else:
            ts_parsed = ts_col.cast(pl.Datetime)

        min_dt = ts_parsed.min()
        max_dt = ts_parsed.max()
        total_bars = len(ohlcv)

        if min_dt is not None and max_dt is not None:
            min_date = min_dt.date()
            max_date = max_dt.date()
            default_end = max_date
            default_start = max(min_date, max_date - timedelta(days=180))

            col_range1, col_range2 = st.columns(2)
            with col_range1:
                start_date = st.date_input(
                    "From",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="_candle_start",
                )
            with col_range2:
                end_date = st.date_input(
                    "To",
                    value=default_end,
                    min_value=min_date,
                    max_value=max_date,
                    key="_candle_end",
                )

            start_str = str(start_date)
            end_str = str(end_date) + " 23:59:59"
            ohlcv_filtered = ohlcv.filter(
                (ts_parsed >= pl.lit(start_str).str.to_datetime())
                & (ts_parsed <= pl.lit(end_str).str.to_datetime())
            )
        else:
            ohlcv_filtered = ohlcv

        if len(ohlcv_filtered) > 0:
            chart, info = build_candlestick_chart(ohlcv_filtered, config)
            _render_chart(chart, height="700px")
            if info["total_bars"] < total_bars:
                st.caption(
                    f"Showing {info['displayed_bars']:,} of {total_bars:,} total bars "
                    f"({len(ohlcv_filtered):,} in selected range)"
                )
            elif info["downsampled"]:
                st.caption(
                    f"Showing {info['displayed_bars']:,} of {info['total_bars']:,} bars "
                    f"(downsampled). Use DataZoom to navigate."
                )
        else:
            st.info("No data in selected date range.")
    else:
        st.info("No OHLCV data available.")

    if features is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Feature Correlation")
            chart = build_correlation_heatmap(features)
            _render_chart(chart, height="600px")
        with col2:
            st.subheader("Label Distribution")
            if labels is not None and "label" in labels.columns:
                chart = build_label_distribution_chart(labels)
                _render_chart(chart, height="500px")
            else:
                st.info("No labels data.")

        st.subheader("Feature Distributions")
        feature_cols = [c for c in features.columns if c not in EXCLUDED_FEATURE_COLS]
        if feature_cols:
            tabs = st.tabs(feature_cols)
            for col, tab in zip(feature_cols, tabs):
                with tab:
                    vals = features[col].drop_nulls().to_numpy()
                    if len(vals) > 0:
                        counts, bin_edges = np.histogram(vals, bins=50)
                        bin_centers = [
                            (bin_edges[i] + bin_edges[i + 1]) / 2
                            for i in range(len(counts))
                        ]
                        x_labels = [f"{v:.2f}" for v in bin_centers]
                        bar = (
                            Bar(init_opts=opts.InitOpts(height="400px"))
                            .add_xaxis(x_labels)
                            .add_yaxis(
                                series_name=col,
                                y_axis=counts.tolist(),
                                label_opts=opts.LabelOpts(is_show=False),
                                itemstyle_opts=opts.ItemStyleOpts(
                                    color=COLORS["primary"]
                                ),
                            )
                            .set_global_opts(
                                title_opts=opts.TitleOpts(title=f"Distribution: {col}"),
                                xaxis_opts=opts.AxisOpts(name=col),
                                yaxis_opts=opts.AxisOpts(name="Count"),
                                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                                datazoom_opts=[opts.DataZoomOpts(type_="inside")],
                            )
                        )
                        _render_chart(bar, height="400px")
                    else:
                        st.info(f"No data for {col}")
    else:
        st.info("No features data available.")


# ===================================================================
# Section: Model Performance
# ===================================================================


def _render_model_section(data: dict, session_dir: str = "") -> None:
    """Render model performance metrics and model-analysis charts."""
    st.markdown("> 🏠 Dashboard > **Model Performance**")
    st.header("Model Performance")

    preds = data.get("predictions")
    fi = data.get("feature_importance", {})

    if preds is not None and len(preds) > 0:
        required_cols = {"true_label", "pred_label"}
        if not required_cols.issubset(set(preds.columns)):
            st.warning(
                f"Predictions missing columns: {required_cols - set(preds.columns)}"
            )
            return

        y_true = preds["true_label"].to_numpy()
        y_pred = preds["pred_label"].to_numpy()
        total = len(y_true)

        exact_acc = float((y_true == y_pred).mean())

        non_hold_mask = (y_true != 0) & (y_pred != 0)
        if non_hold_mask.sum() > 0:
            dir_correct = y_true[non_hold_mask] == y_pred[non_hold_mask]
            dir_acc = float(dir_correct.mean())
            dir_baseline = 0.5
        else:
            dir_acc = 0.0
            dir_baseline = 0.5

        per_class: dict[str, dict[str, float | int]] = {}
        for cls, name in [(-1, "Short"), (0, "Hold"), (1, "Long")]:
            true_mask = y_true == cls
            pred_mask = y_pred == cls
            recall = (
                float((y_pred[true_mask] == cls).mean()) if true_mask.sum() > 0 else 0.0
            )
            precision = (
                float((y_true[pred_mask] == cls).mean()) if pred_mask.sum() > 0 else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class[name] = {
                "true_count": int(true_mask.sum()),
                "pred_count": int(pred_mask.sum()),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }

        with st.container(border=True):
            st.subheader("Accuracy Metrics")
            st.caption("Model prediction accuracy against test set labels")

            acc_cols = st.columns(4, gap="small")
            _render_metric_card(
                acc_cols[0],
                "Directional Accuracy",
                f"{dir_acc:.1%}",
                f"+{(dir_acc - dir_baseline) * 100:.1f}pp vs random",
                "#3b82f6",
            )
            _render_metric_card(
                acc_cols[1],
                "Exact-Match Accuracy",
                f"{exact_acc:.1%}",
                None,
                "#8b5cf6",
            )
            _render_metric_card(
                acc_cols[2],
                "Directional Baseline",
                f"{dir_baseline:.1%}",
                "Random guess baseline",
                "#6b7280",
            )
            _render_metric_card(
                acc_cols[3],
                "Test Samples",
                f"{total:,}",
                None,
                "#10b981",
            )

        st.subheader("Per-Class Performance")
        cls_col1, cls_col2, cls_col3 = st.columns(3)
        for idx, (name, cls_metrics) in enumerate(per_class.items()):
            col = [cls_col1, cls_col2, cls_col3][idx]
            with col:
                st.markdown(f"**{name}**")
                st.caption(
                    f"True: {cls_metrics['true_count']:,} | Predicted: {cls_metrics['pred_count']:,}"
                )
                st.progress(
                    cls_metrics["recall"], text=f"Recall: {cls_metrics['recall']:.1%}"
                )
                st.progress(
                    cls_metrics["precision"],
                    text=f"Precision: {cls_metrics['precision']:.1%}",
                )
                st.progress(cls_metrics["f1"], text=f"F1: {cls_metrics['f1']:.2f}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            chart = build_confusion_matrix_chart(y_true, y_pred)
            _render_chart(chart, height="500px")
        with col2:
            st.subheader("Confidence Distribution")
            chart = build_confidence_distribution_chart(preds)
            _render_chart(chart, height="500px")

        st.subheader("Prediction Distribution")
        pred_counts = {
            "Short": int((y_pred == -1).sum()),
            "Hold": int((y_pred == 0).sum()),
            "Long": int((y_pred == 1).sum()),
        }
        true_counts = {
            "Short": int((y_true == -1).sum()),
            "Hold": int((y_true == 0).sum()),
            "Long": int((y_true == 1).sum()),
        }
        labels_list = list(true_counts.keys())
        actual_vals = [true_counts[k] for k in labels_list]
        predicted_vals = [pred_counts[k] for k in labels_list]
        dist_chart = (
            Bar(init_opts=opts.InitOpts(height="400px"))
            .add_xaxis(labels_list)
            .add_yaxis(
                series_name="Actual",
                y_axis=actual_vals,
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
                label_opts=opts.LabelOpts(is_show=True, position="top"),
            )
            .add_yaxis(
                series_name="Predicted",
                y_axis=predicted_vals,
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["secondary"]),
                label_opts=opts.LabelOpts(is_show=True, position="top"),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Actual vs Predicted Label Distribution"
                ),
                xaxis_opts=opts.AxisOpts(name="Label"),
                yaxis_opts=opts.AxisOpts(name="Count"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                legend_opts=opts.LegendOpts(),
            )
        )
        _render_chart(dist_chart, height="400px")
    else:
        st.info("No predictions data available.")

    if fi:
        st.subheader("LightGBM Feature Importance")
        chart = build_feature_importance_chart(fi)
        _render_chart(chart, height="600px")
    else:
        st.info("No feature importance data available.")

    shap_data = data.get("shap_values")
    if shap_data:
        st.subheader("SHAP Summary")
        chart = build_shap_chart(shap_data)
        _render_chart(chart, height="600px")
    elif session_dir:
        shap_png = Path(session_dir) / "reports" / "shap_summary.png"
        if shap_png.exists():
            st.subheader("SHAP Summary")
            st.image(str(shap_png), width="stretch")


# ===================================================================
# Section: Training
# ===================================================================


def _render_training_section(data: dict, session_dir: str) -> None:
    """Render GRU/LGBM training history and pipeline logs."""
    st.markdown("> 🏠 Dashboard > **Training**")
    st.header("Training History")

    session_path = Path(session_dir)

    history_path = session_path / "models" / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        gru_history = history.get("gru", [])
        lgbm_info = history.get("lightgbm", {})

        if gru_history:
            with st.container(border=True):
                st.subheader("GRU Training Progress")
                st.caption("GRU neural network training curves and best epoch metrics")

                epochs = [e["epoch"] for e in gru_history]
                train_loss = [e["train_loss"] for e in gru_history]
                val_loss = [e["val_loss"] for e in gru_history]
                train_acc = [e["train_acc"] for e in gru_history]
                val_acc = [e["val_acc"] for e in gru_history]

                best_epoch = max(gru_history, key=lambda e: e["val_acc"])
                gru_cols = st.columns(4, gap="small")
                _render_metric_card(
                    gru_cols[0],
                    "Best Val Accuracy",
                    f"{best_epoch['val_acc']:.2%}",
                    f"Epoch {best_epoch['epoch']}",
                    "#22c55e",
                )
                _render_metric_card(
                    gru_cols[1],
                    "Best Epoch",
                    f"{best_epoch['epoch']}",
                    f"Val acc: {best_epoch['val_acc']:.2%}",
                    "#3b82f6",
                )
                _render_metric_card(
                    gru_cols[2],
                    "Final Train Loss",
                    f"{gru_history[-1]['train_loss']:.4f}",
                    f"Started: {gru_history[0]['train_loss']:.4f}",
                    "#f59e0b",
                )
                _render_metric_card(
                    gru_cols[3],
                    "Final Val Loss",
                    f"{gru_history[-1]['val_loss']:.4f}",
                    f"Best: {best_epoch['val_loss']:.4f}",
                    "#ef4444",
                )

            loss_chart = (
                Line(init_opts=opts.InitOpts(height="550px"))
                .add_xaxis([str(e) for e in epochs])
                .add_yaxis(
                    series_name="Train Loss",
                    y_axis=[round(v, 4) for v in train_loss],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["primary"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name="Val Loss",
                    y_axis=[round(v, 4) for v in val_loss],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["danger"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="GRU Loss Curves"),
                    xaxis_opts=opts.AxisOpts(name="Epoch"),
                    yaxis_opts=opts.AxisOpts(name="Loss", is_scale=True),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    legend_opts=opts.LegendOpts(pos_right="right"),
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_show=False,
                            type_="slider",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                        opts.DataZoomOpts(
                            type_="inside",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                    ],
                )
            )
            _render_chart(loss_chart, height="550px")

            acc_chart = (
                Line(init_opts=opts.InitOpts(height="550px"))
                .add_xaxis([str(e) for e in epochs])
                .add_yaxis(
                    series_name="Train Accuracy",
                    y_axis=[round(v, 4) for v in train_acc],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["primary"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name="Val Accuracy",
                    y_axis=[round(v, 4) for v in val_acc],
                    linestyle_opts=opts.LineStyleOpts(width=2, color=COLORS["success"]),
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="GRU Accuracy Curves"),
                    xaxis_opts=opts.AxisOpts(name="Epoch"),
                    yaxis_opts=opts.AxisOpts(
                        name="Accuracy",
                        is_scale=True,
                        max_=1.0 if max(val_acc) > 0.9 else None,
                    ),
                    tooltip_opts=opts.TooltipOpts(trigger="axis"),
                    legend_opts=opts.LegendOpts(pos_right="right"),
                    datazoom_opts=[
                        opts.DataZoomOpts(
                            is_show=False,
                            type_="slider",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                        opts.DataZoomOpts(
                            type_="inside",
                            xaxis_index=0,
                            range_start=0,
                            range_end=100,
                        ),
                    ],
                )
            )
            _render_chart(acc_chart, height="550px")
        else:
            st.info("No GRU training history available.")

        if lgbm_info:
            with st.container(border=True):
                st.subheader("LightGBM Configuration")
                st.caption("Gradient boosting model training parameters and results")

                lgbm_cols = st.columns(3, gap="small")
                _render_metric_card(
                    lgbm_cols[0],
                    "Best Iteration",
                    f"{lgbm_info.get('best_iteration', 'N/A')}",
                    "Optimal boosting round",
                    "#22c55e",
                )
                _render_metric_card(
                    lgbm_cols[1],
                    "Features",
                    f"{lgbm_info.get('n_features', 'N/A')}",
                    "Input feature count",
                    "#3b82f6",
                )
                _render_metric_card(
                    lgbm_cols[2],
                    "Classes",
                    f"{lgbm_info.get('n_classes', 'N/A')}",
                    "Target labels",
                    "#8b5cf6",
                )
    else:
        st.info("No training history file found for this session.")

    st.divider()

    log_path = session_path / "logs" / "pipeline.log"
    if log_path.exists():
        st.subheader("Pipeline Log")

        with open(log_path) as f:
            all_lines = f.readlines()

        with st.expander("Recent Log (last 150 lines)", expanded=True):
            st.code("".join(all_lines[-150:]), language="log")

        with st.expander("Full Pipeline Log", expanded=False):
            st.code("".join(all_lines), language="log")
    else:
        st.info("No pipeline log found for this session.")


# ===================================================================
# Section: Backtest Results
# ===================================================================


def _render_backtest_section(data: dict) -> None:
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

        pnls = [t["pnl"] for t in trades] if trades else []
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_loss_abs = abs(sum(losses) / len(losses)) if losses else 0
        rr = abs(sum(wins) / len(wins)) / avg_loss_abs if avg_loss_abs > 0 else 0.0

        st.markdown("**📊 Key Performance Indicators**")
        kpi_cols = st.columns(5, gap="small")
        _render_zoned_metric(
            kpi_cols[0],
            "Total Return",
            metrics.get("return_pct", 0),
            "return_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[1],
            "Sharpe Ratio",
            metrics.get("sharpe_ratio", 0),
            "sharpe_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            kpi_cols[2],
            "Max Drawdown",
            metrics.get("max_drawdown_pct", 0),
            "max_drawdown_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[3],
            "Win Rate",
            metrics.get("win_rate_pct", 0),
            "win_rate_pct",
            "{:.1f}",
            "%",
        )
        _render_zoned_metric(
            kpi_cols[4], "Trades", metrics.get("num_trades", 0), "num_trades", "{:.0f}"
        )

        st.markdown("---")

        st.markdown("**⚖️ Risk-Adjusted Returns**")
        risk_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            risk_cols[0],
            "Sortino Ratio",
            metrics.get("sortino_ratio", 0),
            "sortino_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            risk_cols[1],
            "Calmar Ratio",
            metrics.get("calmar_ratio", 0),
            "calmar_ratio",
            "{:.2f}",
        )
        _render_zoned_metric(
            risk_cols[2], "SQN", metrics.get("sqn", 0), "sqn", "{:.2f}"
        )
        _render_zoned_metric(
            risk_cols[3],
            "Volatility",
            metrics.get("volatility_ann_pct", 0),
            "volatility_ann_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            risk_cols[4],
            "Recovery Factor",
            metrics.get("recovery_factor", 0),
            "recovery_factor",
            "{:.2f}",
        )

        st.markdown("**💰 Profitability Metrics**")
        profit_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            profit_cols[0],
            "CAGR",
            metrics.get("cagr_pct", 0),
            "cagr_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            profit_cols[1],
            "Annual Return",
            metrics.get("return_ann_pct", 0),
            "return_ann_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            profit_cols[2],
            "Profit Factor",
            metrics.get("profit_factor", 0),
            "profit_factor",
            "{:.2f}",
        )
        _render_zoned_metric(
            profit_cols[3],
            "Avg Trade",
            metrics.get("avg_trade_pct", 0),
            "avg_trade_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            profit_cols[4],
            "Kelly Criterion",
            metrics.get("kelly_criterion", 0),
            "kelly_criterion",
            "{:.1%}",
            "",
        )

        st.markdown("**📈 Trade Analysis**")
        trade_cols = st.columns([1, 1, 1, 1, 1], gap="small")
        _render_zoned_metric(
            trade_cols[0], "Avg Win", metrics.get("avg_win", 0), "avg_win", "${:.0f}"
        )
        _render_zoned_metric(
            trade_cols[1], "Avg Loss", metrics.get("avg_loss", 0), "avg_loss", "${:.0f}"
        )
        _render_zoned_metric(
            trade_cols[2], "Risk/Reward", rr, "risk_reward_ratio", "1:{:.2f}"
        )
        _render_zoned_metric(
            trade_cols[3],
            "Best Trade",
            metrics.get("best_trade_pct", 0),
            "best_trade_pct",
            "{:.2f}",
            "%",
        )
        _render_zoned_metric(
            trade_cols[4],
            "Worst Trade",
            metrics.get("worst_trade_pct", 0),
            "worst_trade_pct",
            "{:.2f}",
            "%",
        )

        st.markdown("**💼 Account Summary**")
        account_cols = st.columns(5, gap="small")
        _render_zoned_metric(
            account_cols[0],
            "Equity Final",
            metrics.get("equity_final", 0),
            "equity_final",
            "${:.0f}",
        )
        _render_zoned_metric(
            account_cols[1],
            "Equity Peak",
            metrics.get("equity_peak", 0),
            "equity_peak",
            "${:.0f}",
        )
        _render_zoned_metric(
            account_cols[2],
            "Commissions",
            metrics.get("commissions", 0),
            "commissions",
            "${:.0f}",
        )
        _render_zoned_metric(
            account_cols[3],
            "Exposure",
            metrics.get("exposure_time_pct", 0),
            "exposure_time_pct",
            "{:.1f}",
            "%",
        )
        account_cols[4].markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #4b556322 0%, #4b556311 100%);
                border-left: 3px solid #4b5563;
                border-radius: 8px;
                padding: 12px 14px;
                margin: 4px 0;
                min-height: 110px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-sizing: border-box;
            ">
                <div>
                    <div style="font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Period</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;">
                        {metrics.get("start", "N/A")[:10]}<br/>→ {metrics.get("end", "N/A")[:10]}
                    </div>
                </div>
                <div style="margin-top: 8px;">
                    <span style="
                        background: #4b556333;
                        color: #9ca3af;
                        padding: 2px 10px;
                        border-radius: 12px;
                        font-size: 0.65rem;
                        font-weight: 700;
                        text-transform: uppercase;
                        letter-spacing: 0.03em;
                    ">Duration</span>
                    <div style="font-size: 0.65rem; color: inherit; opacity: 0.6; margin-top: 4px; line-height: 1.3;">Trading period</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "🟢 Excellent  🟡 Good  🟠 Moderate  🔴 Poor/Dangerous  ⚪ N/A (context-dependent)"
        )

    st.divider()

    st.subheader("Equity Curve & Drawdown")
    chart = build_equity_drawdown_chart(trades, metrics)
    _render_chart(chart, height="600px")

    st.divider()

    st.subheader("Trade PnL Distribution")
    chart = build_pnl_histogram_chart(trades, metrics)
    _render_chart(chart, height="500px")

    st.subheader("Trade Duration vs PnL")
    chart = build_duration_pnl_scatter(trades)
    _render_chart(chart, height="500px")

    st.divider()

    st.subheader("Monthly Returns")
    chart = build_monthly_returns_heatmap(trades)
    _render_chart(chart, height="400px")

    st.divider()

    if trades:
        st.subheader("Individual Trade Returns")
        trade_pnls = [t["pnl"] for t in trades]
        x_labels = [str(i) for i in range(len(trade_pnls))]
        win_pnls = [p if p > 0 else 0 for p in trade_pnls]
        loss_pnls = [p if p <= 0 else 0 for p in trade_pnls]

        returns_chart = (
            Bar(init_opts=opts.InitOpts(height="400px"))
            .add_xaxis(x_labels)
            .add_yaxis(
                series_name="Win",
                y_axis=[round(v, 2) for v in win_pnls],
                stack="pnl",
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["success"]),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="Loss",
                y_axis=[round(v, 2) for v in loss_pnls],
                stack="pnl",
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["danger"]),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Individual Trade Returns"),
                xaxis_opts=opts.AxisOpts(name="Trade #"),
                yaxis_opts=opts.AxisOpts(name="PnL (USD)"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                legend_opts=opts.LegendOpts(is_show=False),
                datazoom_opts=[
                    opts.DataZoomOpts(
                        is_show=False, type_="slider", range_start=0, range_end=100
                    ),
                    opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
                ],
            )
        )
        _render_chart(returns_chart, height="400px")

        st.divider()

        st.subheader("Direction Analysis")
        col_left, col_right = st.columns(2)
        with col_left:
            directions = [t.get("direction", "unknown") for t in trades]
            long_count = directions.count("long")
            short_count = directions.count("short")
            dir_chart = (
                Pie(init_opts=opts.InitOpts(height="400px"))
                .add(
                    series_name="Direction",
                    data_pair=[("Long", long_count), ("Short", short_count)],
                    label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"),
                )
                .set_colors([COLORS["long"], COLORS["short"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="Trade Direction Distribution"),
                    tooltip_opts=opts.TooltipOpts(trigger="item"),
                )
            )
            _render_chart(dir_chart, height="400px")

        with col_right:
            long_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "long")
            short_pnl = sum(t["pnl"] for t in trades if t.get("direction") == "short")
            pnl_dir_chart = (
                Bar(init_opts=opts.InitOpts(height="400px"))
                .add_xaxis(["Long", "Short"])
                .add_yaxis("PnL", [round(long_pnl, 2), round(short_pnl, 2)])
                .set_colors([COLORS["long"], COLORS["short"]])
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="PnL by Direction"),
                    tooltip_opts=opts.TooltipOpts(
                        trigger="axis", formatter="{b}: ${c}"
                    ),
                    xaxis_opts=opts.AxisOpts(type_="category"),
                    yaxis_opts=opts.AxisOpts(
                        axisline_opts=opts.AxisLineOpts(
                            linestyle_opts=opts.LineStyleOpts(is_show=True, opacity=0.5)
                        ),
                    ),
                )
                .set_series_opts(
                    label_opts=opts.LabelOpts(formatter="{b}: ${c}", is_show=True)
                )
            )
            _render_chart(pnl_dir_chart, height="400px")

    if len(trades) > 30:
        st.divider()
        st.subheader("Rolling Metrics")
        chart = build_rolling_sharpe_chart(trades)
        _render_chart(chart, height="400px")

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


# ===================================================================
# Section: Reports
# ===================================================================


def _render_reports_section(session_dir: str) -> None:
    """Render the reports page with thesis markdown and artifact visuals."""
    st.markdown("> 🏠 Dashboard > **Reports**")

    session_path = Path(session_dir)
    reports_dir = session_path / "reports"

    # --- Thesis Report ---
    report_md_path = reports_dir / "thesis_report.md"
    if report_md_path.exists():
        content = report_md_path.read_text()
        section_10_marker = "## 10. Visual Evidence & Analytics"
        if section_10_marker in content:
            content = content.split(section_10_marker)[0]
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
            _render_metric_card(
                summary_cols[0],
                "Windows",
                str(wf_data.get("num_windows", "?")),
                "Total walk-forward windows",
                "#3b82f6",
            )
            _render_metric_card(
                summary_cols[1],
                "OOF Predictions",
                f"{wf_data.get('total_oof_predictions', 0):,}",
                "Out-of-fold prediction count",
                "#10b981",
            )
            _render_metric_card(
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
                        use_container_width=True,
                        hide_index=True,
                    )

    # --- Feature Importance ---
    fi_path = reports_dir / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi_data = json.load(f)
        if fi_data:
            st.divider()
            st.subheader("LightGBM Feature Importance")
            chart = build_feature_importance_chart(fi_data)
            _render_chart(chart, height="600px")

    # --- SHAP (graceful fallback) ---
    shap_json_path = reports_dir / "shap_values.json"
    if shap_json_path.exists():
        with open(shap_json_path) as f:
            shap_data = json.load(f)
        st.divider()
        st.subheader("SHAP Feature Importance")
        chart = build_shap_chart(shap_data)
        _render_chart(chart, height="600px")
    else:
        shap_png = reports_dir / "shap_summary.png"
        if shap_png.exists():
            st.divider()
            st.subheader("SHAP Feature Importance")
            st.image(str(shap_png), width="stretch")


# ===================================================================
# Main entry point
# ===================================================================


def main() -> None:
    """Render the Streamlit dashboard with session selection and navigation.

    Sets up page layout and styling, discovers and loads a selected session
    from the local results directory, and dispatches rendering to the
    appropriate section renderer.
    """
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
        selected = _session_selector_fragment()

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
                if st.button(
                    sec, key=f"nav_{sec}", type=btn_type, use_container_width=True
                ):
                    st.session_state.nav_section = sec
                    st.rerun()

    section = st.session_state.get("nav_section", "📊 Data")

    # ── Load data ──
    session_path = str(Path("results") / selected)
    loaded = _load_config(session_path)
    config = loaded["config"]
    data = loaded["data"]
    metrics = data.get("metrics", {})

    # ── Configuration sidebar ──
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        st.markdown(
            f"**GRU**: hidden={config.gru.hidden_size}, layers={config.gru.num_layers}, "
            f"seq={config.gru.sequence_length}, epochs={config.gru.epochs}"
        )
        st.markdown(
            f"**LightGBM**: leaves={config.model.num_leaves}, "
            f"depth={config.model.max_depth}, lr={config.model.learning_rate}"
        )
        st.markdown(
            f"**Backtest**: leverage={config.backtest.leverage}:1, "
            f"lots={config.backtest.lots_per_trade}, "
            f"conf≥{config.backtest.confidence_threshold}"
        )
        st.markdown(
            f"**Split**: train={config.splitting.train_start[:10]}→"
            f"{config.splitting.train_end[:10]}"
        )

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
        _render_data_section(data, config)
    elif section_name == "Model Performance":
        _render_model_section(data, session_path)
    elif section_name == "Training":
        _render_training_section(data, session_path)
    elif section_name == "Reports":
        _render_reports_section(session_path)
    else:
        _render_backtest_section(data)


if __name__ == "__main__":
    main()
