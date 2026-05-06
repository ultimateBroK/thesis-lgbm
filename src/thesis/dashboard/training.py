"""Training section renderer."""

from __future__ import annotations

import json
from pathlib import Path

from pyecharts import options as opts
from pyecharts.charts import Line
import streamlit as st

from thesis.charts import COLORS
from thesis.dashboard.cards import render_metric_card
from thesis.dashboard.shared import render_chart


def render_training_section(data: dict, session_dir: str) -> None:
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
                render_metric_card(
                    gru_cols[0],
                    "Best Val Accuracy",
                    f"{best_epoch['val_acc']:.2%}",
                    f"Epoch {best_epoch['epoch']}",
                    "#22c55e",
                )
                render_metric_card(
                    gru_cols[1],
                    "Best Epoch",
                    f"{best_epoch['epoch']}",
                    f"Val acc: {best_epoch['val_acc']:.2%}",
                    "#3b82f6",
                )
                render_metric_card(
                    gru_cols[2],
                    "Final Train Loss",
                    f"{gru_history[-1]['train_loss']:.4f}",
                    f"Started: {gru_history[0]['train_loss']:.4f}",
                    "#f59e0b",
                )
                render_metric_card(
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
            render_chart(loss_chart, height="550px")

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
            render_chart(acc_chart, height="550px")
        else:
            st.info("No GRU training history available.")

        if lgbm_info:
            with st.container(border=True):
                st.subheader("LightGBM Configuration")
                st.caption("Gradient boosting model training parameters and results")

                lgbm_cols = st.columns(3, gap="small")
                render_metric_card(
                    lgbm_cols[0],
                    "Best Iteration",
                    f"{lgbm_info.get('best_iteration', 'N/A')}",
                    "Optimal boosting round",
                    "#22c55e",
                )
                render_metric_card(
                    lgbm_cols[1],
                    "Features",
                    f"{lgbm_info.get('n_features', 'N/A')}",
                    "Input feature count",
                    "#3b82f6",
                )
                render_metric_card(
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
