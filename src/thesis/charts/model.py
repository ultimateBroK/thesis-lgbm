"""Model performance chart builders."""

from __future__ import annotations

import numpy as np
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, HeatMap

from thesis.charts.shared import COLORS


def build_confusion_matrix_chart(
    true: np.ndarray,
    pred: np.ndarray,
) -> HeatMap:
    """Build a normalized confusion-matrix heatmap for 3-class labels.

    Args:
        true: Ground-truth labels encoded as ``-1``, ``0``, or ``1``.
        pred: Predicted labels encoded as ``-1``, ``0``, or ``1``.

    Returns:
        A pyecharts ``HeatMap`` showing row-normalized confusion values.
    """
    labels_order = [-1, 0, 1]
    display_labels = ["Short (-1)", "Hold (0)", "Long (1)"]
    n = len(labels_order)

    # Build raw confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred, strict=True):
        if t in labels_order and p in labels_order:
            ti = labels_order.index(int(t))
            pi = labels_order.index(int(p))
            cm[ti, pi] += 1

    # Normalize by row
    cm_norm = cm.astype(float)
    for i in range(n):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm[i] / row_sum

    data = []
    for i in range(n):
        for j in range(n):
            data.append([j, i, round(float(cm_norm[i, j]), 3)])

    chart = (
        HeatMap(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(display_labels)
        .add_yaxis(
            series_name="Confusion",
            yaxis_data=display_labels,
            value=data,
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Normalized Confusion Matrix (Test Set)"),
            visualmap_opts=opts.VisualMapOpts(
                min_=0,
                max_=1,
                is_calculable=True,
                orient="vertical",
                pos_right="0%",
                pos_top="center",
                range_color=["#FFFFFF", "#93C5FD", "#2563EB"],
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="item",
            ),
        )
    )
    return chart


def build_confidence_distribution_chart(preds_df: pl.DataFrame) -> Bar:
    """Build grouped confidence-distribution bars for long/short predictions.

    Args:
        preds_df: Prediction dataframe containing predicted labels and class
            probabilities.

    Returns:
        A pyecharts ``Bar`` chart with normalized long/short confidence
        distributions, or an empty chart when required columns are missing.
    """
    if "pred_label" not in preds_df.columns:
        return Bar()
    y_pred = preds_df["pred_label"].to_numpy()

    if "pred_proba_class_1" not in preds_df.columns:
        return Bar()

    if "pred_proba_class_minus1" not in preds_df.columns:
        return Bar()

    long_conf = preds_df["pred_proba_class_1"].to_numpy()
    short_conf = preds_df["pred_proba_class_minus1"].to_numpy()

    long_vals = long_conf[y_pred == 1]
    short_vals = short_conf[y_pred == -1]

    # Histogram bins - use 20 bins for cleaner visualization
    bins = np.linspace(0, 1, 21)
    long_counts, _ = np.histogram(long_vals, bins=bins)
    short_counts, _ = np.histogram(short_vals, bins=bins)
    bin_labels = [f"{bins[i]:.2f}" for i in range(len(bins) - 1)]

    # Normalize to relative frequency (percentage) for comparison
    long_total = long_counts.sum()
    short_total = short_counts.sum()
    long_pct = (long_counts / long_total * 100) if long_total > 0 else long_counts
    short_pct = (short_counts / short_total * 100) if short_total > 0 else short_counts

    chart = (
        Bar(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(bin_labels)
        .add_yaxis(
            series_name="Long",
            y_axis=[round(v, 2) for v in long_pct.tolist()],
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["long"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="Short",
            y_axis=[round(v, 2) for v in short_pct.tolist()],
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["short"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Prediction Confidence Distribution"),
            xaxis_opts=opts.AxisOpts(
                name="Confidence", axislabel_opts=opts.LabelOpts(rotate=30)
            ),
            yaxis_opts=opts.AxisOpts(name="Relative Frequency (%)"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="slider",
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            ],
        )
    )
    return chart


def build_feature_importance_chart(
    fi: dict[str, float],
    top_n: int = 20,
) -> Bar:
    """Build a horizontal top-N feature-importance chart.

    Args:
        fi: Mapping from feature name to importance score.
        top_n: Number of top-ranked features to display.

    Returns:
        A pyecharts ``Bar`` chart with stacked GRU/static contributions.
    """
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    items = items[::-1]
    names = [n for n, _ in items]

    # Split into two series: static features and GRU features
    static_values = [v if not n.startswith("gru_") else 0 for n, v in items]
    gru_values = [v if n.startswith("gru_") else 0 for n, v in items]

    chart = (
        Bar(init_opts=opts.InitOpts(height="600px"))
        .add_xaxis(names)
        .add_yaxis(
            series_name="Static Features",
            y_axis=static_values,
            stack="importance",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
        )
        .add_yaxis(
            series_name="GRU Features",
            y_axis=gru_values,
            stack="importance",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["secondary"]),
        )
        .reversal_axis()
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"Feature Importance (Top {top_n})"),
            xaxis_opts=opts.AxisOpts(name="Importance"),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=9)),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )
    return chart


def build_prediction_distribution_chart(
    true: np.ndarray,
    pred: np.ndarray,
) -> Bar:
    """Build actual-vs-predicted label distribution bars."""
    labels = ["Short", "Hold", "Long"]
    label_values = [-1, 0, 1]
    actual_vals = [int((true == value).sum()) for value in label_values]
    predicted_vals = [int((pred == value).sum()) for value in label_values]

    return (
        Bar(init_opts=opts.InitOpts(height="400px"))
        .add_xaxis(labels)
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
            title_opts=opts.TitleOpts(title="Actual vs Predicted Label Distribution"),
            xaxis_opts=opts.AxisOpts(name="Label"),
            yaxis_opts=opts.AxisOpts(name="Count"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )


__all__ = [
    "build_confusion_matrix_chart",
    "build_confidence_distribution_chart",
    "build_feature_importance_chart",
    "build_prediction_distribution_chart",
]
