"""Interactive ECharts chart builders for thesis visualization.

Each function builds a pyecharts chart object that can be:
- Rendered via st_pyecharts() in Streamlit (pyecharts charts)
- Exported as HTML via chart.render("path.html")

Usage:
    from thesis.charts import build_candlestick_chart
    chart = build_candlestick_chart(ohlcv_df, config)
    chart.render("candlestick.html")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, HeatMap, Kline, Line, Pie, Scatter, Tab

from thesis.constants import CHART_COLORS, EXCLUDED_FEATURE_COLS

if TYPE_CHECKING:
    from thesis.config import Config

logger = logging.getLogger("thesis.charts")

# --- Constants (single source: thesis.constants) ----------------------------

COLORS: dict[str, str] = CHART_COLORS


# --- Helpers -----------------------------------------------------------------


def _get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return feature columns excluding non-feature metadata fields.

    Args:
        df: Input dataframe.

    Returns:
        Ordered list of feature column names.
    """
    return [c for c in df.columns if c not in EXCLUDED_FEATURE_COLS]


# --- Data Loading ------------------------------------------------------------


def load_session_data(config: Config) -> dict[str, Any]:
    """Load session artifacts required by interactive chart builders.

    Args:
        config: Runtime configuration containing artifact paths.

    Returns:
        Dictionary with loaded dataframes and JSON artifacts used across data,
        model, and backtest chart tabs.
    """
    data: dict[str, Any] = {}

    # Session dir (for download paths)
    data["session_dir"] = config.paths.session_dir

    # OHLCV
    ohlcv_path = Path(config.paths.ohlcv)
    data["ohlcv"] = pl.read_parquet(ohlcv_path) if ohlcv_path.exists() else None

    # Features
    features_path = Path(config.paths.features)
    data["features"] = (
        pl.read_parquet(features_path) if features_path.exists() else None
    )

    # Test data (for manual backtesting)
    test_path = Path(config.paths.test_data)
    data["test"] = pl.read_parquet(test_path) if test_path.exists() else None

    # Labels
    labels_path = Path(config.paths.labels)
    data["labels"] = pl.read_parquet(labels_path) if labels_path.exists() else None

    # Predictions
    if config.paths.session_dir:
        preds_path = (
            Path(config.paths.session_dir) / "predictions" / "final_predictions.parquet"
        )
    else:
        preds_path = Path(config.paths.predictions)
    data["predictions"] = pl.read_parquet(preds_path) if preds_path.exists() else None

    # Backtest results (JSON)
    if config.paths.session_dir:
        bt_path = Path(config.paths.session_dir) / "backtest" / "backtest_results.json"
    else:
        bt_path = Path(config.paths.backtest_results)
    if bt_path.exists():
        with open(bt_path) as f:
            bt = json.load(f)
        data["backtest_results"] = bt
        data["trades"] = bt.get("trades", [])
        data["metrics"] = bt.get("metrics", {})
    else:
        data["backtest_results"] = None
        data["trades"] = []
        data["metrics"] = {}

    # Feature importance (JSON)
    if config.paths.session_dir:
        fi_path = Path(config.paths.session_dir) / "reports" / "feature_importance.json"
    else:
        fi_path = Path("results/feature_importance.json")
    if fi_path.exists():
        with open(fi_path) as f:
            data["feature_importance"] = json.load(f)
    else:
        data["feature_importance"] = {}

    # SHAP values (JSON)
    if config.paths.session_dir:
        shap_path = Path(config.paths.session_dir) / "reports" / "shap_values.json"
    else:
        shap_path = Path("results/shap_values.json")
    if shap_path.exists():
        with open(shap_path) as f:
            data["shap_values"] = json.load(f)
    else:
        data["shap_values"] = None

    logger.info("Session data loaded from %s", config.paths.session_dir or "default")
    return data


# =============================================================================
# Data Exploration Charts (candlestick, correlation, labels, features)
# =============================================================================


def _downsample_ohlcv(df: pl.DataFrame, max_bars: int) -> pl.DataFrame:
    """Reduce an OHLCV DataFrame to at most ``max_bars`` rows.

    Aggregates contiguous rows into fixed-size groups.

    Args:
        df: Input OHLCV time series with ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, and optionally ``volume``.
        max_bars: Maximum number of bars to retain after downsampling.

    Returns:
        Aggregated OHLCV DataFrame with at most *max_bars* rows.
    """
    stride = max(1, len(df) // max_bars)
    group_col = pl.int_range(0, len(df)) // stride
    agg_exprs = [
        pl.col("timestamp").first(),
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
    ]
    if "volume" in df.columns:
        agg_exprs.append(pl.col("volume").sum())
    return (
        df.with_columns(group_col.alias("_group"))
        .group_by("_group", maintain_order=True)
        .agg(*agg_exprs)
        .drop("_group")
    )


def build_candlestick_chart(
    df: pl.DataFrame,
    config: Config,
    max_bars: int = 3000,
) -> tuple[Grid, dict]:
    """Build an interactive OHLCV candlestick chart with stacked volume.

    Expects *df* to contain columns: ``timestamp``, ``open``, ``high``,
    ``low``, ``close``; ``volume`` is optional.  The chart is laid out as
    price (top) and volume (bottom) with a visible slider and inside data zoom.
    Downsamples *df* when its row count exceeds *max_bars*.

    Args:
        df: OHLCV data. ``timestamp`` may be temporal or UTF-8 strings.
        config: Application configuration used for chart title.
        max_bars: Maximum number of bars to render before downsampling.

    Returns:
        A tuple containing the pyecharts ``Grid`` chart and an info dict.
    """
    total_bars = len(df)
    if total_bars > max_bars:
        df = _downsample_ohlcv(df, max_bars)
        downsampled = True
        logger.info(
            "Candlestick: downsampled %d -> %d bars (stride=%d)",
            total_bars,
            len(df),
            max(1, total_bars // max_bars),
        )
    else:
        downsampled = False

    n = len(df)
    logger.info("Candlestick: rendering %d bars", n)

    # Format timestamps — detect whether data has intraday time
    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()
    if ts_col.dtype.is_temporal():
        has_intraday = (ts_col.dt.hour().sum() + ts_col.dt.minute().sum()) > 0
        fmt = "%Y-%m-%d %H:%M" if has_intraday else "%Y-%m-%d"
        timestamps = ts_col.dt.strftime(fmt).to_list()
    else:
        timestamps = ts_col.cast(pl.Utf8).to_list()

    # ECharts candlestick format: [open, close, low, high]
    opens = df["open"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    kline_data = [
        [float(o), float(c), float(lo), float(hi)]
        for o, c, lo, hi in zip(opens, closes, lows, highs, strict=True)
    ]

    # Volume with color
    volumes = df["volume"].to_numpy().astype(float) if "volume" in df.columns else None

    kline = (
        Kline()
        .add_xaxis(xaxis_data=timestamps)
        .add_yaxis(
            series_name=f"{config.data.symbol}",
            y_axis=kline_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color=COLORS["long"],
                color0=COLORS["short"],
                border_color=COLORS["long"],
                border_color0=COLORS["short"],
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{config.data.symbol} Candlestick ({config.data.timeframe})"
            ),
            legend_opts=opts.LegendOpts(
                is_show=False, pos_bottom=10, pos_left="center"
            ),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            xaxis_opts=opts.AxisOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    xaxis_index=[0, 1],
                    range_start=50,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    xaxis_index=[0, 1],
                    type_="slider",
                    pos_top="85%",
                    range_start=50,
                    range_end=100,
                ),
            ],
            visualmap_opts=opts.VisualMapOpts(
                is_show=False,
                dimension=2,
                series_index=5,
                is_piecewise=True,
                pieces=[
                    {"value": 1, "color": COLORS["long"]},
                    {"value": -1, "color": COLORS["short"]},
                ],
            ),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
            brush_opts=opts.BrushOpts(
                x_axis_index="all",
                brush_link="all",
                out_of_brush={"colorAlpha": 0.1},
                brush_type="lineX",
            ),
        )
    )

    # Volume bar chart - single series with color based on price direction
    if volumes is not None:
        # Format: [index, volume, direction]
        volume_data = [
            [i, float(volumes[i]), 1 if closes[i] >= opens[i] else -1]
            for i in range(len(volumes))
        ]
        bar = (
            Bar()
            .add_xaxis(xaxis_data=timestamps)
            .add_yaxis(
                series_name="Volume",
                y_axis=volume_data,
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    grid_index=1,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=1,
                    is_scale=True,
                    split_number=2,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )
    else:
        bar = Bar()

    # Grid with clear separation between price and volume
    grid = (
        Grid(
            init_opts=opts.InitOpts(
                width="1000px",
                height="800px",
                animation_opts=opts.AnimationOpts(animation=False),
            )
        )
        .add(
            kline,
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="50%"),
        )
        .add(
            bar,
            grid_opts=opts.GridOpts(
                pos_left="10%", pos_right="8%", pos_top="63%", height="16%"
            ),
        )
    )

    info = {
        "total_bars": total_bars,
        "displayed_bars": n,
        "downsampled": downsampled,
    }
    return grid, info


def build_correlation_heatmap(df: pl.DataFrame) -> HeatMap:
    """Build an interactive correlation heatmap for feature columns.

    Args:
        df: Feature dataframe used to compute pairwise correlations.

    Returns:
        A pyecharts ``HeatMap`` chart of the correlation matrix.
    """
    feature_cols = _get_feature_cols(df)
    if len(feature_cols) < 2:
        feature_cols = [
            c
            for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

    numeric_df = df.select(feature_cols)
    corr = numeric_df.corr().to_numpy()
    n = len(feature_cols)

    # Flatten to [x, y, value] for pyecharts HeatMap
    data = []
    for i in range(n):
        for j in range(n):
            data.append([j, i, round(float(corr[i, j]), 3)])

    chart = (
        HeatMap(init_opts=opts.InitOpts(height="600px"))
        .add_xaxis(feature_cols)
        .add_yaxis(
            series_name="Correlation",
            yaxis_data=feature_cols,
            value=data,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Feature Correlation Matrix"),
            visualmap_opts=opts.VisualMapOpts(
                min_=-1,
                max_=1,
                is_calculable=True,
                orient="vertical",
                pos_right="0%",
                pos_top="center",
                range_color=["#DC2626", "#FFFFFF", "#2563EB"],
            ),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(rotate=45, font_size=8),
            ),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(font_size=8),
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )
    return chart


def build_label_distribution_chart(df: pl.DataFrame) -> Pie:
    """Build a pie chart for triple-barrier label distribution.

    Args:
        df: Labels dataframe containing a ``label`` column.

    Returns:
        A pyecharts ``Pie`` chart.
    """
    labels = df["label"].to_numpy()
    counts = {k: int((labels == k).sum()) for k in [-1, 0, 1]}

    data_pairs = [
        ("Short (-1)", counts.get(-1, 0)),
        ("Hold (0)", counts.get(0, 0)),
        ("Long (1)", counts.get(1, 0)),
    ]

    chart = (
        Pie(init_opts=opts.InitOpts(height="500px"))
        .add(
            series_name="Labels",
            data_pair=data_pairs,
            radius="75%",
            label_opts=opts.LabelOpts(
                formatter="{name|{b}}\n{count|{c}} {per|{d}%}",
                position="outside",
                rich={
                    "name": {
                        "fontSize": 14,
                        "fontWeight": "bold",
                        "padding": [0, 0, 4, 0],
                    },
                    "count": {
                        "fontSize": 12,
                        "color": "#666",
                    },
                    "per": {
                        "fontSize": 12,
                        "fontWeight": "bold",
                        "color": "#333",
                    },
                },
            ),
        )
        .set_colors([COLORS["short"], COLORS["flat"], COLORS["long"]])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Triple-Barrier Label Distribution"),
            legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"),
            tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} ({d}%)"),
        )
    )
    return chart


def build_feature_distributions_chart(df: pl.DataFrame) -> Tab:
    """Build tabbed histograms for feature distributions.

    Args:
        df: Feature dataframe used to compute per-column histograms.

    Returns:
        A pyecharts ``Tab`` containing one histogram bar chart per feature.
    """
    feature_cols = _get_feature_cols(df)
    tab = Tab()

    for col in feature_cols:
        vals = df[col].drop_nulls().to_numpy()
        if len(vals) == 0:
            continue

        counts, bin_edges = np.histogram(vals, bins=50)
        # Use bin centers as labels
        bin_centers = [
            (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))
        ]
        x_labels = [f"{v:.2f}" for v in bin_centers]

        bar = (
            Bar(init_opts=opts.InitOpts(height="400px"))
            .add_xaxis(x_labels)
            .add_yaxis(
                series_name=col,
                y_axis=counts.tolist(),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color=COLORS["primary"]),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Distribution: {col}"),
                xaxis_opts=opts.AxisOpts(name=col),
                yaxis_opts=opts.AxisOpts(name="Count"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                datazoom_opts=[opts.DataZoomOpts(type_="inside")],
            )
        )
        tab.add(bar, col)

    return tab


# =============================================================================
# Model Performance Charts (confusion matrix, confidence, feature importance)
# =============================================================================


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


def build_shap_chart(shap_data: dict, top_n: int = 20) -> Bar:
    """Build a stacked horizontal SHAP-importance chart by class.

    Args:
        shap_data: SHAP payload containing feature names, class names, and
            mean absolute SHAP arrays.
        top_n: Number of top features to display.

    Returns:
        A pyecharts ``Bar`` chart showing class-wise mean |SHAP| values, or an
        empty chart when SHAP input is incomplete.
    """
    features = shap_data.get("features", [])
    class_names = shap_data.get("class_names", ["Short", "Hold", "Long"])
    mean_abs_shap = shap_data.get("mean_abs_shap", [])

    if not features or not mean_abs_shap:
        return Bar()

    # Validate that SHAP array lengths match feature count
    for cls_idx, cls_vals in enumerate(mean_abs_shap):
        if len(cls_vals) != len(features):
            logger.warning(
                "SHAP class %d has %d values but %d features — skipping chart",
                cls_idx,
                len(cls_vals),
                len(features),
            )
            return Bar()

    # Compute total importance per feature for sorting
    totals = [
        sum(cls_vals[i] for cls_vals in mean_abs_shap) for i in range(len(features))
    ]
    sorted_indices = sorted(
        range(len(features)), key=lambda i: totals[i], reverse=True
    )[:top_n]
    sorted_indices = sorted_indices[
        ::-1
    ]  # Reverse for horizontal bar (bottom = highest)

    sorted_features = [features[i] for i in sorted_indices]
    class_colors = [COLORS["short"], COLORS["flat"], COLORS["long"]]

    chart = Bar(init_opts=opts.InitOpts(height="600px"))
    chart.add_xaxis(sorted_features)

    for cls_idx, cls_name in enumerate(class_names):
        if cls_idx < len(mean_abs_shap):
            cls_vals = mean_abs_shap[cls_idx]
            y_data = [round(cls_vals[i], 4) for i in sorted_indices]
        else:
            y_data = [0] * len(sorted_features)
        chart.add_yaxis(
            series_name=cls_name,
            y_axis=y_data,
            stack="shap",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=class_colors[cls_idx]
                if cls_idx < len(class_colors)
                else COLORS["primary"]
            ),
        )

    chart.reversal_axis().set_global_opts(
        title_opts=opts.TitleOpts(title="SHAP Feature Importance by Class"),
        xaxis_opts=opts.AxisOpts(name="Mean |SHAP Value|"),
        yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=9)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(),
    )
    return chart


# =============================================================================
# Backtest Charts (equity, drawdown, PnL, monthly returns, rolling Sharpe)
# =============================================================================


def build_equity_drawdown_chart(
    trades: list[dict],
    metrics: dict,
    initial_capital: float = 10_000.0,
) -> Grid:
    """Build an equity-curve chart with a drawdown subplot.

    Args:
        trades: Trade records containing at least ``pnl``, with optional times.
        metrics: Backtest metrics used to annotate chart title.
        initial_capital: Starting equity value.

    Returns:
        A pyecharts ``Grid`` containing equity and drawdown charts.
    """
    if not trades or initial_capital <= 0:
        return Grid()

    pnls = [t["pnl"] for t in trades]
    equity = [initial_capital]
    for p in pnls:
        equity.append(equity[-1] + p)

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown_pct = (equity_arr - peak) / peak * 100

    try:
        times = [pd.to_datetime(trades[0]["entry_time"]).strftime("%Y-%m-%d %H:%M")]
        for t in trades:
            times.append(pd.to_datetime(t["exit_time"]).strftime("%Y-%m-%d %H:%M"))
        x_labels = times
    except Exception:
        x_labels = [str(i) for i in range(len(equity))]

    total_trades = metrics.get("total_trades", len(trades))
    total_return = metrics.get("return_pct", 0)

    equity_line = (
        Line()
        .add_xaxis(x_labels)
        .add_yaxis(
            series_name="Equity",
            y_axis=[round(v, 2) for v in equity],
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(width=1.5, color=COLORS["primary"]),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.1),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Equity Curve — {total_trades} trades, {total_return:.2f}% return"
            ),
            yaxis_opts=opts.AxisOpts(name="Equity (USD)", is_scale=True),
            xaxis_opts=opts.AxisOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(is_show=False),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="slider",
                    xaxis_index=[0, 1],
                    range_start=0,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    type_="inside",
                    xaxis_index=[0, 1],
                    range_start=0,
                    range_end=100,
                ),
            ],
        )
    )

    dd_line = (
        Line()
        .add_xaxis(x_labels)
        .add_yaxis(
            series_name="Drawdown",
            y_axis=[round(v, 2) for v in drawdown_pct],
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(width=0.8, color=COLORS["danger"]),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.4, color=COLORS["danger"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(name="Drawdown (%)"),
            xaxis_opts=opts.AxisOpts(name="Trade #"),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    grid = (
        Grid(init_opts=opts.InitOpts(height="600px"))
        .add(equity_line, grid_opts=opts.GridOpts(pos_top="5%", pos_bottom="35%"))
        .add(dd_line, grid_opts=opts.GridOpts(pos_top="73%", pos_bottom="16%"))
    )
    return grid


def build_pnl_histogram_chart(
    trades: list[dict],
    metrics: dict,
) -> Bar:
    """Build a histogram of winning and losing trade PnL values.

    Args:
        trades: Trade records containing numeric ``pnl`` values.
        metrics: Backtest metrics used to annotate averages in chart title.

    Returns:
        A pyecharts ``Bar`` histogram with separate win/loss series.
    """
    if not trades:
        return Bar()

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    all_pnls = np.array(pnls)
    if all_pnls.min() == all_pnls.max():
        # Constant PnL: create a single centred bin so the histogram still renders
        center = all_pnls.min()
        bins = np.array([center - 0.5, center + 0.5])
    else:
        bins = np.linspace(all_pnls.min(), all_pnls.max(), 51)
    win_counts, _ = np.histogram(wins, bins=bins)
    loss_counts, _ = np.histogram(losses, bins=bins)
    bin_labels = [f"{bins[i]:.0f}" for i in range(len(bins) - 1)]

    avg_win = metrics.get("avg_win", np.mean(wins) if wins else 0)
    avg_loss = metrics.get("avg_loss", np.mean(losses) if losses else 0)

    chart = (
        Bar(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis(bin_labels)
        .add_yaxis(
            series_name=f"Wins ({len(wins)})",
            y_axis=win_counts.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["success"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name=f"Losses ({len(losses)})",
            y_axis=loss_counts.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["danger"]),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Trade PnL — Avg Win: ${avg_win:.0f}, Avg Loss: ${avg_loss:.0f}"
            ),
            xaxis_opts=opts.AxisOpts(name="PnL (USD)"),
            yaxis_opts=opts.AxisOpts(name="Count"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(),
        )
    )
    return chart


def _compute_monthly_returns(
    trades: list[dict],
    initial_capital: float = 10_000.0,
) -> dict[tuple[int, int], float]:
    """Compute monthly percentage returns from sequential trades.

    Args:
        trades: Trade records containing ``pnl`` and ``exit_time`` values.
        initial_capital: Starting equity before applying trade PnL.

    Returns:
        Mapping of ``(year, month)`` tuples to monthly return percentages.
    """
    equity = initial_capital
    equity_by_month: dict[tuple[int, int], tuple[float, float]] = {}

    for t in trades:
        try:
            exit_time = datetime.fromisoformat(
                str(t["exit_time"]).replace("Z", "+00:00")
            )
            key = (exit_time.year, exit_time.month)
            start_eq = equity
            equity += t["pnl"]
            end_eq = equity
            if key not in equity_by_month:
                equity_by_month[key] = (start_eq, end_eq)
            else:
                old_start, _old_end = equity_by_month[key]
                equity_by_month[key] = (old_start, end_eq)
        except (ValueError, TypeError):
            continue

    monthly_returns = {}
    for key, (start, end) in equity_by_month.items():
        if start > 0:
            monthly_returns[key] = (end - start) / start * 100

    return monthly_returns


def build_monthly_returns_heatmap(
    trades: list[dict], initial_capital: float = 10_000.0
) -> HeatMap:
    """Build a year-by-month heatmap of monthly returns.

    Args:
        trades: Trade records containing ``pnl`` and ``exit_time`` values.
        initial_capital: Starting equity used for return computation.

    Returns:
        A pyecharts ``HeatMap`` of monthly return percentages.
    """
    monthly = _compute_monthly_returns(trades, initial_capital)
    if not monthly:
        return HeatMap()

    years = sorted(set(k[0] for k in monthly))
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    data = []
    for (yr, mo), ret in monthly.items():
        yi = years.index(yr)
        data.append([mo - 1, yi, round(ret, 2)])

    chart = (
        HeatMap(init_opts=opts.InitOpts(height="400px"))
        .add_xaxis(month_names)
        .add_yaxis(
            series_name="Return",
            yaxis_data=[str(y) for y in years],
            value=data,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Monthly Returns Heatmap"),
            visualmap_opts=opts.VisualMapOpts(
                min_=-5,
                max_=10,
                is_calculable=True,
                orient="vertical",
                pos_right="0%",
                pos_top="center",
                range_color=["#DC2626", "#FDE68A", "#059669"],
            ),
            tooltip_opts=opts.TooltipOpts(trigger="item"),
        )
    )
    return chart


def build_rolling_sharpe_chart(
    trades: list[dict],
    window: int = 30,
) -> Line:
    """Build a rolling annualized Sharpe-ratio line chart.

    Args:
        trades: Ordered trade list containing numeric ``pnl`` values.
        window: Number of trades per rolling Sharpe window.

    Returns:
        A pyecharts ``Line`` chart, or an empty chart when insufficient trades
        are available.
    """
    if len(trades) <= window:
        return Line()

    pnls = np.array([t["pnl"] for t in trades])
    rolling_mean = np.convolve(pnls, np.ones(window) / window, mode="valid")
    rolling_std = np.array(
        [pnls[i : i + window].std() for i in range(len(pnls) - window + 1)]
    )

    try:
        entry = pd.to_datetime(trades[0]["entry_time"])
        exit_ = pd.to_datetime(trades[-1]["exit_time"])
        days = max((exit_ - entry).days, 1)
        trades_per_year = len(trades) / (days / 365.25)
    except Exception:
        trades_per_year = 100  # Fallback

    annualization_factor = np.sqrt(trades_per_year)

    with np.errstate(divide="ignore", invalid="ignore"):
        rolling_sharpe = rolling_mean / rolling_std * annualization_factor
    rolling_sharpe = np.where(rolling_std == 0, np.nan, rolling_sharpe)

    x_labels = [str(i + window) for i in range(len(rolling_sharpe))]

    chart = (
        Line(init_opts=opts.InitOpts(height="400px"))
        .add_xaxis(x_labels)
        .add_yaxis(
            series_name="Rolling Sharpe",
            y_axis=[round(v, 2) for v in rolling_sharpe],
            is_smooth=False,
            linestyle_opts=opts.LineStyleOpts(width=1, color=COLORS["secondary"]),
            label_opts=opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(
                data=[
                    opts.MarkLineItem(
                        y=0,
                        linestyle_opts=opts.LineStyleOpts(color="#333", width=0.5),
                    ),
                    opts.MarkLineItem(
                        y=2,
                        linestyle_opts=opts.LineStyleOpts(
                            color=COLORS["success"], width=1, type_="dashed"
                        ),
                    ),
                ]
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"Rolling Sharpe Ratio (window={window} trades)"
            ),
            xaxis_opts=opts.AxisOpts(name="Trade #"),
            yaxis_opts=opts.AxisOpts(name="Annualized Sharpe"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
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


def build_duration_pnl_scatter(trades: list[dict]) -> Scatter:
    """Build a scatter plot of trade duration versus PnL.

    Args:
        trades: Trade records with ``entry_time``, ``exit_time``, and ``pnl``
            keys.

    Returns:
        A pyecharts ``Scatter`` chart with wins and losses as separate series.
    """
    win_data: list[list[float]] = []
    loss_data: list[list[float]] = []

    for t in trades:
        try:
            entry = datetime.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00"))
            exit_ = datetime.fromisoformat(str(t["exit_time"]).replace("Z", "+00:00"))
            dur_hours = (exit_ - entry).total_seconds() / 3600
            dur = round(dur_hours, 2)
            pnl = round(t["pnl"], 2)
            if t["pnl"] > 0:
                win_data.append([dur, pnl])
            else:
                loss_data.append([dur, pnl])
        except (ValueError, TypeError):
            continue

    if not win_data and not loss_data:
        return Scatter()

    chart = (
        Scatter(init_opts=opts.InitOpts(height="500px"))
        .add_xaxis([])
        .add_yaxis(
            series_name="Wins",
            y_axis=win_data,
            symbol_size=8,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["success"]),
        )
        .add_yaxis(
            series_name="Losses",
            y_axis=loss_data,
            symbol_size=8,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS["danger"]),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Trade Duration vs PnL"),
            xaxis_opts=opts.AxisOpts(
                type_="value",
                name="Duration (hours)",
            ),
            yaxis_opts=opts.AxisOpts(name="PnL (USD)"),
            legend_opts=opts.LegendOpts(),
        )
    )
    return chart


# --- Public API --------------------------------------------------------------

__all__ = [
    # Constants
    "COLORS",
    "EXCLUDED_FEATURE_COLS",
    # Data loading
    "load_session_data",
    # Data charts
    "build_candlestick_chart",
    "build_correlation_heatmap",
    "build_label_distribution_chart",
    "build_feature_distributions_chart",
    # Model charts
    "build_confusion_matrix_chart",
    "build_confidence_distribution_chart",
    "build_feature_importance_chart",
    "build_shap_chart",
    # Backtest charts
    "build_equity_drawdown_chart",
    "build_pnl_histogram_chart",
    "build_monthly_returns_heatmap",
    "build_rolling_sharpe_chart",
    "build_duration_pnl_scatter",
    # Private (for backward compat)
    "_compute_monthly_returns",
    "_get_feature_cols",
]
