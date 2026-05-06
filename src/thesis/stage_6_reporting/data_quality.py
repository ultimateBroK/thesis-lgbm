"""Data quality evidence for the thesis report.

Provides functions to compute OHLCV consistency, missing-bar gaps,
label distribution, outlier-return detection, and a combined report
rendered as markdown.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

# OHLCV consistency


def compute_ohlcv_consistency(df: pl.DataFrame) -> dict[str, Any]:
    """Check that OHLCV relationships hold for every row.

    * high >= open, close, low
    * low  <= open, close, high
    * all prices > 0
    """
    total = len(df)
    ohlc_violations = 0
    price_negative = 0

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            price_negative += int((df[col] <= 0).sum())

    if all(c in df.columns for c in ("open", "high", "low", "close")):
        ohlc_violations += int((df["high"] < df["open"]).sum())
        ohlc_violations += int((df["high"] < df["close"]).sum())
        ohlc_violations += int((df["high"] < df["low"]).sum())
        ohlc_violations += int((df["low"] > df["open"]).sum())
        ohlc_violations += int((df["low"] > df["close"]).sum())
        ohlc_violations += int((df["low"] > df["high"]).sum())

    return {
        "total_rows": total,
        "ohlc_violations": ohlc_violations,
        "price_negative_count": price_negative,
        "is_consistent": ohlc_violations == 0 and price_negative == 0,
    }


# Missing-bar statistics

_INTERVAL_TD: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


def compute_missing_bar_stats(
    df: pl.DataFrame, expected_interval: str = "1h"
) -> dict[str, Any]:
    """Analyse gaps between consecutive bars.

    Weekend gaps (Sat–Sun) are expected for crypto-forex hybrids and are
    reported separately rather than counted as missing.
    """
    total_bars = len(df)

    if "timestamp" not in df.columns or total_bars < 2:
        return {
            "total_bars": total_bars,
            "gaps_found": 0,
            "weekend_gaps": 0,
            "missing_ratio": 0.0,
        }

    ts = df["timestamp"].sort()
    diffs = ts.diff().slice(1)  # drop first null

    expected_td = _INTERVAL_TD.get(expected_interval, timedelta(hours=1))
    expected_ms = expected_td.total_seconds() * 1000

    # Convert diffs to milliseconds for comparison
    diff_ms = diffs.dt.total_milliseconds()

    gaps_found = 0
    weekend_gaps = 0

    for i in range(len(diff_ms)):
        gap_ms = diff_ms[i]
        if gap_ms is None or gap_ms <= expected_ms * 1.5:
            continue

        gaps_found += 1

        # Check if gap spans a weekend (Sat=5, Sun=6)
        t_start = ts[i]
        t_end = ts[i + 1]
        # Simple heuristic: if gap >= 48h and spans weekend days
        if gap_ms >= 48 * 3600 * 1000:
            dow_start = t_start.weekday()
            dow_end = t_end.weekday()
            if dow_start >= 5 or dow_end >= 5 or dow_end < dow_start:
                weekend_gaps += 1

    missing_ratio = (gaps_found - weekend_gaps) / total_bars if total_bars > 0 else 0.0

    return {
        "total_bars": total_bars,
        "gaps_found": gaps_found,
        "weekend_gaps": weekend_gaps,
        "missing_ratio": round(max(missing_ratio, 0.0), 6),
    }


# Label distribution


def compute_label_distribution(
    labels: npt.NDArray, classes: list[int] | None = None
) -> dict[str, Any]:
    """Count and percentage of each label class, plus imbalance ratio."""
    if classes is None:
        classes = [-1, 0, 1]

    total = len(labels)
    counts: dict[int, int] = {}
    percentages: dict[int, float] = {}

    for c in classes:
        cnt = int((labels == c).sum())
        counts[c] = cnt
        percentages[c] = round(cnt / total * 100, 2) if total > 0 else 0.0

    non_zero_counts = [counts[c] for c in classes if counts[c] > 0]
    imbalance_ratio = (
        round(max(non_zero_counts) / min(non_zero_counts), 2)
        if len(non_zero_counts) >= 2
        else 0.0
    )

    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "imbalance_ratio": imbalance_ratio,
    }


# Outlier returns


def compute_outlier_returns(
    df: pl.DataFrame, z_threshold: float = 5.0
) -> dict[str, Any]:
    """Flag returns that exceed *z_threshold* standard deviations."""
    if "close" not in df.columns or len(df) < 2:
        return {
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "max_return": 0.0,
            "min_return": 0.0,
            "outlier_dates": [],
        }

    close = df["close"].cast(pl.Float64).to_numpy()
    log_returns = np.diff(np.log(close))
    mean_r = np.mean(log_returns)
    std_r = np.std(log_returns)

    if std_r == 0:
        return {
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "max_return": float(np.max(log_returns)),
            "min_return": float(np.min(log_returns)),
            "outlier_dates": [],
        }

    z_scores = np.abs((log_returns - mean_r) / std_r)
    outlier_mask = z_scores > z_threshold
    outlier_count = int(outlier_mask.sum())

    outlier_dates: list[str] = []
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_list()
        for idx in np.where(outlier_mask)[0]:
            outlier_dates.append(str(ts[idx + 1]))

    return {
        "outlier_count": outlier_count,
        "outlier_ratio": round(outlier_count / len(log_returns), 6),
        "max_return": float(np.max(log_returns)),
        "min_return": float(np.min(log_returns)),
        "outlier_dates": outlier_dates,
    }


# Markdown rendering

_CLASS_NAMES: dict[int, str] = {-1: "Short", 0: "Hold", 1: "Long"}


def render_data_quality_markdown(stats: dict[str, Any]) -> str:
    """Render all quality stats as a markdown section for the report."""
    lines: list[str] = ["## Data Quality Report", ""]

    ohlcv = stats.get("ohlcv_consistency", {})
    lines.append("### OHLCV Consistency")
    lines.append("")
    lines.append(f"- Total rows: {ohlcv.get('total_rows', 'N/A')}")
    lines.append(f"- OHLC violations: {ohlcv.get('ohlc_violations', 'N/A')}")
    lines.append(f"- Negative prices: {ohlcv.get('price_negative_count', 'N/A')}")
    lines.append(f"- Consistent: {'Yes' if ohlcv.get('is_consistent') else 'No'}")
    lines.append("")

    mb = stats.get("missing_bars", {})
    lines.append("### Missing Bar Analysis")
    lines.append("")
    lines.append(f"- Total bars: {mb.get('total_bars', 'N/A')}")
    lines.append(f"- Gaps found: {mb.get('gaps_found', 'N/A')}")
    lines.append(f"- Weekend gaps: {mb.get('weekend_gaps', 'N/A')}")
    lines.append(f"- Missing ratio: {mb.get('missing_ratio', 0.0):.6f}")
    lines.append("")

    lbl = stats.get("label_distribution")
    if lbl:
        lines.append("### Label Distribution")
        lines.append("")
        lines.append(f"- Total samples: {lbl.get('total', 'N/A')}")
        for cls_val, name in _CLASS_NAMES.items():
            cnt = lbl.get("counts", {}).get(cls_val, "N/A")
            pct = lbl.get("percentages", {}).get(cls_val, "N/A")
            lines.append(f"- {name} ({cls_val}): {cnt} ({pct}%)")
        lines.append(f"- Imbalance ratio: {lbl.get('imbalance_ratio', 'N/A')}")
        lines.append("")

    out = stats.get("outlier_returns", {})
    lines.append("### Outlier Returns")
    lines.append("")
    lines.append(f"- Outlier count: {out.get('outlier_count', 'N/A')}")
    lines.append(f"- Outlier ratio: {out.get('outlier_ratio', 0.0):.6f}")
    lines.append(f"- Max return: {out.get('max_return', 0.0):.8f}")
    lines.append(f"- Min return: {out.get('min_return', 0.0):.8f}")
    lines.append("")

    return "\n".join(lines)


# Main entry point


def compute_data_quality_report(
    df: pl.DataFrame, labels: npt.NDArray | None = None
) -> dict[str, Any]:
    """Run all quality checks and return a comprehensive dict."""
    stats: dict[str, Any] = {
        "ohlcv_consistency": compute_ohlcv_consistency(df),
        "missing_bars": compute_missing_bar_stats(df),
        "outlier_returns": compute_outlier_returns(df),
    }

    if labels is not None:
        stats["label_distribution"] = compute_label_distribution(labels)

    stats["markdown"] = render_data_quality_markdown(stats)
    return stats
