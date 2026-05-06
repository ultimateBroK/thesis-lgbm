"""Metric card renderers and CSS constants for the dashboard."""

from __future__ import annotations

import html

from thesis.shared.zones import (
    _ZONE_COLORS,
    _get_metric_zone,
    _is_extreme_value,
)

# CSS style strings extracted to avoid embedding long inline attributes.
CSS_METRIC_LABEL = (
    "font-size: 0.7rem; color: inherit; opacity: 0.7; text-transform: uppercase;"
    " letter-spacing: 0.05em; margin-bottom: 4px;"
)
CSS_METRIC_VALUE = (
    "font-size: 1.5rem; font-weight: 700; color: inherit; line-height: 1.2;"
)
CSS_METRIC_REC = (
    "font-size: 0.65rem; color: inherit; opacity: 0.6;"
    " margin-top: 4px; line-height: 1.3;"
)


def render_zoned_metric(
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
                <div style="{CSS_METRIC_LABEL}">{safe_label}</div>
                <div style="{CSS_METRIC_VALUE}">
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
                <div style="{CSS_METRIC_REC}">{safe_rec}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
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
        f'<div style="{CSS_METRIC_REC}">{html.escape(caption)}</div>' if caption else ""
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
                <div style="{CSS_METRIC_LABEL}">{safe_label}</div>
                <div style="{CSS_METRIC_VALUE}">{safe_value}</div>
            </div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
