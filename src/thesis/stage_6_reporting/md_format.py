"""Shared Markdown formatting helpers for stage 6 reporting.

This module centralizes tiny formatting helpers that were previously duplicated
across report-generation modules. Keep these functions stable: many report
sections rely on their exact output.
"""

from __future__ import annotations


def _tbl_row(*cells: str) -> str:
    """Format cells as a markdown table row."""
    return "| " + " | ".join(cells) + " |"


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def _fmt_f2(v: float) -> str:
    return f"{v:.2f}"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.0f}"


__all__ = ["_tbl_row", "_fmt_pct", "_fmt_f2", "_fmt_dollar"]
