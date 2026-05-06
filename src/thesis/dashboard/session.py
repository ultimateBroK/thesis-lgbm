"""Session discovery, parsing, and loading helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import streamlit as st

from thesis.charts import load_session_data
from thesis.shared.session_paths import load_config_for_session


def find_sessions() -> list[Path]:
    """Discover available session directories under ``results/``."""
    results = Path("results")
    if not results.exists():
        return []

    def _parse_session_timestamp(path: Path) -> datetime | None:
        """Parse a session directory name into a datetime."""
        m = re.search(r"(\d{8})_(\d{6})$", path.name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            return None

    sessions = sorted(
        [p for p in results.iterdir() if p.is_dir() and (p / "config").exists()],
        key=lambda p: _parse_session_timestamp(p) or datetime.min,
        reverse=True,
    )
    return sessions


def parse_session_meta(name: str) -> dict[str, str]:
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
def load_config(session_dir: str) -> dict:
    """Load configuration and session data for *session_dir*."""
    config = load_config_for_session(session_dir)
    data = load_session_data(config)
    return {"config": config, "data": data}


@st.fragment(run_every=30)
def session_selector_fragment() -> str | None:
    """Render a sidebar session selector and return the chosen session name."""
    sessions = find_sessions()
    if not sessions:
        return None

    session_names = [s.name for s in sessions]

    known = st.session_state.get("known_sessions", set())
    current_set = set(session_names)
    new_sessions = current_set - known
    if new_sessions and known:
        for ns in sorted(new_sessions):
            meta = parse_session_meta(ns)
            st.toast(f"🆕 New session: {meta['date']} {meta['time']}", icon="📈")
    st.session_state.known_sessions = current_set

    session_labels = []
    for name in session_names:
        meta = parse_session_meta(name)
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
