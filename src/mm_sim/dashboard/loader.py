"""Experiment discovery and cached loading for the Streamlit dashboard.

All public functions accept an explicit `experiments_dir` so they are
testable without Streamlit. The page modules call the module-level
`cached_*` wrappers that add `@st.cache_data`.
"""
from __future__ import annotations

from pathlib import Path


def list_seasons(experiments_dir: Path) -> list[str]:
    """Return season directory names, sorted alphabetically."""
    if not experiments_dir.exists():
        return []
    return sorted(
        p.name for p in experiments_dir.iterdir() if p.is_dir()
    )
