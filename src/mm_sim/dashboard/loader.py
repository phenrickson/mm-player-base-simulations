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


def list_scenarios(experiments_dir: Path, season: str) -> list[str]:
    """Return scenario directory names under a season, sorted."""
    season_dir = experiments_dir / season
    if not season_dir.exists():
        return []
    return sorted(
        p.name
        for p in season_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def list_versions(
    experiments_dir: Path, season: str, scenario: str
) -> list[str]:
    """Return version directory names (e.g. ['v1', 'v2']) sorted by
    integer suffix ascending. Latest is the last element."""
    scen_dir = experiments_dir / season / scenario
    if not scen_dir.exists():
        return []
    versions = [
        p.name
        for p in scen_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    return sorted(versions, key=lambda v: int(v[1:]))


def latest_version(
    experiments_dir: Path, season: str, scenario: str
) -> str:
    """Return the highest-numbered version under (season, scenario).

    Raises FileNotFoundError if none exist.
    """
    versions = list_versions(experiments_dir, season, scenario)
    if not versions:
        raise FileNotFoundError(
            f"no versions under {experiments_dir / season / scenario}"
        )
    return versions[-1]
