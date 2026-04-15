"""Experiment discovery and cached loading for the Streamlit dashboard.

All public functions accept an explicit `experiments_dir` so they are
testable without Streamlit. The page modules call the module-level
`cached_*` wrappers that add `@st.cache_data`.

Sweep support: a directory under `<season>/<sweep>/v<N>/` that contains
a `sweep.json` is recognized as a sweep. Each of its points is exposed
as a synthetic scenario id `"<sweep>/<point_label>"`. `load_run`
detects the slash and resolves to the sweep-point subdirectory.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import streamlit as st

from mm_sim.experiments import Experiment, ExperimentMetadata, load_experiment
from mm_sim.config import SimulationConfig


def list_seasons(experiments_dir: Path) -> list[str]:
    """Return season directory names, sorted alphabetically."""
    if not experiments_dir.exists():
        return []
    return sorted(
        p.name for p in experiments_dir.iterdir() if p.is_dir()
    )


def _latest_sweep_version_dir(sweep_dir: Path) -> Path | None:
    """Return the vN subdir containing sweep.json (latest version), or None."""
    if not sweep_dir.exists():
        return None
    candidates = [
        p for p in sweep_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        and (p / "sweep.json").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: int(p.name[1:]))[-1]


def _list_sweep_points(sweep_dir: Path) -> list[tuple[str, Path]]:
    """Return [(point_label, point_version_dir), ...] for a sweep, in index order.

    Returns [] if the directory is not a recognizable sweep.
    """
    version_dir = _latest_sweep_version_dir(sweep_dir)
    if version_dir is None:
        return []
    try:
        sweep = json.loads((version_dir / "sweep.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    points = sweep.get("points", [])
    out: list[tuple[str, Path]] = []
    for pt in sorted(points, key=lambda p: p.get("index", 0)):
        label = pt.get("label") or pt.get("experiment_name")
        exp_name = pt.get("experiment_name", label)
        exp_ver = pt.get("experiment_version", "v1")
        if not label:
            continue
        point_dir = version_dir / "points" / exp_name / exp_ver
        if point_dir.exists():
            out.append((label, point_dir))
    return out


def list_scenarios(experiments_dir: Path, season: str) -> list[str]:
    """Return scenario ids under a season.

    Regular scenarios appear as their directory name. Sweeps are
    expanded into one id per point: ``"<sweep_name>/<point_label>"``.
    Sweep points are grouped together and ordered by sweep index.
    """
    season_dir = experiments_dir / season
    if not season_dir.exists():
        return []
    regular: list[str] = []
    sweep_entries: list[str] = []
    for p in sorted(season_dir.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        points = _list_sweep_points(p)
        if points:
            for label, _ in points:
                sweep_entries.append(f"{p.name}/{label}")
        else:
            regular.append(p.name)
    return sorted(regular) + sweep_entries


def _resolve_sweep_point(
    experiments_dir: Path, season: str, scenario: str
) -> Path | None:
    """If `scenario` is a sweep-point id ``"<sweep>/<label>"``, return the
    path to its experiment version directory; otherwise None."""
    if "/" not in scenario:
        return None
    sweep_name, label = scenario.split("/", 1)
    sweep_dir = experiments_dir / season / sweep_name
    for pt_label, pt_dir in _list_sweep_points(sweep_dir):
        if pt_label == label:
            return pt_dir
    return None


def list_versions(
    experiments_dir: Path, season: str, scenario: str
) -> list[str]:
    """Return version directory names (e.g. ['v1', 'v2']) sorted by
    integer suffix ascending. Latest is the last element.

    For sweep-point scenarios, returns the single experiment version
    inside the point directory.
    """
    point_dir = _resolve_sweep_point(experiments_dir, season, scenario)
    if point_dir is not None:
        return [point_dir.name]
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


def _load_experiment_from_dir(exp_dir: Path) -> Experiment:
    """Load an Experiment from an arbitrary version directory."""
    metadata = ExperimentMetadata(
        **json.loads((exp_dir / "metadata.json").read_text())
    )
    cfg = SimulationConfig.model_validate_json(
        (exp_dir / "config.json").read_text()
    )
    aggregate = pl.read_parquet(exp_dir / "aggregate.parquet")
    pop_path = exp_dir / "population.parquet"
    population = pl.read_parquet(pop_path) if pop_path.exists() else None
    match_path = exp_dir / "matches.parquet"
    matches = pl.read_parquet(match_path) if match_path.exists() else None
    return Experiment(
        metadata=metadata,
        config=cfg,
        aggregate=aggregate,
        population=population,
        matches=matches,
    )


def load_run(
    experiments_dir: Path,
    season: str,
    scenario: str,
    version: str,
) -> Experiment:
    """Load a single experiment run.

    If `scenario` is a sweep-point id (``"<sweep>/<label>"``), resolves to
    the point's experiment directory; otherwise delegates to
    `mm_sim.experiments.load_experiment`.
    """
    point_dir = _resolve_sweep_point(experiments_dir, season, scenario)
    if point_dir is not None:
        return _load_experiment_from_dir(point_dir)
    return load_experiment(
        scenario,
        season=season,
        version=version,
        experiments_dir=experiments_dir,
    )


@st.cache_data(show_spinner=False)
def cached_list_seasons(experiments_dir_str: str) -> list[str]:
    return list_seasons(Path(experiments_dir_str))


@st.cache_data(show_spinner=False)
def cached_list_scenarios(experiments_dir_str: str, season: str) -> list[str]:
    return list_scenarios(Path(experiments_dir_str), season)


@st.cache_data(show_spinner=False)
def cached_list_versions(
    experiments_dir_str: str, season: str, scenario: str
) -> list[str]:
    return list_versions(Path(experiments_dir_str), season, scenario)


@st.cache_data(show_spinner="Loading experiment\u2026")
def cached_load_run(
    experiments_dir_str: str,
    season: str,
    scenario: str,
    version: str,
) -> Experiment:
    return load_run(Path(experiments_dir_str), season, scenario, version)
