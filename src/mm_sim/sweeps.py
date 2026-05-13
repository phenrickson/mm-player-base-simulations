"""Parameter sweeps: run the simulator across a grid of config values.

A Sweep TOML declares a base scenario and either a cross-product (`grid`) or
zipped (`zip`) parameter variation. Each point materializes a full
SimulationConfig via dotted-path overrides, then runs as a normal experiment
saved under experiments/<season>/<sweep_name>/vN/points/.
"""

from __future__ import annotations

import copy
import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import logging

from mm_sim.config import SimulationConfig
from mm_sim.experiments import (
    DEFAULT_EXPERIMENTS_DIR,
    Experiment,
    ExperimentRunner,
)
from mm_sim.scenarios import (
    DEFAULT_SCENARIOS_DIR,
    DEFAULTS_FILENAME,
    _load_defaults_config,
    load_scenario,
    load_season_name,
)

log = logging.getLogger(__name__)


def set_nested(d: dict, path: str, value: Any) -> None:
    """Set d[k1][k2]...[kN] = value, creating intermediate dicts as needed.

    Path is dotted (e.g. "config.matchmaker.composite_weights.skill").
    Mutates d in place.
    """
    if not path:
        raise ValueError("path must be non-empty")
    keys = path.split(".")
    node = d
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _leaf_label(parameter: str) -> str:
    return parameter.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class SweepPoint:
    index: int
    label: str
    overrides: dict[str, Any]


@dataclass
class SweepSpec:
    name: str
    base_scenario: str
    mode: str  # "grid" or "zip"
    parameters: list[str]
    value_lists: list[list[Any]]

    @classmethod
    def from_toml_file(cls, path: Path) -> "SweepSpec":
        raw = tomllib.loads(Path(path).read_text())
        name = raw.get("name")
        if not name:
            raise ValueError(f"sweep {path} missing 'name'")
        base_scenario = raw.get("base_scenario", "defaults")
        sweep = raw.get("sweep", {})
        grid = sweep.get("grid") or []
        zipped = sweep.get("zip") or []
        if bool(grid) == bool(zipped):
            raise ValueError(
                f"sweep {path} must declare exactly one of sweep.grid or sweep.zip"
            )
        mode = "grid" if grid else "zip"
        entries = grid if mode == "grid" else zipped
        parameters = [e["parameter"] for e in entries]
        value_lists = [list(e["values"]) for e in entries]
        if mode == "zip":
            first_len = len(value_lists[0])
            if any(len(v) != first_len for v in value_lists):
                raise ValueError(
                    f"sweep {path}: all zip axes must have equal length"
                )
        return cls(
            name=name,
            base_scenario=base_scenario,
            mode=mode,
            parameters=parameters,
            value_lists=value_lists,
        )

    def iter_points(self):
        if self.mode == "grid":
            combos = itertools.product(*self.value_lists)
        else:
            combos = zip(*self.value_lists)
        for i, combo in enumerate(combos):
            overrides = {p: v for p, v in zip(self.parameters, combo)}
            label_parts = [
                f"{_leaf_label(p)}={_format_value(v)}"
                for p, v in zip(self.parameters, combo)
            ]
            label = f"p{i:04d}_" + "_".join(label_parts)
            yield SweepPoint(index=i, label=label, overrides=overrides)


def materialize_point(base_dict: dict, point: SweepPoint) -> SimulationConfig:
    """Apply the point's overrides onto a deep copy of base_dict, then
    validate as SimulationConfig. base_dict must be shaped as
    `{"config": {...}}` — overrides use full "config.x.y" paths.
    """
    d = copy.deepcopy(base_dict)
    for path, value in point.overrides.items():
        set_nested(d, path, value)
    return SimulationConfig.model_validate(d.get("config", {}))


@dataclass
class SweepResult:
    spec: SweepSpec
    sweep_dir: Path
    point_experiments: list[Experiment]


def _base_config_dict(
    base_scenario: str, scenarios_dir: Path
) -> dict:
    """Resolve the sweep's base scenario to a `{"config": {...}}` dict.

    Overrides use full dotted paths starting with "config." so we wrap
    the validated config into a "config" key to match.
    """
    if base_scenario == "defaults":
        return {"config": _load_defaults_config(scenarios_dir)}
    scenario = load_scenario(base_scenario, scenarios_dir=scenarios_dir)
    return {"config": scenario.config.model_dump()}


def _next_version(parent: Path) -> str:
    if not parent.exists():
        return "v1"
    existing = sorted(
        p.name for p in parent.iterdir() if p.name.startswith("v")
    )
    if not existing:
        return "v1"
    last = existing[-1]
    try:
        return f"v{int(last[1:]) + 1}"
    except ValueError:
        return "v1"


def load_sweep(
    name_or_path: str | Path,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> SweepSpec:
    base = Path(scenarios_dir)
    path = Path(name_or_path)
    if path.suffix != ".toml":
        path = base / f"{name_or_path}.toml"
    if not path.exists():
        raise FileNotFoundError(f"sweep not found: {path}")
    return SweepSpec.from_toml_file(path)


def list_sweeps(
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> list[str]:
    """Names of sweep TOMLs (files in scenarios_dir containing a `[sweep]` table)."""
    base = Path(scenarios_dir)
    if not base.exists():
        return []
    names = []
    for path in sorted(base.glob("*.toml")):
        if path.name == DEFAULTS_FILENAME:
            continue
        try:
            raw = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError:
            continue
        if "sweep" in raw:
            names.append(raw.get("name", path.stem))
    return names


def run_sweep(
    name_or_path: str | Path,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> SweepResult:
    scenarios_dir = Path(scenarios_dir)
    experiments_dir = Path(experiments_dir)
    spec = load_sweep(name_or_path, scenarios_dir=scenarios_dir)
    season = load_season_name(scenarios_dir)
    sweep_parent = experiments_dir / season / spec.name
    version = _next_version(sweep_parent)
    sweep_dir = sweep_parent / version
    (sweep_dir / "points").mkdir(parents=True, exist_ok=True)

    base_dict = _base_config_dict(spec.base_scenario, scenarios_dir)
    point_runner = ExperimentRunner(experiments_dir=sweep_dir / "points")
    experiments: list[Experiment] = []
    points_meta: list[dict] = []
    for point in spec.iter_points():
        sim_cfg = materialize_point(base_dict, point)
        log.info("sweep %s: running %s", spec.name, point.label)
        exp = point_runner.run(sim_cfg, name=point.label)
        experiments.append(exp)
        points_meta.append({
            "index": point.index,
            "label": point.label,
            "overrides": point.overrides,
            "experiment_name": exp.metadata.name,
            "experiment_version": exp.metadata.version,
        })

    metadata = {
        "name": spec.name,
        "base_scenario": spec.base_scenario,
        "mode": spec.mode,
        "parameters": spec.parameters,
        "value_lists": spec.value_lists,
        "points": points_meta,
    }
    (sweep_dir / "sweep.json").write_text(json.dumps(metadata, indent=2))
    return SweepResult(
        spec=spec, sweep_dir=sweep_dir, point_experiments=experiments
    )
