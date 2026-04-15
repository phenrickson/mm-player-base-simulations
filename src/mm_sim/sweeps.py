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

from mm_sim.config import SimulationConfig


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
