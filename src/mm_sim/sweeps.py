"""Parameter sweeps: run the simulator across a grid of config values.

A Sweep TOML declares a base scenario and either a cross-product (`grid`) or
zipped (`zip`) parameter variation. Each point materializes a full
SimulationConfig via dotted-path overrides, then runs as a normal experiment
saved under experiments/<season>/<sweep_name>/vN/points/.
"""

from __future__ import annotations

from typing import Any


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
