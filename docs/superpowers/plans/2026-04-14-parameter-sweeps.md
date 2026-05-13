# Parameter Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the simulator across a grid of parameter values in one command, producing a single grouped experiment with sweep-aware comparison plots, so we can see how a metric varies with a parameter (or pair of parameters) instead of eyeballing N scenario runs.

**Architecture:** A new `Sweep` concept parallel to `Scenario`. A sweep TOML declares a `base_scenario` plus either `sweep.grid` (cross-product of parameter lists) or `sweep.zip` (parallel parameter lists). The runner materializes one full `SimulationConfig` per point by overlaying that point's values onto the base scenario's config, then runs each point sequentially via the existing `ExperimentRunner`. Output goes to `experiments/<season>/<sweep_name>/vN/points/<index>_<labels>/`. A new `compare` entrypoint draws sweep-specific plots (metric vs parameter for 1D, heatmap for 2D).

**Tech Stack:** Python 3, pydantic (config), polars (aggregate frames), matplotlib (plots), pytest.

---

## Design Decisions (locked)

- **Parameter path syntax**: dotted paths walking the pydantic config tree. E.g. `config.matchmaker.composite_weights.skill`, `config.gear.transfer_rate`. Supports both pydantic models and plain dicts (like `composite_weights`).
- **Sweep modes**:
  - `sweep.grid`: list of `{parameter, values}` entries → Cartesian product.
  - `sweep.zip`: list of `{parameter, values}` entries of identical length → element-wise.
  - Exactly one of `grid` / `zip` per sweep file. Error if both or neither.
- **Base scenario**: a `base_scenario = "defaults"` field names another scenario TOML (or `"defaults"` for `defaults.toml` alone). Sweep point values are layered on top via the existing `_deep_merge`.
- **Execution**: sequential. Each point is a normal experiment run via `ExperimentRunner`, saved into a sweep subdirectory.
- **Output layout**:
  ```
  experiments/<season>/<sweep_name>/v1/
    sweep.json                   # sweep metadata (param names, points index)
    points/
      p0000_skill=0.0_gear=0.0/
        aggregate.parquet, population.parquet, matches.parquet, config.json, plots/
      p0001_skill=0.0_gear=0.5/
      ...
  ```
- **Point names**: `pNNNN_<k1>=<v1>_<k2>=<v2>` — zero-padded index for sort order, short labels derived from leaf path segments (`skill`, `gear`) and formatted values (drop trailing zeros).
- **CLI**: `mm_sim sweep <name>`, `mm_sim sweeps` (list), `mm_sim sweep-compare <name>` (generate sweep plots). Wire `just` recipes to match.
- **Plots**:
  - 1D sweep: final-day-of-metric line chart (x = parameter value, one line per metric). Metrics: `active_count`, `true_skill_mean`, `rating_error_mean`, `gear_mean`, `blowouts / matches_played`.
  - 2D sweep: heatmap per metric, axes are the two parameters.
  - Per-point detail plots: reuse existing single-experiment plot machinery; no changes needed.
- **No new plot machinery beyond the sweep-compare entrypoint** — per-point experiments remain first-class and already render via existing `plots.py`.

---

## File Structure

**Create:**
- `src/mm_sim/sweeps.py` — `SweepSpec`, loader, point materializer, runner.
- `src/mm_sim/sweep_plots.py` — sweep-specific plotting (1D line charts, 2D heatmaps).
- `tests/test_sweeps.py` — unit tests for loader, path walker, grid/zip expansion, and a smoke end-to-end run.
- `scenarios/sweep_skill_weight.toml` — first example sweep (1D, varies matchmaker skill weight).
- `scenarios/sweep_skill_gear_grid.toml` — example 2D grid sweep.

**Modify:**
- `src/mm_sim/cli.py` — add `sweep`, `sweeps`, `sweep-compare` subcommands.
- `justfile` — add `sweep`, `sweeps-list`, `sweep-compare` recipes.

---

## Task 1: Parameter path walker

Walking a dotted path into a pydantic-validated config dict. This is the core primitive; everything else builds on it.

**Files:**
- Create: `src/mm_sim/sweeps.py`
- Test: `tests/test_sweeps.py`

- [ ] **Step 1: Failing test**

Create `tests/test_sweeps.py`:
```python
"""Tests for parameter sweeps."""

from __future__ import annotations

import pytest

from mm_sim.sweeps import set_nested


def test_set_nested_single_level():
    d = {"a": 1, "b": 2}
    set_nested(d, "a", 99)
    assert d == {"a": 99, "b": 2}


def test_set_nested_deep():
    d = {"config": {"matchmaker": {"kind": "composite"}}}
    set_nested(d, "config.matchmaker.kind", "random")
    assert d["config"]["matchmaker"]["kind"] == "random"


def test_set_nested_creates_missing_dicts():
    d = {}
    set_nested(d, "config.gear.transfer_rate", 0.1)
    assert d == {"config": {"gear": {"transfer_rate": 0.1}}}


def test_set_nested_into_existing_dict_leaf():
    d = {"config": {"matchmaker": {"composite_weights": {"skill": 1.0, "gear": 0.0}}}}
    set_nested(d, "config.matchmaker.composite_weights.skill", 0.5)
    assert d["config"]["matchmaker"]["composite_weights"]["skill"] == 0.5


def test_set_nested_rejects_empty_path():
    with pytest.raises(ValueError):
        set_nested({}, "", 1)
```

- [ ] **Step 2: Run `uv run pytest tests/test_sweeps.py -v` — expect ImportError.**

- [ ] **Step 3: Implement `set_nested`**

Create `src/mm_sim/sweeps.py`:
```python
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
```

- [ ] **Step 4: Run `uv run pytest tests/test_sweeps.py -v` — 5 pass.**

- [ ] **Step 5: Commit**
  ```bash
  git add src/mm_sim/sweeps.py tests/test_sweeps.py
  git commit -m "feat(sweeps): add set_nested helper for dotted-path config overrides"
  ```

---

## Task 2: Sweep spec + loader

Parse a sweep TOML into a structured `SweepSpec` and expand it into a list of (point_label, overrides_dict) pairs.

**Files:**
- Modify: `src/mm_sim/sweeps.py`
- Test: `tests/test_sweeps.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_sweeps.py`:
```python
def test_sweep_grid_expansion(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    toml.write_text("""
name = "sk_gear"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.0, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.gear"
values = [0.0, 0.5]
""")
    spec = SweepSpec.from_toml_file(toml)
    points = list(spec.iter_points())
    assert len(points) == 4
    # Deterministic order: first axis varies slowest.
    labels = [p.label for p in points]
    assert labels == [
        "p0000_skill=0_gear=0",
        "p0001_skill=0_gear=0.5",
        "p0002_skill=1_gear=0",
        "p0003_skill=1_gear=0.5",
    ]
    # Overrides are dotted-path → value.
    assert points[1].overrides == {
        "config.matchmaker.composite_weights.skill": 0.0,
        "config.matchmaker.composite_weights.gear": 0.5,
    }


def test_sweep_zip_expansion(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    toml.write_text("""
name = "tradeoff"
base_scenario = "defaults"

[[sweep.zip]]
parameter = "config.matchmaker.composite_weights.skill"
values = [1.0, 0.5, 0.0]

[[sweep.zip]]
parameter = "config.matchmaker.composite_weights.gear"
values = [0.0, 0.5, 1.0]
""")
    spec = SweepSpec.from_toml_file(toml)
    points = list(spec.iter_points())
    assert len(points) == 3
    assert points[0].overrides == {
        "config.matchmaker.composite_weights.skill": 1.0,
        "config.matchmaker.composite_weights.gear": 0.0,
    }


def test_sweep_zip_rejects_unequal_lengths(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    toml.write_text("""
name = "bad"
base_scenario = "defaults"

[[sweep.zip]]
parameter = "a.b"
values = [1, 2, 3]

[[sweep.zip]]
parameter = "c.d"
values = [1, 2]
""")
    with pytest.raises(ValueError, match="zip"):
        SweepSpec.from_toml_file(toml)


def test_sweep_requires_exactly_one_mode(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    # Both grid and zip: error.
    toml.write_text("""
name = "bad"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "a.b"
values = [1, 2]

[[sweep.zip]]
parameter = "c.d"
values = [1, 2]
""")
    with pytest.raises(ValueError):
        SweepSpec.from_toml_file(toml)
```

- [ ] **Step 2: Run `uv run pytest tests/test_sweeps.py -v` — 4 new failures.**

- [ ] **Step 3: Implement SweepSpec**

In `src/mm_sim/sweeps.py`, add:
```python
import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        s = f"{v:g}"  # drops trailing zeros, "0.5" not "0.500000"
        return s
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
    mode: str              # "grid" or "zip"
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
```

- [ ] **Step 4: Run `uv run pytest tests/test_sweeps.py -v` — 9 pass total.**

- [ ] **Step 5: Commit**
  ```bash
  git add src/mm_sim/sweeps.py tests/test_sweeps.py
  git commit -m "feat(sweeps): add SweepSpec loader with grid and zip modes"
  ```

---

## Task 3: Materialize a point into a full `SimulationConfig`

Given a `SweepPoint` and the base scenario's config dict, apply the overrides and validate.

**Files:**
- Modify: `src/mm_sim/sweeps.py`
- Test: `tests/test_sweeps.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_sweeps.py`:
```python
def test_materialize_point_applies_overrides():
    from mm_sim.sweeps import SweepPoint, materialize_point

    base_dict = {"config": {"matchmaker": {
        "kind": "composite",
        "composite_weights": {"skill": 1.0, "experience": 0.0, "gear": 0.0},
    }}}
    point = SweepPoint(
        index=0,
        label="p0000_skill=0.5",
        overrides={"config.matchmaker.composite_weights.skill": 0.5},
    )
    cfg = materialize_point(base_dict, point)
    assert cfg.matchmaker.composite_weights == {
        "skill": 0.5, "experience": 0.0, "gear": 0.0
    }


def test_materialize_point_does_not_mutate_base():
    from mm_sim.sweeps import SweepPoint, materialize_point

    base_dict = {"config": {"matchmaker": {
        "kind": "composite",
        "composite_weights": {"skill": 1.0, "gear": 0.0},
    }}}
    point = SweepPoint(
        index=0, label="x",
        overrides={"config.matchmaker.composite_weights.skill": 0.1},
    )
    _ = materialize_point(base_dict, point)
    # Base unchanged
    assert base_dict["config"]["matchmaker"]["composite_weights"]["skill"] == 1.0
```

- [ ] **Step 2: Run `uv run pytest tests/test_sweeps.py -v -k materialize` — 2 new failures.**

- [ ] **Step 3: Implement**

In `src/mm_sim/sweeps.py`, add:
```python
import copy

from mm_sim.config import SimulationConfig


def materialize_point(base_dict: dict, point: SweepPoint) -> SimulationConfig:
    """Apply the point's overrides onto a deep copy of base_dict, then
    validate as SimulationConfig. base_dict must already be shaped like
    `{"config": {...}}` (i.e. wrapped) — overrides use full "config.x.y"
    paths matching that shape.
    """
    d = copy.deepcopy(base_dict)
    for path, value in point.overrides.items():
        set_nested(d, path, value)
    return SimulationConfig.model_validate(d.get("config", d))
```

Note: `set_nested` sets into the raw dict *including* the "config" wrapper; the final validation reaches into `d["config"]`. This matches how scenarios use dotted paths starting with `config.`.

- [ ] **Step 4: Run tests; should be 11 passing.**

- [ ] **Step 5: Commit**
  ```bash
  git add src/mm_sim/sweeps.py tests/test_sweeps.py
  git commit -m "feat(sweeps): materialize SimulationConfig per sweep point"
  ```

---

## Task 4: End-to-end sweep runner

Read a sweep TOML, resolve the base scenario, run each point sequentially via the experiment runner into a dedicated sweep directory, write `sweep.json` metadata.

**Files:**
- Modify: `src/mm_sim/sweeps.py`
- Test: `tests/test_sweeps.py`

- [ ] **Step 1: Failing test**

Append to `tests/test_sweeps.py`:
```python
def test_run_sweep_end_to_end(tmp_path, monkeypatch):
    """Small sweep runs and produces expected output layout."""
    import json
    from mm_sim.sweeps import run_sweep

    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "defaults.toml").write_text("""
season = "test-season"

[config]
seed = 7
season_days = 2

[config.population]
initial_size = 100
daily_new_player_fraction = 0.0
""")
    (scenarios_dir / "sweep_mini.toml").write_text("""
name = "sweep_mini"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.5, 1.0]
""")

    experiments_dir = tmp_path / "experiments"
    result = run_sweep(
        "sweep_mini",
        scenarios_dir=scenarios_dir,
        experiments_dir=experiments_dir,
    )

    sweep_dir = experiments_dir / "test-season" / "sweep_mini" / "v1"
    assert sweep_dir.exists()
    assert (sweep_dir / "sweep.json").exists()
    metadata = json.loads((sweep_dir / "sweep.json").read_text())
    assert metadata["name"] == "sweep_mini"
    assert metadata["mode"] == "grid"
    assert len(metadata["points"]) == 2

    points = list((sweep_dir / "points").iterdir())
    assert len(points) == 2
    for p in points:
        assert (p / "aggregate.parquet").exists()
        assert (p / "config.json").exists()

    assert result.sweep_dir == sweep_dir
    assert len(result.point_experiments) == 2
```

- [ ] **Step 2: Run `uv run pytest tests/test_sweeps.py::test_run_sweep_end_to_end -v` — failure, `run_sweep` doesn't exist.**

- [ ] **Step 3: Implement**

In `src/mm_sim/sweeps.py`, add:
```python
import json
import logging
from dataclasses import dataclass

from mm_sim.experiments import DEFAULT_EXPERIMENTS_DIR, Experiment, ExperimentRunner
from mm_sim.scenarios import (
    DEFAULT_SCENARIOS_DIR,
    DEFAULTS_FILENAME,
    _load_defaults_config,
    load_scenario,
    load_season_name,
)

log = logging.getLogger(__name__)


@dataclass
class SweepResult:
    spec: SweepSpec
    sweep_dir: Path
    point_experiments: list[Experiment]


def _base_config_dict(
    base_scenario: str, scenarios_dir: Path
) -> dict:
    """Resolve base scenario to its raw dict (deep-merged with defaults.toml).

    Returns a {"config": {...}} wrapper so dotted paths like
    "config.matchmaker.xxx" land correctly.
    """
    if base_scenario == "defaults":
        return {"config": _load_defaults_config(scenarios_dir)}
    scenario = load_scenario(base_scenario, scenarios_dir=scenarios_dir)
    # Easiest: round-trip through the validated model.
    return {"config": scenario.config.model_dump()}


def _next_version(parent: Path) -> str:
    if not parent.exists():
        return "v1"
    existing = sorted(p.name for p in parent.iterdir() if p.name.startswith("v"))
    if not existing:
        return "v1"
    last = existing[-1]
    return f"v{int(last[1:]) + 1}"


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
    """Return the names of sweep TOMLs in scenarios_dir (files that parse
    as sweeps — i.e. contain a `[sweep]` table)."""
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

    return SweepResult(spec=spec, sweep_dir=sweep_dir, point_experiments=experiments)
```

- [ ] **Step 4: Run `uv run pytest tests/test_sweeps.py -v` — 12 pass.**

- [ ] **Step 5: Run full suite: `uv run pytest -x`.**

- [ ] **Step 6: Commit**
  ```bash
  git add src/mm_sim/sweeps.py tests/test_sweeps.py
  git commit -m "feat(sweeps): implement sequential sweep runner with sweep.json manifest"
  ```

---

## Task 5: Sweep-aware comparison plots

Render sweep-specific plots from the points' aggregate frames. For 1D sweeps: line charts of final-day metrics vs parameter value. For 2D grid sweeps: heatmaps.

**Files:**
- Create: `src/mm_sim/sweep_plots.py`
- Test: extend `tests/test_sweeps.py` with a smoke test that `plot_sweep` produces PNG files.

- [ ] **Step 1: Failing test**

Append to `tests/test_sweeps.py`:
```python
def test_plot_sweep_1d_produces_plots(tmp_path):
    from mm_sim.sweep_plots import plot_sweep
    from mm_sim.sweeps import run_sweep

    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "defaults.toml").write_text("""
season = "test-season"
[config]
seed = 1
season_days = 2
[config.population]
initial_size = 50
daily_new_player_fraction = 0.0
""")
    (scenarios_dir / "sweep_mini.toml").write_text("""
name = "sweep_mini"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.5, 1.0]
""")
    experiments_dir = tmp_path / "experiments"
    result = run_sweep(
        "sweep_mini", scenarios_dir=scenarios_dir,
        experiments_dir=experiments_dir,
    )
    plot_sweep(result.sweep_dir)
    plots_dir = result.sweep_dir / "plots"
    assert plots_dir.exists()
    assert (plots_dir / "final_metrics.png").exists()


def test_plot_sweep_2d_produces_heatmap(tmp_path):
    from mm_sim.sweep_plots import plot_sweep
    from mm_sim.sweeps import run_sweep

    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "defaults.toml").write_text("""
season = "test-season"
[config]
seed = 1
season_days = 2
[config.population]
initial_size = 50
daily_new_player_fraction = 0.0
""")
    (scenarios_dir / "sweep_grid.toml").write_text("""
name = "sweep_grid"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.5, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.gear"
values = [0.0, 0.5]
""")
    experiments_dir = tmp_path / "experiments"
    result = run_sweep(
        "sweep_grid", scenarios_dir=scenarios_dir,
        experiments_dir=experiments_dir,
    )
    plot_sweep(result.sweep_dir)
    plots_dir = result.sweep_dir / "plots"
    assert (plots_dir / "heatmaps.png").exists()
```

- [ ] **Step 2: Run the new tests; both fail (import missing).**

- [ ] **Step 3: Implement**

Create `src/mm_sim/sweep_plots.py`:
```python
"""Sweep-specific plots: how a metric varies across parameter values."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


METRICS = [
    ("active_count", "Active players (final day)"),
    ("true_skill_mean", "True skill mean (final day)"),
    ("rating_error_mean", "Rating error (final day)"),
    ("gear_mean", "Gear mean (final day)"),
    ("blowout_share", "Blowout share (final day)"),
]


def _final_row(path: Path) -> dict:
    df = pl.read_parquet(path)
    last_day = df.select(pl.col("day").max()).item()
    row = df.filter(pl.col("day") == last_day).to_dicts()[0]
    matches = float(row.get("matches_played") or 0.0)
    blowouts = float(row.get("blowouts") or 0.0)
    row["blowout_share"] = (blowouts / matches) if matches > 0 else float("nan")
    return row


def _load_sweep(sweep_dir: Path) -> tuple[dict, list[tuple[list[float], dict]]]:
    """Return (metadata, list of (param_values, final_row) tuples)."""
    metadata = json.loads((sweep_dir / "sweep.json").read_text())
    points_dir = sweep_dir / "points"
    data: list[tuple[list[float], dict]] = []
    for point in metadata["points"]:
        label = point["label"]
        experiment_name = point["experiment_name"]
        version = point["experiment_version"]
        agg_path = points_dir / experiment_name / version / "aggregate.parquet"
        row = _final_row(agg_path)
        values = [point["overrides"][p] for p in metadata["parameters"]]
        data.append((values, row))
    return metadata, data


def plot_sweep(sweep_dir: Path) -> None:
    sweep_dir = Path(sweep_dir)
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    metadata, data = _load_sweep(sweep_dir)
    if len(metadata["parameters"]) == 1:
        _plot_1d(metadata, data, plots_dir)
    elif len(metadata["parameters"]) == 2 and metadata["mode"] == "grid":
        _plot_2d_heatmap(metadata, data, plots_dir)
    else:
        _plot_1d(metadata, data, plots_dir)  # fall back for zip or >2D: use index


def _plot_1d(metadata, data, plots_dir):
    xs = np.array([d[0][0] for d in data], dtype=float)
    param_label = metadata["parameters"][0].rsplit(".", 1)[-1]
    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(8, 2.5 * len(METRICS)), sharex=True
    )
    for ax, (key, title) in zip(axes, METRICS):
        ys = np.array([d[1].get(key, float("nan")) for d in data], dtype=float)
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker="o", linewidth=2)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(param_label)
    fig.suptitle(f"sweep: {metadata['name']}")
    fig.tight_layout()
    fig.savefig(plots_dir / "final_metrics.png", dpi=120)
    plt.close(fig)


def _plot_2d_heatmap(metadata, data, plots_dir):
    ax_params = metadata["parameters"]
    ax_values = [sorted(set(v)) for v in metadata["value_lists"]]
    x_values, y_values = ax_values
    x_idx = {v: i for i, v in enumerate(x_values)}
    y_idx = {v: i for i, v in enumerate(y_values)}
    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(4 * len(METRICS), 4), squeeze=False
    )
    for ax, (key, title) in zip(axes[0], METRICS):
        grid = np.full((len(y_values), len(x_values)), float("nan"))
        for values, row in data:
            xi = x_idx[values[0]]
            yi = y_idx[values[1]]
            grid[yi, xi] = row.get(key, float("nan"))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([f"{v:g}" for v in x_values], rotation=45)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([f"{v:g}" for v in y_values])
        ax.set_xlabel(ax_params[0].rsplit(".", 1)[-1])
        ax.set_ylabel(ax_params[1].rsplit(".", 1)[-1])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"sweep: {metadata['name']}")
    fig.tight_layout()
    fig.savefig(plots_dir / "heatmaps.png", dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run `uv run pytest tests/test_sweeps.py -v` — 14 pass.**

- [ ] **Step 5: Commit**
  ```bash
  git add src/mm_sim/sweep_plots.py tests/test_sweeps.py
  git commit -m "feat(sweeps): add 1D line and 2D heatmap sweep plots"
  ```

---

## Task 6: CLI + justfile + example sweeps

**Files:**
- Modify: `src/mm_sim/cli.py`, `justfile`
- Create: `scenarios/sweep_skill_weight.toml`, `scenarios/sweep_skill_gear_grid.toml`

- [ ] **Step 1: Add `sweep`, `sweeps`, `sweep-compare` CLI commands**

In `src/mm_sim/cli.py`, find the existing argument parser / command dispatch and add three commands:

- `sweep <name>` → calls `run_sweep(name)`, then `plot_sweep(result.sweep_dir)`; logs the resulting `sweep_dir`.
- `sweeps` → calls `list_sweeps()`, prints each on its own line.
- `sweep-compare <name> [--version vN]` → finds the latest version of that sweep under `experiments/<season>/<name>/`, calls `plot_sweep(sweep_dir)`.

Keep consistency with how existing `scenario`, `scenarios`, `plots` commands are structured in `cli.py`. Do NOT invent new CLI frameworks; match what's already there.

- [ ] **Step 2: Add just recipes**

Append to `justfile`:
```
# run a parameter sweep by name (looks up scenarios/NAME.toml)
sweep NAME:
    uv run python -m mm_sim.cli sweep {{NAME}}

# list sweep files available to run
sweeps-list:
    @uv run python -c "import glob, os, tomllib; names = []; [names.append(os.path.splitext(os.path.basename(p))[0]) for p in sorted(glob.glob('scenarios/*.toml')) if 'sweep' in tomllib.loads(open(p).read())]; print('\n'.join(names) if names else 'no sweep scenarios found')"

# regenerate sweep comparison plots (latest version by default)
sweep-compare NAME *ARGS:
    uv run python -m mm_sim.cli sweep-compare {{NAME}} {{ARGS}}
```

- [ ] **Step 3: Create example sweeps**

Create `scenarios/sweep_skill_weight.toml`:
```toml
name = "sweep_skill_weight"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

Create `scenarios/sweep_skill_gear_grid.toml`:
```toml
name = "sweep_skill_gear_grid"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.0, 0.25, 0.5, 0.75, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.gear"
values = [0.0, 0.25, 0.5, 0.75, 1.0]

[config.outcomes]
# Make gear actually affect outcomes for this sweep (otherwise the gear
# axis does nothing — see 2026-04-14 conversation on gear weight).
# Note: this is a base-scenario override baked into the sweep itself,
# applied to every point via `base_scenario = "defaults"` merging.
```

Wait — the `[config.outcomes]` table at the bottom won't work because the sweep TOML schema we defined doesn't read a `[config]` section. We need one of:
- (a) add base-config merging to the sweep TOML (sweep TOML can override base scenario's config)
- (b) create a separate intermediate scenario that sets `gear_weight = 0.5` and use `base_scenario = "that_scenario"` in the sweep

Option (b) is cleaner — it keeps sweep TOMLs as pure sweep definitions. So:

Create `scenarios/skill_gear_composite.toml` (the base for the 2D sweep):
```toml
name = "skill_gear_composite"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.outcomes]
gear_weight = 0.5

[config.gear]
transfer_enabled = true
```

Then update `scenarios/sweep_skill_gear_grid.toml`:
```toml
name = "sweep_skill_gear_grid"
base_scenario = "skill_gear_composite"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.0, 0.25, 0.5, 0.75, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.gear"
values = [0.0, 0.25, 0.5, 0.75, 1.0]
```

- [ ] **Step 4: Run the 1D sweep to smoke-test**

Run: `just sweep sweep_skill_weight`
Expected: 10 points run (one per value), plots render, logged `sweep_dir` exists.

- [ ] **Step 5: Run full suite + list sweeps**

Run:
```bash
uv run pytest -x
just sweeps-list
```
Expected: all tests pass; `sweeps-list` shows the two new sweep names.

- [ ] **Step 6: Commit**
```bash
git add src/mm_sim/cli.py justfile scenarios/sweep_skill_weight.toml scenarios/sweep_skill_gear_grid.toml scenarios/skill_gear_composite.toml
git commit -m "feat(sweeps): add CLI commands, just recipes, and two example sweeps"
```

---

## Task 7: Final verification

- [ ] **Step 1: Full test suite**

Run: `uv run pytest`
Expected: all tests pass.

- [ ] **Step 2: Run the 2D grid sweep**

Run: `just sweep sweep_skill_gear_grid`
Expected: 25 points complete; `heatmaps.png` renders under the sweep's `plots/`.

- [ ] **Step 3: Eyeball the plots**

Open the produced `heatmaps.png` and `final_metrics.png`. Confirm parameter axes are labeled, metric scales are sensible, heatmap cells have a visible gradient (not uniform).

- [ ] **Step 4: No commit — verification only.**

---

## Out of scope

- **Parallel sweep execution** — sequential only for v1. If runtime becomes a pain we can parallelize via `multiprocessing` or farm points out across processes later.
- **Multi-seed replication per point** — each point is one run at one seed. Adding N seeds per point (for confidence intervals) is a separate feature.
- **Time-series sweep plots** — only final-day metrics for now. "Metric-over-time with one line per parameter value" could be added later.
- **Sweep over ChurnConfig or other nested dicts with validator-dependent fields** — the path walker is generic but some combos may violate pydantic validators; points that fail validation will raise at `materialize_point` time. Fine for v1.
- **Resume / skip-existing** — re-running a sweep always creates a fresh `vN+1` directory. If you want to resume after a crash, that's future work.
