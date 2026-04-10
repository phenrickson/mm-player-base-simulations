"""Scenarios: research recipes stored as TOML files in `scenarios/`.

A scenario is a named config + hypothesis. Running it produces one
experiment in the experiment tracker. The same scenario can be run
multiple times — each run produces a new experiment (auto-versioned).

TOML schema (all fields except `name` are optional):

    name = "experience_only"
    hypothesis = "Level-based matchmaking: does the feedback loop emerge?"

    [config]
    seed = 1999
    season_days = 30

    [config.population]
    initial_size = 5000
    daily_new_players = 50

    [config.matchmaker]
    kind = "composite"
    composite_weights = {skill = 0.0, experience = 1.0, gear = 0.0}

Only fields explicitly set in the TOML are overridden; everything else
uses `SimulationConfig` defaults.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from mm_sim.config import SimulationConfig
from mm_sim.experiments import Experiment, ExperimentRunner


DEFAULT_SCENARIOS_DIR = Path("scenarios")


@dataclass
class Scenario:
    name: str
    hypothesis: str | None
    config: SimulationConfig

    @classmethod
    def from_toml_file(cls, path: Path) -> "Scenario":
        raw = tomllib.loads(path.read_text())
        if "name" not in raw:
            raise ValueError(f"scenario {path} is missing required field 'name'")
        name = raw["name"]
        hypothesis = raw.get("hypothesis")
        config_dict = raw.get("config", {})
        config = SimulationConfig.model_validate(config_dict)
        return cls(name=name, hypothesis=hypothesis, config=config)


def load_scenario(
    name_or_path: str | Path,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> Scenario:
    """Load a single scenario by name (resolved against scenarios_dir) or path."""
    path = Path(name_or_path)
    if path.suffix != ".toml":
        path = Path(scenarios_dir) / f"{name_or_path}.toml"
    if not path.exists():
        raise FileNotFoundError(f"scenario not found: {path}")
    return Scenario.from_toml_file(path)


def load_scenarios_dir(
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> dict[str, Scenario]:
    """Load every .toml file in the scenarios directory, keyed by scenario name."""
    base = Path(scenarios_dir)
    if not base.exists():
        return {}
    scenarios: dict[str, Scenario] = {}
    for path in sorted(base.glob("*.toml")):
        scenario = Scenario.from_toml_file(path)
        scenarios[scenario.name] = scenario
    return scenarios


def run_scenario(
    scenario: Scenario | str | Path,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    runner: ExperimentRunner | None = None,
) -> Experiment:
    """Run one scenario via the experiment tracker."""
    if not isinstance(scenario, Scenario):
        scenario = load_scenario(scenario, scenarios_dir)
    if runner is None:
        runner = ExperimentRunner()
    return runner.run(
        scenario.config,
        name=scenario.name,
        hypothesis=scenario.hypothesis,
    )


def run_all_scenarios(
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    runner: ExperimentRunner | None = None,
) -> list[Experiment]:
    """Run every scenario in the directory. Returns the produced experiments."""
    scenarios = load_scenarios_dir(scenarios_dir)
    if runner is None:
        runner = ExperimentRunner()
    return [run_scenario(s, runner=runner) for s in scenarios.values()]
