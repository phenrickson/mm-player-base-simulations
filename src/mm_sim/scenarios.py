"""Scenarios: named configs stored as TOML files in `scenarios/`.

Running a scenario produces one experiment in the experiment tracker.
The same scenario can be run multiple times — each run produces a new
experiment (auto-versioned).

TOML schema (all fields except `name` are optional):

    name = "experience_only"

    [config]
    seed = 1999
    season_days = 30

    [config.population]
    initial_size = 5000
    daily_new_player_fraction = 0.01

    [config.matchmaker]
    kind = "composite"
    composite_weights = {skill = 0.0, experience = 1.0, gear = 0.0}

Only fields explicitly set in the TOML are overridden; everything else
uses `SimulationConfig` defaults.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

import shutil

log = logging.getLogger(__name__)

from mm_sim.config import SimulationConfig
from mm_sim.experiments import DEFAULT_EXPERIMENTS_DIR, Experiment, ExperimentRunner


DEFAULT_SCENARIOS_DIR = Path("scenarios")
DEFAULTS_FILENAME = "defaults.toml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with `override` deep-merged into `base`.

    Scalar and list values in `override` win; nested dicts are merged
    recursively. `base` is not mutated.
    """
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_defaults_config(scenarios_dir: Path) -> dict:
    path = scenarios_dir / DEFAULTS_FILENAME
    if not path.exists():
        return {}
    raw = tomllib.loads(path.read_text())
    return raw.get("config", {})


def load_season_name(
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> str:
    """Return the `season` field from scenarios/defaults.toml.

    Raises if defaults.toml doesn't exist or doesn't set `season`.
    """
    path = Path(scenarios_dir) / DEFAULTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — defaults.toml is required to name the "
            "experiment season"
        )
    raw = tomllib.loads(path.read_text())
    season = raw.get("season")
    if not season:
        raise ValueError(
            f"{path} is missing the required top-level `season` field"
        )
    return str(season)


def defaults_toml_path(
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> Path:
    return Path(scenarios_dir) / DEFAULTS_FILENAME


@dataclass
class Scenario:
    name: str
    config: SimulationConfig
    category: str = "other"

    @classmethod
    def from_toml_file(
        cls,
        path: Path,
        defaults_config: dict | None = None,
    ) -> "Scenario":
        raw = tomllib.loads(path.read_text())
        if "name" not in raw:
            raise ValueError(f"scenario {path} is missing required field 'name'")
        name = raw["name"]
        category = raw.get("category", "other")
        scenario_config = raw.get("config", {})
        merged = _deep_merge(defaults_config or {}, scenario_config)
        config = SimulationConfig.model_validate(merged)
        return cls(name=name, config=config, category=category)


def load_scenario(
    name_or_path: str | Path,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> Scenario:
    """Load a single scenario by name (resolved against scenarios_dir) or path.

    Values from `scenarios/defaults.toml` (if present) are deep-merged
    underneath the scenario's own config so individual scenarios only
    need to specify what differs.
    """
    base_dir = Path(scenarios_dir)
    path = Path(name_or_path)
    if path.suffix != ".toml":
        path = base_dir / f"{name_or_path}.toml"
    if not path.exists():
        raise FileNotFoundError(f"scenario not found: {path}")
    defaults_config = _load_defaults_config(base_dir)
    return Scenario.from_toml_file(path, defaults_config=defaults_config)


def load_scenarios_dir(
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> dict[str, Scenario]:
    """Load every .toml file in the scenarios directory, keyed by scenario name.

    `defaults.toml` is not loaded as a scenario; it's the shared-defaults
    file and is merged underneath every other scenario.
    """
    base = Path(scenarios_dir)
    if not base.exists():
        return {}
    defaults_config = _load_defaults_config(base)
    scenarios: dict[str, Scenario] = {}
    for path in sorted(base.glob("*.toml")):
        if path.name == DEFAULTS_FILENAME:
            continue
        raw = tomllib.loads(path.read_text())
        if "sweep" in raw:
            continue  # sweep TOMLs are not scenarios
        scenario = Scenario.from_toml_file(path, defaults_config=defaults_config)
        scenarios[scenario.name] = scenario
    return scenarios


def _season_runner(
    scenarios_dir: Path,
    experiments_dir: Path | str,
) -> tuple[ExperimentRunner, Path]:
    """Build an ExperimentRunner scoped into experiments/<season>/ and
    copy the current defaults.toml into it (first time only).

    Returns (runner, season_dir).
    """
    season = load_season_name(scenarios_dir)
    season_dir = Path(experiments_dir) / season
    season_dir.mkdir(parents=True, exist_ok=True)
    # Copy defaults.toml for transparency — only if missing, so subsequent
    # runs don't silently overwrite an earlier snapshot.
    defaults_src = defaults_toml_path(scenarios_dir)
    defaults_dst = season_dir / "defaults.toml"
    if defaults_src.exists() and not defaults_dst.exists():
        shutil.copy2(defaults_src, defaults_dst)
    return ExperimentRunner(experiments_dir=season_dir), season_dir


def run_scenario(
    scenario: Scenario | str | Path,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
    runner: ExperimentRunner | None = None,
) -> Experiment:
    """Run one scenario via the experiment tracker, scoped into the
    current season directory from scenarios/defaults.toml."""
    scenarios_dir = Path(scenarios_dir)
    if not isinstance(scenario, Scenario):
        scenario = load_scenario(scenario, scenarios_dir)
    if runner is None:
        runner, _ = _season_runner(scenarios_dir, experiments_dir)
    log.info(
        "running scenario: %s (seed=%d, season_days=%d, mm=%s)",
        scenario.name,
        scenario.config.seed,
        scenario.config.season_days,
        scenario.config.matchmaker.kind,
    )
    return runner.run(
        scenario.config,
        name=scenario.name,
    )


def run_all_scenarios(
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
    runner: ExperimentRunner | None = None,
) -> list[Experiment]:
    """Run every scenario in the directory. Returns the produced experiments."""
    scenarios_dir = Path(scenarios_dir)
    scenarios = load_scenarios_dir(scenarios_dir)
    if runner is None:
        runner, _ = _season_runner(scenarios_dir, experiments_dir)
    return [
        run_scenario(s, scenarios_dir=scenarios_dir, runner=runner)
        for s in scenarios.values()
    ]
