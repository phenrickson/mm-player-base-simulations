"""Experiment tracking: run simulations, persist artifacts, compare later.

An **experiment** is a run of the simulator with a specific config, saved to
disk as a directory under `experiments/`. Each experiment directory contains:

- metadata.json       — name, hypothesis, timestamp, git SHA, elapsed seconds
- config.json         — the full SimulationConfig as JSON
- aggregate.parquet   — one row per day: active_count, skill percentiles, ...
- population.parquet  — one row per (day, player_id): full per-player state

Experiments are never overwritten. If a name collides, a `_v2`, `_v3`, ...
suffix is appended.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from mm_sim import __version__
from mm_sim.config import SimulationConfig
from mm_sim.engine import SimulationEngine
from mm_sim.plots import generate_plots


DEFAULT_EXPERIMENTS_DIR = Path("experiments")


@dataclass
class ExperimentMetadata:
    name: str
    hypothesis: str | None
    created_at: str
    elapsed_seconds: float
    git_sha: str | None
    mm_sim_version: str
    seed: int
    season_days: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "created_at": self.created_at,
            "elapsed_seconds": self.elapsed_seconds,
            "git_sha": self.git_sha,
            "mm_sim_version": self.mm_sim_version,
            "seed": self.seed,
            "season_days": self.season_days,
        }


@dataclass
class Experiment:
    metadata: ExperimentMetadata
    config: SimulationConfig
    aggregate: pl.DataFrame
    population: pl.DataFrame | None  # None if per-day snapshots weren't saved


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def _auto_name(cfg: SimulationConfig) -> str:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    mm_kind = cfg.matchmaker.kind
    if mm_kind == "composite":
        w = cfg.matchmaker.composite_weights
        parts = [
            f"{k}{v:g}" for k, v in sorted(w.items()) if v > 0
        ]
        weight_slug = "_".join(parts) if parts else "zero"
        return f"{date}_composite_{weight_slug}"
    return f"{date}_{mm_kind}"


def _resolve_experiment_dir(base: Path, name: str) -> tuple[Path, str]:
    """Return a non-colliding directory path. Appends _v2, _v3, ... if needed."""
    candidate = base / name
    if not candidate.exists():
        return candidate, name
    version = 2
    while True:
        versioned_name = f"{name}_v{version}"
        candidate = base / versioned_name
        if not candidate.exists():
            return candidate, versioned_name
        version += 1


class ExperimentRunner:
    """Wraps SimulationEngine with artifact persistence."""

    def __init__(
        self, experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR
    ) -> None:
        self.experiments_dir = Path(experiments_dir)

    def run(
        self,
        cfg: SimulationConfig,
        name: str | None = None,
        hypothesis: str | None = None,
        save_population: bool = True,
    ) -> Experiment:
        resolved_name = name or _auto_name(cfg)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        exp_dir, final_name = _resolve_experiment_dir(
            self.experiments_dir, resolved_name
        )
        exp_dir.mkdir(parents=True, exist_ok=False)

        t0 = time.time()
        engine = SimulationEngine(cfg)
        aggregate = engine.run()
        elapsed = time.time() - t0

        population = (
            engine.snapshot_writer.population_dataframe()
            if save_population
            else None
        )

        metadata = ExperimentMetadata(
            name=final_name,
            hypothesis=hypothesis,
            created_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(elapsed, 3),
            git_sha=_git_sha(),
            mm_sim_version=__version__,
            seed=cfg.seed,
            season_days=cfg.season_days,
        )

        _write_experiment(exp_dir, metadata, cfg, aggregate, population)

        if population is not None:
            generate_plots(
                population=population,
                aggregate=aggregate,
                out_dir=exp_dir / "plots",
                experiment_name=final_name,
            )

        return Experiment(
            metadata=metadata,
            config=cfg,
            aggregate=aggregate,
            population=population,
        )


def _write_experiment(
    exp_dir: Path,
    metadata: ExperimentMetadata,
    cfg: SimulationConfig,
    aggregate: pl.DataFrame,
    population: pl.DataFrame | None,
) -> None:
    (exp_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2)
    )
    (exp_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    aggregate.write_parquet(exp_dir / "aggregate.parquet")
    if population is not None:
        population.write_parquet(exp_dir / "population.parquet")


def load_experiment(
    name: str, experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR
) -> Experiment:
    exp_dir = Path(experiments_dir) / name
    if not exp_dir.exists():
        raise FileNotFoundError(f"experiment not found: {exp_dir}")
    metadata = ExperimentMetadata(**json.loads((exp_dir / "metadata.json").read_text()))
    cfg = SimulationConfig.model_validate_json((exp_dir / "config.json").read_text())
    aggregate = pl.read_parquet(exp_dir / "aggregate.parquet")
    pop_path = exp_dir / "population.parquet"
    population = pl.read_parquet(pop_path) if pop_path.exists() else None
    return Experiment(
        metadata=metadata, config=cfg, aggregate=aggregate, population=population
    )


def list_experiments(
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> pl.DataFrame:
    base = Path(experiments_dir)
    if not base.exists():
        return pl.DataFrame()
    rows: list[dict] = []
    for exp_dir in sorted(base.iterdir()):
        meta_path = exp_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        agg_path = exp_dir / "aggregate.parquet"
        final_active = None
        final_blowouts = None
        if agg_path.exists():
            agg = pl.read_parquet(agg_path)
            if agg.height > 0:
                last = agg.sort("day").tail(1).row(0, named=True)
                final_active = last["active_count"]
                final_blowouts = last["blowouts"]
        rows.append(
            {
                "name": meta["name"],
                "created_at": meta["created_at"],
                "seed": meta.get("seed"),
                "season_days": meta.get("season_days"),
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "final_active_count": final_active,
                "final_day_blowouts": final_blowouts,
                "git_sha": (meta.get("git_sha") or "")[:8],
                "hypothesis": meta.get("hypothesis"),
            }
        )
    return pl.DataFrame(rows)


def compare_experiments(
    names: list[str],
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> pl.DataFrame:
    """Stack the aggregate DataFrames of several experiments with a `name` col."""
    frames: list[pl.DataFrame] = []
    for name in names:
        exp = load_experiment(name, experiments_dir)
        frames.append(exp.aggregate.with_columns(pl.lit(name).alias("experiment")))
    return pl.concat(frames)
