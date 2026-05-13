"""Experiment tracking: run simulations, persist artifacts, compare later.

An **experiment** is a run of the simulator with a specific config, saved to
disk as a directory under `experiments/`. Each experiment directory contains:

- metadata.json       — name, timestamp, git SHA, elapsed seconds
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
    version: str  # "v1", "v2", ...
    created_at: str
    elapsed_seconds: float
    git_sha: str | None
    mm_sim_version: str
    seed: int
    season_days: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
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
    matches: pl.DataFrame | None = None  # Per-match quality log
    match_teams: pl.DataFrame | None = None  # Per-team-per-match detail (extraction mode)


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


def _next_version_dir(base: Path, name: str) -> tuple[Path, str]:
    """Return (path, version_label) for the next un-used version of an
    experiment. Layout is `<base>/<name>/v<N>/`."""
    name_dir = base / name
    name_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(p.name[1:])
        for p in name_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    )
    next_n = (existing[-1] + 1) if existing else 1
    version = f"v{next_n}"
    return name_dir / version, version


def latest_version_dir(base: Path, name: str) -> Path:
    name_dir = base / name
    if not name_dir.exists():
        raise FileNotFoundError(f"experiment not found: {name_dir}")
    versions = sorted(
        (int(p.name[1:]), p)
        for p in name_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    )
    if not versions:
        raise FileNotFoundError(f"no versions under {name_dir}")
    return versions[-1][1]


def _resolve_version_dir(base: Path, name: str, version: str | None) -> Path:
    name_dir = base / name
    if not name_dir.exists():
        raise FileNotFoundError(f"experiment not found: {name_dir}")
    if version is None:
        return latest_version_dir(base, name)
    candidate = name_dir / version
    if not candidate.exists():
        raise FileNotFoundError(f"version not found: {candidate}")
    return candidate


def _list_season_dirs(experiments_dir: Path) -> list[Path]:
    """Return season directories under experiments/, newest-mtime first."""
    if not experiments_dir.exists():
        return []
    seasons = [p for p in experiments_dir.iterdir() if p.is_dir()]
    seasons.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return seasons


def _find_latest_season_for_experiment(
    experiments_dir: Path, name: str
) -> Path:
    """Return the season directory whose most recent run of `name` is
    newest. Raises FileNotFoundError if `name` doesn't exist in any season."""
    candidates: list[tuple[float, Path]] = []
    for season_dir in _list_season_dirs(experiments_dir):
        name_dir = season_dir / name
        if name_dir.exists() and any(name_dir.iterdir()):
            try:
                latest = latest_version_dir(season_dir, name)
                candidates.append((latest.stat().st_mtime, season_dir))
            except FileNotFoundError:
                continue
    if not candidates:
        raise FileNotFoundError(
            f"experiment {name!r} not found in any season under {experiments_dir}"
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


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
        save_population: bool = True,
    ) -> Experiment:
        resolved_name = name or _auto_name(cfg)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        version_dir, version = _next_version_dir(
            self.experiments_dir, resolved_name
        )
        version_dir.mkdir(parents=True, exist_ok=False)

        t0 = time.time()
        engine = SimulationEngine(cfg, progress_label=resolved_name)
        aggregate = engine.run()
        elapsed = time.time() - t0

        population = (
            engine.snapshot_writer.population_dataframe()
            if save_population
            else None
        )
        matches = engine.snapshot_writer.match_dataframe()
        match_teams = engine.snapshot_writer.match_team_dataframe()

        metadata = ExperimentMetadata(
            name=resolved_name,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(elapsed, 3),
            git_sha=_git_sha(),
            mm_sim_version=__version__,
            seed=cfg.seed,
            season_days=cfg.season_days,
        )

        _write_experiment(
            version_dir, metadata, cfg, aggregate, population, matches, match_teams
        )

        if population is not None:
            generate_plots(
                population=population,
                aggregate=aggregate,
                out_dir=version_dir / "plots",
                experiment_name=f"{resolved_name}/{version}",
            )

        return Experiment(
            metadata=metadata,
            config=cfg,
            aggregate=aggregate,
            population=population,
            matches=matches,
            match_teams=match_teams,
        )


def _write_experiment(
    exp_dir: Path,
    metadata: ExperimentMetadata,
    cfg: SimulationConfig,
    aggregate: pl.DataFrame,
    population: pl.DataFrame | None,
    matches: pl.DataFrame | None = None,
    match_teams: pl.DataFrame | None = None,
) -> None:
    (exp_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2)
    )
    (exp_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    aggregate.write_parquet(exp_dir / "aggregate.parquet")
    if matches is not None:
        matches.write_parquet(exp_dir / "matches.parquet")
    if match_teams is not None:
        match_teams.write_parquet(exp_dir / "match_teams.parquet")
    if population is not None:
        population.write_parquet(exp_dir / "population.parquet")


def load_experiment(
    name: str,
    season: str | None = None,
    version: str | None = None,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> Experiment:
    """Load an experiment by name.

    If `season` is None, picks the season with the most recent run of
    `name` (by mtime). If `version` is None, loads the latest version
    within that season.

    `experiments_dir` can point either at the top `experiments/` directory
    (in which case it's expected to contain season sub-dirs), or directly
    at a season dir (back-compat and for scenario-runner scoping).
    """
    base = Path(experiments_dir)
    if season is not None:
        season_dir = base / season
        if not season_dir.exists():
            raise FileNotFoundError(f"season not found: {season_dir}")
    else:
        # Auto-pick: first try treating `base` as a season directly
        if (base / name).exists():
            season_dir = base
        else:
            season_dir = _find_latest_season_for_experiment(base, name)
    version_dir = _resolve_version_dir(season_dir, name, version)
    metadata = ExperimentMetadata(
        **json.loads((version_dir / "metadata.json").read_text())
    )
    cfg = SimulationConfig.model_validate_json(
        (version_dir / "config.json").read_text()
    )
    aggregate = pl.read_parquet(version_dir / "aggregate.parquet")
    pop_path = version_dir / "population.parquet"
    population = pl.read_parquet(pop_path) if pop_path.exists() else None
    match_path = version_dir / "matches.parquet"
    matches = pl.read_parquet(match_path) if match_path.exists() else None
    mt_path = version_dir / "match_teams.parquet"
    match_teams = pl.read_parquet(mt_path) if mt_path.exists() else None
    return Experiment(
        metadata=metadata,
        config=cfg,
        aggregate=aggregate,
        population=population,
        matches=matches,
        match_teams=match_teams,
    )


def list_experiments(
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> pl.DataFrame:
    """One row per (season, name, version). Columns: season, name, version, ..."""
    base = Path(experiments_dir)
    if not base.exists():
        return pl.DataFrame()
    rows: list[dict] = []
    for season_dir in sorted(base.iterdir()):
        if not season_dir.is_dir():
            continue
        for name_dir in sorted(season_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            for version_dir in sorted(name_dir.iterdir()):
                meta_path = version_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                agg_path = version_dir / "aggregate.parquet"
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
                        "season": season_dir.name,
                        "name": meta["name"],
                        "version": meta.get("version", version_dir.name),
                        "created_at": meta["created_at"],
                        "seed": meta.get("seed"),
                        "season_days": meta.get("season_days"),
                        "elapsed_seconds": meta.get("elapsed_seconds"),
                        "final_active_count": final_active,
                        "final_day_blowouts": final_blowouts,
                        "git_sha": (meta.get("git_sha") or "")[:8],
                    }
                )
    return pl.DataFrame(rows)


def compare_experiments(
    names: list[str],
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> pl.DataFrame:
    """Stack the aggregate DataFrames of several experiments (latest version
    of each) with `experiment` and `version` columns."""
    frames: list[pl.DataFrame] = []
    for name in names:
        exp = load_experiment(name, experiments_dir=experiments_dir)
        frames.append(
            exp.aggregate.with_columns(
                pl.lit(name).alias("experiment"),
                pl.lit(exp.metadata.version).alias("version"),
            )
        )
    return pl.concat(frames)
