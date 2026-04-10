"""Cross-scenario comparison plots within a single season.

Loads the latest version of each named scenario from the current season
and overlays their metrics on shared axes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from mm_sim.experiments import DEFAULT_EXPERIMENTS_DIR, load_experiment
from mm_sim.scenarios import DEFAULT_SCENARIOS_DIR, load_season_name

log = logging.getLogger(__name__)


def compare_scenarios(
    names: list[str] | None = None,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> list[Path]:
    """Generate comparison plots across scenarios in the current season.

    If `names` is None, compares every scenario that has at least one
    saved run in the current season (skipping the `_comparisons`
    output directory).
    """
    season = load_season_name(scenarios_dir)
    season_dir = Path(experiments_dir) / season
    if not season_dir.exists():
        raise FileNotFoundError(
            f"season directory not found: {season_dir}. "
            "Run `just scenarios` first."
        )

    if names is None:
        names = sorted(
            p.name
            for p in season_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
    else:
        names = sorted(names)
    if not names:
        raise ValueError(f"no scenarios found in {season_dir}")

    experiments = [
        load_experiment(name, season=season, experiments_dir=experiments_dir)
        for name in names
    ]

    out_dir = season_dir / "_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    retention_path = out_dir / "retention.png"
    _plot_retention_comparison(experiments, retention_path, season)
    written.append(retention_path)

    log.info("wrote %d comparison plot(s) to %s", len(written), out_dir)
    return written


def _plot_retention_comparison(
    experiments: list, out_path: Path, season: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(experiments), 1)))

    for color, exp in zip(colors, experiments):
        pop = exp.population
        if pop is None:
            continue
        day_zero_ids = pop.filter(pl.col("day") == 0)["player_id"].to_list()
        if not day_zero_ids:
            continue
        cohort = pop.filter(pl.col("player_id").is_in(day_zero_ids))
        retention = (
            cohort.group_by("day")
            .agg(
                (pl.col("active").sum() / pl.lit(len(day_zero_ids))).alias(
                    "retention"
                )
            )
            .sort("day")
        )
        ax.plot(
            retention["day"].to_numpy(),
            retention["retention"].to_numpy(),
            linewidth=2,
            color=color,
            label=exp.metadata.name,
        )

    ax.set_xlabel("day")
    ax.set_ylabel("fraction of day-0 cohort still active")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.suptitle(f"Day-0 cohort retention — {season}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
