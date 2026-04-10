"""Per-experiment visualizations.

Generates PNG plots from an Experiment's population and aggregate data
and writes them to `<experiment_dir>/plots/`. Called automatically by
`ExperimentRunner.run()` and also available as a standalone entry point
via `generate_plots_for_experiment(name)` (see `just plots NAME`).

Plots produced (one PNG each plus a 2x2 `overview.png`):
- skill_distribution_intervals.png — histograms at 5 day checkpoints
- population_over_time.png         — active count per day
- retention_over_time.png          — day-0 cohort retention curve
- retention_by_skill_decile.png    — retention per day-0 skill decile
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")  # headless backend — no Tk needed
import matplotlib.pyplot as plt  # noqa: E402


PLOT_FILENAMES = {
    "skill_dist": "skill_distribution_intervals.png",
    "population": "population_over_time.png",
    "retention": "retention_over_time.png",
    "retention_by_decile": "retention_by_skill_decile.png",
    "overview": "overview.png",
}


def generate_plots(
    population: pl.DataFrame,
    aggregate: pl.DataFrame,
    out_dir: Path,
    experiment_name: str = "",
) -> list[Path]:
    """Generate all plots for one experiment and return the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    checkpoint_days = _pick_checkpoint_days(aggregate)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    _plot_skill_distribution_intervals(ax1, population, checkpoint_days)
    fig1.suptitle(_title("Skill distribution across the season", experiment_name))
    path1 = out_dir / PLOT_FILENAMES["skill_dist"]
    fig1.tight_layout()
    fig1.savefig(path1, dpi=120)
    plt.close(fig1)
    written.append(path1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    _plot_population_over_time(ax2, aggregate)
    fig2.suptitle(_title("Active player population over time", experiment_name))
    path2 = out_dir / PLOT_FILENAMES["population"]
    fig2.tight_layout()
    fig2.savefig(path2, dpi=120)
    plt.close(fig2)
    written.append(path2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    _plot_retention_over_time(ax3, population)
    fig3.suptitle(_title("Day-0 cohort retention over time", experiment_name))
    path3 = out_dir / PLOT_FILENAMES["retention"]
    fig3.tight_layout()
    fig3.savefig(path3, dpi=120)
    plt.close(fig3)
    written.append(path3)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    _plot_retention_by_skill_decile(ax4, population)
    fig4.suptitle(_title("Retention by day-0 true-skill decile", experiment_name))
    path4 = out_dir / PLOT_FILENAMES["retention_by_decile"]
    fig4.tight_layout()
    fig4.savefig(path4, dpi=120)
    plt.close(fig4)
    written.append(path4)

    # 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_skill_distribution_intervals(axes[0, 0], population, checkpoint_days)
    _plot_population_over_time(axes[0, 1], aggregate)
    _plot_retention_over_time(axes[1, 0], population)
    _plot_retention_by_skill_decile(axes[1, 1], population)
    axes[0, 0].set_title("Skill distribution across the season")
    axes[0, 1].set_title("Active population over time")
    axes[1, 0].set_title("Day-0 cohort retention")
    axes[1, 1].set_title("Retention by day-0 skill decile")
    fig.suptitle(_title("Overview", experiment_name), fontsize=14)
    path5 = out_dir / PLOT_FILENAMES["overview"]
    fig.tight_layout()
    fig.savefig(path5, dpi=120)
    plt.close(fig)
    written.append(path5)

    return written


def _title(base: str, experiment_name: str) -> str:
    if experiment_name:
        return f"{base} — {experiment_name}"
    return base


def _pick_checkpoint_days(aggregate: pl.DataFrame) -> list[int]:
    days = aggregate["day"].to_list()
    if not days:
        return []
    n = len(days)
    if n <= 5:
        return days
    picks = sorted(
        {
            days[0],
            days[math.ceil(n / 4) - 1],
            days[math.ceil(n / 2) - 1],
            days[math.ceil(3 * n / 4) - 1],
            days[-1],
        }
    )
    return picks


def _plot_skill_distribution_intervals(
    ax, population: pl.DataFrame, checkpoint_days: list[int]
) -> None:
    """Overlapping density-style histograms of active-player true_skill at
    each checkpoint day."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, max(len(checkpoint_days), 1)))
    x_all = population.filter(pl.col("active"))["true_skill"].to_numpy()
    if x_all.size == 0:
        ax.text(0.5, 0.5, "no active players", ha="center", va="center")
        return
    bins = np.linspace(float(x_all.min()) - 0.1, float(x_all.max()) + 0.1, 40)

    for color, day in zip(colors, checkpoint_days):
        day_slice = population.filter(
            (pl.col("day") == day) & pl.col("active")
        )["true_skill"].to_numpy()
        if day_slice.size == 0:
            continue
        ax.hist(
            day_slice,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=f"day {day} (n={day_slice.size})",
        )
    ax.set_xlabel("true_skill")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_population_over_time(ax, aggregate: pl.DataFrame) -> None:
    days = aggregate["day"].to_numpy()
    active = aggregate["active_count"].to_numpy()
    ax.plot(days, active, marker="o", linewidth=2, color="#2b7fff")
    ax.set_xlabel("day")
    ax.set_ylabel("active players")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def _plot_retention_over_time(ax, population: pl.DataFrame) -> None:
    """Fraction of the day-0 player set that is still active on each day."""
    day_zero = population.filter(pl.col("day") == 0)
    day_zero_ids = day_zero["player_id"].to_list()
    if not day_zero_ids:
        ax.text(0.5, 0.5, "no day-0 players", ha="center", va="center")
        return

    cohort = population.filter(pl.col("player_id").is_in(day_zero_ids))
    retention = (
        cohort.group_by("day")
        .agg(
            (pl.col("active").sum() / pl.lit(len(day_zero_ids))).alias("retention")
        )
        .sort("day")
    )
    ax.plot(
        retention["day"].to_numpy(),
        retention["retention"].to_numpy(),
        marker="o",
        linewidth=2,
        color="#d62728",
    )
    ax.set_xlabel("day")
    ax.set_ylabel("fraction of day-0 cohort still active")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)


def _plot_retention_by_skill_decile(ax, population: pl.DataFrame) -> None:
    """Retention curves broken down by day-0 true_skill decile."""
    day_zero = population.filter(pl.col("day") == 0)
    if day_zero.height == 0:
        ax.text(0.5, 0.5, "no day-0 players", ha="center", va="center")
        return

    skills = day_zero["true_skill"].to_numpy()
    ids = day_zero["player_id"].to_numpy()
    # Compute decile edges (0..10 boundaries)
    deciles = np.percentile(skills, np.arange(0, 101, 10))
    # Assign each day-0 player to a decile 0..9
    decile_idx = np.clip(
        np.searchsorted(deciles[1:-1], skills, side="right"), 0, 9
    )

    # Map player_id -> decile
    id_to_decile = dict(zip(ids.tolist(), decile_idx.tolist()))

    # Only look at players who were present on day 0
    cohort = population.filter(pl.col("player_id").is_in(ids.tolist()))
    cohort_np = cohort.select(["day", "player_id", "active"]).to_numpy()

    days_sorted = sorted(set(int(r[0]) for r in cohort_np))
    retention_per_decile: dict[int, list[float]] = {d: [] for d in range(10)}
    decile_counts = np.bincount(decile_idx, minlength=10)

    # Build per-day active counts per decile
    for day in days_sorted:
        active_by_decile = np.zeros(10, dtype=int)
        mask = cohort_np[:, 0] == day
        day_rows = cohort_np[mask]
        for pid, active in zip(day_rows[:, 1], day_rows[:, 2]):
            if active:
                d = id_to_decile[int(pid)]
                active_by_decile[d] += 1
        for d in range(10):
            total = decile_counts[d]
            retention_per_decile[d].append(
                active_by_decile[d] / total if total > 0 else np.nan
            )

    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 10))
    for d in range(10):
        ax.plot(
            days_sorted,
            retention_per_decile[d],
            marker=".",
            linewidth=1.5,
            color=colors[d],
            label=f"decile {d + 1}",
        )
    ax.set_xlabel("day")
    ax.set_ylabel("retention")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2, title="day-0 skill decile")


def generate_plots_for_experiment_dir(exp_dir: Path) -> list[Path]:
    """Regenerate plots for an already-saved experiment directory."""
    aggregate = pl.read_parquet(exp_dir / "aggregate.parquet")
    pop_path = exp_dir / "population.parquet"
    if not pop_path.exists():
        raise FileNotFoundError(
            f"population.parquet not found in {exp_dir}; "
            "can't generate plots without per-day population snapshots"
        )
    population = pl.read_parquet(pop_path)
    return generate_plots(
        population=population,
        aggregate=aggregate,
        out_dir=exp_dir / "plots",
        experiment_name=exp_dir.name,
    )
