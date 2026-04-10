"""Cross-scenario comparison plots within a single season.

Loads the latest version of each named scenario from the current season
and overlays their metrics on shared axes. Writes one PNG per metric
plus a 2x2 overview figure.
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

    If `names` is None, compares every scenario with at least one saved
    run in the current season (skipping underscore-prefixed dirs like
    the `_comparisons` output).
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

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(experiments), 1)))

    written: list[Path] = []

    panels = [
        ("retention.png", "Day-0 cohort retention", _plot_retention),
        ("active_population.png", "Active player population", _plot_active_population),
        ("rating_error.png", "Rating error (|observed − true|)", _plot_rating_error),
        ("blowout_share.png", "Blowout share of matches", _plot_blowout_share),
        (
            "final_skill_distribution.png",
            "Final-day true-skill distribution (active players)",
            _plot_final_skill_distribution,
        ),
    ]

    for filename, title, plot_fn in panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_fn(ax, experiments, colors)
        fig.suptitle(f"{title} — {season}")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)

    # 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_retention(axes[0, 0], experiments, colors)
    _plot_active_population(axes[0, 1], experiments, colors)
    _plot_rating_error(axes[1, 0], experiments, colors)
    _plot_blowout_share(axes[1, 1], experiments, colors)
    axes[0, 0].set_title("Day-0 cohort retention")
    axes[0, 1].set_title("Active player population")
    axes[1, 0].set_title("Rating error (|observed − true|)")
    axes[1, 1].set_title("Blowout share of matches")
    fig.suptitle(f"Scenario comparison — {season}", fontsize=14)
    fig.tight_layout()
    overview_path = out_dir / "overview.png"
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    written.append(overview_path)

    # Churn rate by cohort (1x3: new / casual / experienced)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    _plot_churn_rate_for_cohort(axes[0], experiments, colors, lo=0, hi=20)
    _plot_churn_rate_for_cohort(axes[1], experiments, colors, lo=20, hi=50)
    _plot_churn_rate_for_cohort(axes[2], experiments, colors, lo=50, hi=None)
    axes[0].set_title("new (<20 matches)")
    axes[1].set_title("casual (20–49 matches)")
    axes[2].set_title("experienced (≥50 matches)")
    fig.suptitle(f"Daily churn rate by cohort — {season}", fontsize=14)
    fig.tight_layout()
    churn_path = out_dir / "churn_by_cohort.png"
    fig.savefig(churn_path, dpi=120)
    plt.close(fig)
    written.append(churn_path)

    log.info("wrote %d comparison plot(s) to %s", len(written), out_dir)
    return written


# ---- axis-based plot helpers --------------------------------------------


def _plot_retention(ax, experiments: list, colors) -> None:
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
    ax.legend(fontsize=8, loc="lower left")


def _plot_active_population(ax, experiments: list, colors) -> None:
    for color, exp in zip(colors, experiments):
        agg = exp.aggregate
        ax.plot(
            agg["day"].to_numpy(),
            agg["active_count"].to_numpy(),
            linewidth=2,
            color=color,
            label=exp.metadata.name,
        )
    ax.set_xlabel("day")
    ax.set_ylabel("active players")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")


def _plot_rating_error(ax, experiments: list, colors) -> None:
    for color, exp in zip(colors, experiments):
        agg = exp.aggregate
        if "rating_error_mean" not in agg.columns:
            continue
        ax.plot(
            agg["day"].to_numpy(),
            agg["rating_error_mean"].to_numpy(),
            linewidth=2,
            color=color,
            label=exp.metadata.name,
        )
    ax.set_xlabel("day")
    ax.set_ylabel("mean |observed − true| skill")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _plot_blowout_share(ax, experiments: list, colors) -> None:
    for color, exp in zip(colors, experiments):
        agg = exp.aggregate
        days = agg["day"].to_numpy()
        matches = agg["matches_played"].to_numpy().astype(float)
        blowouts = agg["blowouts"].to_numpy().astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(matches > 0, blowouts / matches, np.nan)
        ax.plot(
            days,
            share,
            linewidth=2,
            color=color,
            label=exp.metadata.name,
        )
    ax.set_xlabel("day")
    ax.set_ylabel("blowout share of matches")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _plot_final_skill_distribution(ax, experiments: list, colors) -> None:
    """Overlaid histograms of final-day active-player true_skill.

    Normalized to density so different active counts don't distort
    the shape comparison.
    """
    # Shared bins across all scenarios for a fair overlay.
    all_skills: list[np.ndarray] = []
    per_scenario: list[tuple] = []
    for color, exp in zip(colors, experiments):
        pop = exp.population
        if pop is None:
            continue
        last_day = int(pop["day"].max())
        final = pop.filter((pl.col("day") == last_day) & pl.col("active"))
        if final.height == 0:
            continue
        skills = final["true_skill"].to_numpy()
        all_skills.append(skills)
        per_scenario.append((color, exp.metadata.name, skills))

    if not per_scenario:
        ax.text(0.5, 0.5, "no active players", ha="center", va="center")
        return

    combined = np.concatenate(all_skills)
    lo = float(combined.min()) - 0.1
    hi = float(combined.max()) + 0.1
    bins = np.linspace(lo, hi, 40)

    for color, name, skills in per_scenario:
        ax.hist(
            skills,
            bins=bins,
            density=True,
            alpha=0.4,
            color=color,
            label=f"{name} (n={skills.size})",
        )

    ax.set_xlabel("true_skill")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _plot_churn_rate_for_cohort(
    ax,
    experiments: list,
    colors,
    lo: int,
    hi: int | None,
) -> None:
    """Daily churn rate restricted to a matches_played cohort, one line
    per scenario.

    `lo` is inclusive, `hi` is exclusive. Pass `hi=None` for open-ended
    top tier.
    """
    for color, exp in zip(colors, experiments):
        pop = exp.population
        if pop is None:
            continue
        df = pop.select(["day", "player_id", "matches_played", "active"]).sort(
            ["player_id", "day"]
        )
        df = df.with_columns(
            pl.col("active").shift(1).over("player_id").alias("prev_active"),
        )
        df = df.with_columns(
            (pl.col("prev_active") & ~pl.col("active")).fill_null(False).alias("quit")
        )
        in_cohort = pl.col("matches_played") >= lo
        if hi is not None:
            in_cohort = in_cohort & (pl.col("matches_played") < hi)
        grouped = (
            df.filter(pl.col("prev_active") & in_cohort)
            .group_by("day")
            .agg(
                pl.col("quit").sum().alias("quits"),
                pl.len().alias("cohort"),
            )
            .sort("day")
        )
        if grouped.height == 0:
            continue
        days = grouped["day"].to_numpy()
        rate = (
            grouped["quits"].cast(pl.Float64) / grouped["cohort"].cast(pl.Float64)
        ).to_numpy()
        ax.plot(days, rate, linewidth=2, color=color, label=exp.metadata.name)

    ax.set_xlabel("day")
    ax.set_ylabel("daily churn rate")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
