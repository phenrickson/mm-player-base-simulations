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
from mm_sim.scenarios import (
    DEFAULT_SCENARIOS_DIR,
    load_scenarios_dir,
    load_season_name,
)

log = logging.getLogger(__name__)


# Each category gets a distinct matplotlib colormap. Variants within a
# category are spaced along the colormap so they read as "same family,
# different member." `other` stays neutral-grey so uncategorized
# scenarios don't steal hues from real categories.
_CATEGORY_COLORMAPS = {
    "matchmaker": "Blues",
    "ablation": "Oranges",
    "sweep_point": "Greens",
    "base": "Purples",
    "other": "Greys",
}


def _assign_category_colors(
    experiments: list,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
) -> tuple[list, list[tuple]]:
    """Order experiments by category, then return (ordered_experiments, colors).

    Each experiment's color is drawn from its category's colormap, spaced
    within the category so members are distinguishable but share a hue
    family. Experiments whose scenario can't be found (e.g. stale runs
    whose TOML was deleted) fall back to the `other` category.
    """
    scenarios = load_scenarios_dir(scenarios_dir)
    by_category: dict[str, list] = {}
    for exp in experiments:
        scenario = scenarios.get(exp.metadata.name)
        category = scenario.category if scenario is not None else "other"
        by_category.setdefault(category, []).append(exp)

    # Stable category order: known categories first (matchmaker, ablation,
    # sweep_point, base), then any novel ones alphabetically, then "other".
    known_order = ["matchmaker", "ablation", "sweep_point", "base"]
    novel = sorted(c for c in by_category if c not in known_order and c != "other")
    ordered_categories = [c for c in known_order if c in by_category] + novel
    if "other" in by_category:
        ordered_categories.append("other")

    ordered_exps: list = []
    colors: list[tuple] = []
    for category in ordered_categories:
        members = by_category[category]
        cmap = plt.get_cmap(_CATEGORY_COLORMAPS.get(category, "Greys"))
        # Sample away from the extremes (too light to read, too dark to tell apart).
        n = len(members)
        positions = (
            np.linspace(0.45, 0.9, n) if n > 1 else np.array([0.7])
        )
        for exp, pos in zip(members, positions):
            ordered_exps.append(exp)
            colors.append(cmap(pos))
    return ordered_exps, colors


def _is_sweep_dir(p: Path) -> bool:
    """A sweep stores a sweep.json inside each version dir, and no top-level
    metadata.json (which a scenario run would have)."""
    return any(v.is_dir() and (v / "sweep.json").exists() for v in p.iterdir())


def compare_scenarios(
    names: list[str] | None = None,
    season: str | None = None,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> list[Path]:
    """Generate comparison plots across scenarios in a season.

    If `season` is None, uses the current season from `defaults.toml`.
    If `names` is None, compares every scenario with at least one saved
    run in that season (skipping underscore-prefixed dirs like the
    `_comparisons` output).
    """
    if season is None:
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
            if p.is_dir()
            and not p.name.startswith("_")
            and not _is_sweep_dir(p)
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

    # Reorder by category and assign category-keyed colors so matchmaker
    # variants, ablation pairs, and sweep points read as distinct families
    # in every comparison plot.
    experiments, colors = _assign_category_colors(
        experiments, scenarios_dir=scenarios_dir
    )

    return _render_comparison_panels(
        experiments, colors, out_dir=out_dir, title_suffix=season
    )


def _render_comparison_panels(
    experiments: list,
    colors,
    out_dir: Path,
    title_suffix: str,
) -> list[Path]:
    """Render the full suite of comparison panels from pre-ordered
    (experiments, colors). Returns the list of written paths.
    """
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
        (
            "lobby_range.png",
            "Lobby true-skill range (max − min)",
            _make_band_plot("lobby_range"),
        ),
        (
            "lobby_std.png",
            "Lobby true-skill std",
            _make_band_plot("lobby_std"),
        ),
        (
            "team_gap.png",
            "Team mean true-skill gap",
            _make_band_plot("team_gap"),
        ),
        (
            "favorite_win_prob.png",
            "Favorite's expected win probability",
            _plot_win_prob_dev_comparison,
        ),
    ]

    for filename, title, plot_fn in panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_fn(ax, experiments, colors)
        fig.suptitle(f"{title} — {title_suffix}")
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
    fig.suptitle(f"Scenario comparison — {title_suffix}", fontsize=14)
    fig.tight_layout()
    overview_path = out_dir / "overview.png"
    fig.savefig(overview_path, dpi=120)
    plt.close(fig)
    written.append(overview_path)

    # Match-quality 2x2 (all 4 per-match metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _make_band_plot("lobby_range")(axes[0, 0], experiments, colors)
    _make_band_plot("lobby_std")(axes[0, 1], experiments, colors)
    _make_band_plot("team_gap")(axes[1, 0], experiments, colors)
    _plot_win_prob_dev_comparison(axes[1, 1], experiments, colors)
    axes[0, 0].set_title("Lobby range (max − min true_skill)")
    axes[0, 1].set_title("Lobby std (true_skill)")
    axes[1, 0].set_title("Team mean true_skill gap")
    axes[1, 1].set_title("Favorite's expected win probability")
    fig.suptitle(f"Match quality — {title_suffix}", fontsize=14)
    fig.tight_layout()
    mq_path = out_dir / "match_quality.png"
    fig.savefig(mq_path, dpi=120)
    plt.close(fig)
    written.append(mq_path)

    # Churn rate by cohort (1x3: new / casual / experienced)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    _plot_churn_rate_for_cohort(axes[0], experiments, colors, lo=0, hi=20)
    _plot_churn_rate_for_cohort(axes[1], experiments, colors, lo=20, hi=50)
    _plot_churn_rate_for_cohort(axes[2], experiments, colors, lo=50, hi=None)
    axes[0].set_title("new (<20 matches)")
    axes[1].set_title("casual (20–49 matches)")
    axes[2].set_title("experienced (≥50 matches)")
    fig.suptitle(f"Daily churn rate by cohort — {title_suffix}", fontsize=14)
    fig.tight_layout()
    churn_path = out_dir / "churn_by_cohort.png"
    fig.savefig(churn_path, dpi=120)
    plt.close(fig)
    written.append(churn_path)

    log.info("wrote %d comparison plot(s) to %s", len(written), out_dir)
    return written


def compare_sweep_with_references(
    sweep_name: str,
    reference_scenarios: list[str] | None = None,
    season: str | None = None,
    sweep_version: str | None = None,
    scenarios_dir: Path | str = DEFAULT_SCENARIOS_DIR,
    experiments_dir: Path | str = DEFAULT_EXPERIMENTS_DIR,
) -> list[Path]:
    """Overlay a parameter sweep's points with named reference scenarios.

    Sweep points are colored as a gradient along the sweep axis (light →
    dark as the first-axis value grows) so the parameter progression is
    visually obvious. References are drawn in a distinct hue family so
    they don't blend into the gradient.

    Writes plots into the sweep's own `_comparisons/` directory.
    """
    import json

    if season is None:
        season = load_season_name(scenarios_dir)
    base = Path(experiments_dir)
    sweep_parent = base / season / sweep_name
    if not sweep_parent.exists():
        raise FileNotFoundError(f"sweep not found: {sweep_parent}")
    if sweep_version is None:
        versions = sorted(
            p for p in sweep_parent.iterdir() if p.name.startswith("v")
        )
        if not versions:
            raise FileNotFoundError(f"no versions under {sweep_parent}")
        sweep_dir = versions[-1]
    else:
        sweep_dir = sweep_parent / sweep_version
    metadata = json.loads((sweep_dir / "sweep.json").read_text())
    points_dir = sweep_dir / "points"

    # Load each sweep point as an Experiment, remembering its first-axis value.
    sweep_experiments: list = []
    sweep_values: list[float] = []
    first_param = metadata["parameters"][0]
    for point in metadata["points"]:
        exp = load_experiment(
            point["experiment_name"],
            version=point["experiment_version"],
            season=None,
            experiments_dir=points_dir,
        )
        sweep_experiments.append(exp)
        sweep_values.append(float(point["overrides"][first_param]))

    # Color sweep points along a gradient by their parameter value.
    cmap = plt.get_cmap("viridis")
    v_min = min(sweep_values)
    v_max = max(sweep_values)
    span = (v_max - v_min) or 1.0
    sweep_colors = [
        cmap(0.15 + 0.75 * (v - v_min) / span) for v in sweep_values
    ]

    # Load references and assign them a contrasting palette (reds).
    reference_experiments: list = []
    reference_colors: list = []
    if reference_scenarios:
        ref_cmap = plt.get_cmap("Reds")
        ref_positions = (
            np.linspace(0.55, 0.85, len(reference_scenarios))
            if len(reference_scenarios) > 1 else [0.7]
        )
        for name, pos in zip(reference_scenarios, ref_positions):
            reference_experiments.append(
                load_experiment(
                    name, season=season, experiments_dir=experiments_dir
                )
            )
            reference_colors.append(ref_cmap(pos))

    # Stable display order: sweep first (low→high), then references.
    order = sorted(range(len(sweep_experiments)), key=lambda i: sweep_values[i])
    experiments = [sweep_experiments[i] for i in order] + reference_experiments
    colors = [sweep_colors[i] for i in order] + reference_colors

    out_dir = sweep_dir / "_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f"{sweep_name} — {season}"
    return _render_comparison_panels(
        experiments, colors, out_dir=out_dir, title_suffix=title_suffix
    )


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
    """Two lines per scenario: solid = all active players (includes new
    arrivals and excludes churned); dashed = day-0 cohort only (fixed
    set of players, shows Elo convergence independent of churn).

    Divergence between the two exposes survivorship / new-arrival
    effects: if the solid line looks better than the dashed one, the
    scenario is benefiting from churning out high-error players or
    has too few high-error arrivals to matter.
    """
    for color, exp in zip(colors, experiments):
        agg = exp.aggregate
        if "rating_error_mean" in agg.columns:
            ax.plot(
                agg["day"].to_numpy(),
                agg["rating_error_mean"].to_numpy(),
                linewidth=2,
                color=color,
                label=exp.metadata.name,
            )
        pop = exp.population
        if pop is None:
            continue
        day_zero_ids = pop.filter(pl.col("day") == 0)["player_id"].to_list()
        if not day_zero_ids:
            continue
        cohort = (
            pop.filter(
                pl.col("player_id").is_in(day_zero_ids) & pl.col("active")
            )
            .with_columns(
                (pl.col("observed_skill") - pl.col("true_skill")).abs().alias("err")
            )
            .group_by("day")
            .agg(pl.col("err").mean().alias("err"))
            .sort("day")
        )
        ax.plot(
            cohort["day"].to_numpy(),
            cohort["err"].to_numpy(),
            linewidth=1.5,
            linestyle="--",
            color=color,
        )

    ax.set_xlabel("day")
    ax.set_ylabel("mean |observed − true| skill")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    # Note in legend that dashed = day-0 cohort subset
    ax.legend(
        fontsize=8,
        loc="upper right",
        title="— all active, -- day-0 cohort",
        title_fontsize=7,
    )


def _plot_blowout_share(ax, experiments: list, colors) -> None:
    peak = 0.0
    for color, exp in zip(colors, experiments):
        agg = exp.aggregate
        days = agg["day"].to_numpy()
        matches = agg["matches_played"].to_numpy().astype(float)
        blowouts = agg["blowouts"].to_numpy().astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(matches > 0, blowouts / matches, np.nan)
        if np.any(~np.isnan(share)):
            peak = max(peak, float(np.nanmax(share)))
        ax.plot(
            days,
            share,
            linewidth=2,
            color=color,
            label=exp.metadata.name,
        )
    ax.set_xlabel("day")
    ax.set_ylabel("blowout share of matches")
    ax.set_ylim(0, max(0.05, peak * 1.2))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _make_band_plot(metric_prefix: str):
    """Build an axis-plotter closure for a match-quality metric.

    Draws one solid mean line per scenario, plus a faint shaded band
    between p50 and p90. Reads `<prefix>_mean/_p50/_p90` columns from
    each experiment's aggregate dataframe.
    """
    mean_col = f"{metric_prefix}_mean"
    p50_col = f"{metric_prefix}_p50"
    p90_col = f"{metric_prefix}_p90"

    def _plot(ax, experiments: list, colors) -> None:
        for color, exp in zip(colors, experiments):
            agg = exp.aggregate
            if mean_col not in agg.columns:
                continue
            days = agg["day"].to_numpy()
            mean = agg[mean_col].to_numpy()
            if p50_col in agg.columns and p90_col in agg.columns:
                p50 = agg[p50_col].to_numpy()
                p90 = agg[p90_col].to_numpy()
                ax.fill_between(days, p50, p90, alpha=0.15, color=color)
            ax.plot(
                days,
                mean,
                linewidth=2,
                color=color,
                label=exp.metadata.name,
            )
        ax.set_xlabel("day")
        ax.set_ylabel(metric_prefix)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    return _plot


def _plot_win_prob_dev_comparison(ax, experiments: list, colors) -> None:
    """Favorite's expected win-or-extract probability.

    In 2-team mode this is 0.5 + |P(team_a wins) - 0.5| from true_skill Elo.
    In extraction mode it's the strongest team's `expected_extract`.
    One axis: higher = more lopsided match.
    """
    mode = "extract"
    for exp in experiments:
        agg = exp.aggregate
        if (
            "favorite_expected_extract_mean" in agg.columns
            and not agg["favorite_expected_extract_mean"].is_null().all()
        ):
            continue
        mode = "win_prob_dev"
        break

    for color, exp in zip(colors, experiments):
        agg = exp.aggregate
        days = agg["day"].to_numpy()
        if mode == "extract":
            if "favorite_expected_extract_mean" not in agg.columns:
                continue
            mean = agg["favorite_expected_extract_mean"].to_numpy()
            p50_col, p90_col = (
                "favorite_expected_extract_p50",
                "favorite_expected_extract_p90",
            )
        else:
            if "win_prob_dev_mean" not in agg.columns:
                continue
            mean = 0.5 + agg["win_prob_dev_mean"].to_numpy()
            p50_col, p90_col = "win_prob_dev_p50", "win_prob_dev_p90"
        if p50_col in agg.columns and p90_col in agg.columns:
            p50 = agg[p50_col].to_numpy()
            p90 = agg[p90_col].to_numpy()
            if mode == "win_prob_dev":
                p50 = 0.5 + p50
                p90 = 0.5 + p90
            ax.fill_between(days, p50, p90, alpha=0.15, color=color)
        ax.plot(days, mean, linewidth=2, color=color, label=exp.metadata.name)

    if mode == "extract":
        ax.set_ylabel("favorite's expected extract probability")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.axhline(
            0.5, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label="coin flip"
        )
        ax.set_ylabel("favorite's expected win probability")
        ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("day")
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
