"""Per-experiment visualizations.

Generates PNG plots from an Experiment's population and aggregate data
and writes them to `<experiment_dir>/plots/`. Called automatically by
`ExperimentRunner.run()` and also available via `just plots NAME`.

Plots are grouped into three figures:
- overview.png    — the 4 most important views at a glance
- churn.png       — detail view for "who leaves and when"
- matches.png     — detail view for "what matches look like"
- population.png  — detail view for "how the population changes shape"

Each sub-plot is also written as its own PNG so they can be embedded
individually.

Color conventions:
- Orange→blue diverging for anything keyed on skill (low = orange, high = blue)
- Viridis for non-valenced categories (checkpoint days, buckets, samples)
- Fixed pairs for binary splits (stayed/left, new/veteran)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")  # headless backend — no Tk needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


# ---- color palette --------------------------------------------------------

ORANGE_BLUE = LinearSegmentedColormap.from_list(
    "orange_blue", ["#d95f02", "#f0f0f0", "#1f78b4"]
)
# Fixed 2-color pairs
COLOR_STAYED = "#1f78b4"   # blue
COLOR_LEFT = "#d95f02"     # orange
COLOR_NEW = "#d95f02"      # orange (new players)
COLOR_VETERAN = "#1f78b4"  # blue (veterans)

PLOT_FILENAMES = {
    # Existing
    "skill_dist": "skill_distribution_intervals.png",
    "population": "population_over_time.png",
    "retention": "retention_over_time.png",
    "retention_by_decile": "retention_by_skill_decile.png",
    # New — individual panels
    "churn_rate": "churn_rate_over_time.png",
    "retention_by_early_winrate": "retention_by_early_win_rate.png",
    "rating_error": "rating_error_over_time.png",
    "match_quality": "match_quality_over_time.png",
    "who_left": "who_left_vs_stayed.png",
    "skill_percentiles": "true_skill_percentiles_over_time.png",
    "player_trajectories": "cumulative_score_by_skill.png",
    # Grouped figures
    "overview": "overview.png",
    "churn_detail": "churn_detail.png",
    "matches_detail": "matches_detail.png",
    "population_detail": "population_detail.png",
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

    # ---- individual panels ------------------------------------------------

    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["skill_dist"],
            _plot_skill_distribution_intervals,
            (population, checkpoint_days),
            _title("Skill distribution across the season", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["population"],
            _plot_population_over_time,
            (aggregate,),
            _title("Active player population over time", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["retention"],
            _plot_retention_over_time,
            (population,),
            _title("Day-0 cohort retention over time", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["retention_by_decile"],
            _plot_retention_by_skill_decile,
            (population,),
            _title("Retention by day-0 true-skill decile", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["churn_rate"],
            _plot_churn_rate_over_time,
            (population,),
            _title("Daily churn rate — new vs veteran", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["retention_by_early_winrate"],
            _plot_retention_by_early_win_rate,
            (population,),
            _title("Retention by day 1–7 win rate", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["rating_error"],
            _plot_rating_error_over_time,
            (aggregate,),
            _title("Mean |observed − true| skill error over time", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["match_quality"],
            _plot_match_quality_over_time,
            (aggregate,),
            _title("Blowout share of daily matches", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["who_left"],
            _plot_who_left_vs_stayed,
            (population,),
            _title("True-skill of churned vs still-active players", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["skill_percentiles"],
            _plot_true_skill_percentiles,
            (aggregate,),
            _title("True-skill percentiles over time (active players)", experiment_name),
        )
    )
    written.append(
        _save_single(
            out_dir / PLOT_FILENAMES["player_trajectories"],
            _plot_cumulative_score_by_skill,
            (population,),
            _title("Cumulative (wins − losses) by day-0 true skill", experiment_name),
        )
    )

    # ---- grouped figures --------------------------------------------------

    # Overview (high-signal 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_population_over_time(axes[0, 0], aggregate)
    _plot_churn_rate_over_time(axes[0, 1], population)
    _plot_true_skill_percentiles(axes[1, 0], aggregate)
    _plot_match_quality_over_time(axes[1, 1], aggregate)
    axes[0, 0].set_title("Active population")
    axes[0, 1].set_title("Daily churn rate (new vs veteran)")
    axes[1, 0].set_title("True-skill percentiles")
    axes[1, 1].set_title("Blowout share")
    fig.suptitle(_title("Overview", experiment_name), fontsize=14)
    path = out_dir / PLOT_FILENAMES["overview"]
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    # Churn detail
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_retention_over_time(axes[0, 0], population)
    _plot_churn_rate_over_time(axes[0, 1], population)
    _plot_retention_by_early_win_rate(axes[1, 0], population)
    _plot_retention_by_skill_decile(axes[1, 1], population)
    axes[0, 0].set_title("Day-0 cohort retention")
    axes[0, 1].set_title("Daily churn rate (new vs veteran)")
    axes[1, 0].set_title("Retention by day 1–7 win rate")
    axes[1, 1].set_title("Retention by day-0 skill decile")
    fig.suptitle(_title("Churn detail", experiment_name), fontsize=14)
    path = out_dir / PLOT_FILENAMES["churn_detail"]
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    # Matches detail
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_match_quality_over_time(axes[0, 0], aggregate)
    _plot_rating_error_over_time(axes[0, 1], aggregate)
    _plot_cumulative_score_by_skill(axes[1, 0], population)
    _plot_skill_distribution_intervals(axes[1, 1], population, checkpoint_days)
    axes[0, 0].set_title("Blowout share")
    axes[0, 1].set_title("Rating error (|observed − true|)")
    axes[1, 0].set_title("Cumulative (wins − losses) by day-0 true skill")
    axes[1, 1].set_title("Observed-skill distribution at checkpoints")
    fig.suptitle(_title("Matches detail", experiment_name), fontsize=14)
    path = out_dir / PLOT_FILENAMES["matches_detail"]
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    # Population detail
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_true_skill_percentiles(axes[0, 0], aggregate)
    _plot_who_left_vs_stayed(axes[0, 1], population)
    _plot_population_over_time(axes[1, 0], aggregate)
    _plot_retention_by_skill_decile(axes[1, 1], population)
    axes[0, 0].set_title("True-skill percentiles")
    axes[0, 1].set_title("Who left vs who stayed")
    axes[1, 0].set_title("Active population")
    axes[1, 1].set_title("Retention by day-0 skill decile")
    fig.suptitle(_title("Population detail", experiment_name), fontsize=14)
    path = out_dir / PLOT_FILENAMES["population_detail"]
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path)

    return written


# ---- helpers --------------------------------------------------------------


def _save_single(path: Path, plot_fn, args: tuple, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fn(ax, *args)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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


# ---- individual plot functions --------------------------------------------


def _plot_skill_distribution_intervals(
    ax, population: pl.DataFrame, checkpoint_days: list[int]
) -> None:
    """Histograms of active-player observed_skill at each checkpoint day."""
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, max(len(checkpoint_days), 1)))
    active_all = population.filter(pl.col("active"))
    if active_all.height == 0:
        ax.text(0.5, 0.5, "no active players", ha="center", va="center")
        return

    obs_all = active_all["observed_skill"].to_numpy()
    lo = float(obs_all.min()) - 0.1
    hi = float(obs_all.max()) + 0.1
    bins = np.linspace(lo, hi, 40)

    for color, day in zip(colors, checkpoint_days):
        day_slice = population.filter(
            (pl.col("day") == day) & pl.col("active")
        )["observed_skill"].to_numpy()
        if day_slice.size == 0:
            continue
        ax.hist(
            day_slice,
            bins=bins,
            density=True,
            alpha=0.4,
            color=color,
            label=f"day {day} (n={day_slice.size})",
        )
    ax.set_xlabel("observed_skill")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_population_over_time(ax, aggregate: pl.DataFrame) -> None:
    days = aggregate["day"].to_numpy()
    active = aggregate["active_count"].to_numpy()
    ax.plot(days, active, linewidth=2, color=COLOR_STAYED)
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
        linewidth=2,
        color=COLOR_STAYED,
    )
    ax.set_xlabel("day")
    ax.set_ylabel("fraction of day-0 cohort still active")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)


def _plot_retention_by_skill_decile(ax, population: pl.DataFrame) -> None:
    """Retention curves broken down by day-0 true_skill decile.

    Uses the orange→blue diverging palette so low-skill deciles read
    orange and high-skill deciles read blue.
    """
    day_zero = population.filter(pl.col("day") == 0)
    if day_zero.height == 0:
        ax.text(0.5, 0.5, "no day-0 players", ha="center", va="center")
        return

    skills = day_zero["true_skill"].to_numpy()
    ids = day_zero["player_id"].to_numpy()
    deciles = np.percentile(skills, np.arange(0, 101, 10))
    decile_idx = np.clip(
        np.searchsorted(deciles[1:-1], skills, side="right"), 0, 9
    )
    id_to_decile = dict(zip(ids.tolist(), decile_idx.tolist()))

    cohort = population.filter(pl.col("player_id").is_in(ids.tolist()))
    cohort_np = cohort.select(["day", "player_id", "active"]).to_numpy()

    days_sorted = sorted(set(int(r[0]) for r in cohort_np))
    retention_per_decile: dict[int, list[float]] = {d: [] for d in range(10)}
    decile_counts = np.bincount(decile_idx, minlength=10)

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

    decile_labels = [
        "bottom 10%",
        "10–20%",
        "20–30%",
        "30–40%",
        "40–50%",
        "50–60%",
        "60–70%",
        "70–80%",
        "80–90%",
        "top 10%",
    ]
    colors = ORANGE_BLUE(np.linspace(0.05, 0.95, 10))
    for d in range(10):
        ax.plot(
            days_sorted,
            retention_per_decile[d],
            linewidth=1.5,
            color=colors[d],
            label=decile_labels[d],
        )
    ax.set_xlabel("day")
    ax.set_ylabel("retention")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=2, title="day-0 true skill")


def _plot_churn_rate_over_time(ax, population: pl.DataFrame) -> None:
    """Daily churn rate, split into new-player vs veteran.

    A 'new player' is one whose matches_played that day was below a
    threshold (here: 20). Churn rate = (players who went active→inactive
    between day d and day d+1) / (active players in that segment on day d).
    """
    threshold = 20
    df = population.select(
        ["day", "player_id", "matches_played", "active"]
    ).sort(["player_id", "day"])

    # Per-player previous-day active state so we can detect active→inactive transitions.
    df = df.with_columns(
        pl.col("active").shift(1).over("player_id").alias("prev_active"),
    )
    # A "quit event" this day: was active yesterday, not active today.
    df = df.with_columns(
        (pl.col("prev_active") & ~pl.col("active")).fill_null(False).alias("quit"),
        (pl.col("matches_played") < threshold).alias("is_new"),
    )

    # For each day, churn rate in new vs veteran sub-populations.
    # Denominator: players who were active yesterday (prev_active), split by
    # today's is_new flag.
    grouped = (
        df.filter(pl.col("prev_active"))
        .group_by(["day", "is_new"])
        .agg(
            pl.col("quit").sum().alias("quits"),
            pl.len().alias("cohort"),
        )
        .sort(["day", "is_new"])
    )
    if grouped.height == 0:
        ax.text(0.5, 0.5, "no churn data", ha="center", va="center")
        return

    for is_new, color, label in [
        (True, COLOR_NEW, f"new (<{threshold} matches)"),
        (False, COLOR_VETERAN, f"veteran (≥{threshold} matches)"),
    ]:
        sub = grouped.filter(pl.col("is_new") == is_new).sort("day")
        if sub.height == 0:
            continue
        days = sub["day"].to_numpy()
        rate = (
            sub["quits"].cast(pl.Float64) / sub["cohort"].cast(pl.Float64)
        ).to_numpy()
        ax.plot(days, rate, linewidth=2, color=color, label=label)

    ax.set_xlabel("day")
    ax.set_ylabel("daily churn rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)


def _plot_retention_by_early_win_rate(ax, population: pl.DataFrame) -> None:
    """Retention curves for day-0 players bucketed by their win rate over
    days 1–7.

    Excludes players who churned during the window (partial data).
    Buckets are quartiles of the surviving cohort's win rate. Shows
    whether early match outcomes predict long-term retention.
    """
    if "recent_wins" not in population.columns or "recent_losses" not in population.columns:
        ax.text(0.5, 0.5, "recent_wins/losses not recorded", ha="center", va="center")
        return

    day_zero_ids = population.filter(pl.col("day") == 0)["player_id"].to_numpy()
    if day_zero_ids.size == 0:
        ax.text(0.5, 0.5, "no day-0 players", ha="center", va="center")
        return

    window_days = 7
    # Survivors: still active at end of window
    window_end = population.filter(
        (pl.col("day") == window_days) & pl.col("active")
    )
    survivor_ids = set(window_end["player_id"].to_list()) & set(
        day_zero_ids.tolist()
    )
    if not survivor_ids:
        ax.text(0.5, 0.5, "no survivors through window", ha="center", va="center")
        return

    # Sum wins and losses per player across days 1..window_days
    win_stats = (
        population.filter(
            (pl.col("day") >= 1)
            & (pl.col("day") <= window_days)
            & pl.col("player_id").is_in(list(survivor_ids))
        )
        .group_by("player_id")
        .agg(
            pl.col("recent_wins").sum().alias("wins"),
            pl.col("recent_losses").sum().alias("losses"),
        )
    )
    win_stats = win_stats.with_columns(
        (pl.col("wins") + pl.col("losses")).alias("matches")
    ).filter(pl.col("matches") > 0)
    win_stats = win_stats.with_columns(
        (pl.col("wins").cast(pl.Float64) / pl.col("matches").cast(pl.Float64)).alias(
            "win_rate"
        )
    )
    if win_stats.height == 0:
        ax.text(0.5, 0.5, "no match data in window", ha="center", va="center")
        return

    win_rates = win_stats["win_rate"].to_numpy()
    ids = win_stats["player_id"].to_numpy()
    # Quartile edges
    edges = np.percentile(win_rates, [0, 25, 50, 75, 100])
    bucket = np.clip(np.searchsorted(edges[1:-1], win_rates, side="right"), 0, 3)
    id_to_bucket = {int(pid): int(b) for pid, b in zip(ids, bucket)}

    n_buckets = 4
    bucket_counts = np.bincount(bucket, minlength=n_buckets)
    labels = [
        f"Q1 ({edges[0]:.2f}–{edges[1]:.2f})",
        f"Q2 ({edges[1]:.2f}–{edges[2]:.2f})",
        f"Q3 ({edges[2]:.2f}–{edges[3]:.2f})",
        f"Q4 ({edges[3]:.2f}–{edges[4]:.2f})",
    ]

    cohort = population.filter(pl.col("player_id").is_in(list(id_to_bucket.keys())))
    cohort_np = cohort.select(["day", "player_id", "active"]).to_numpy()
    days_sorted = sorted(set(int(r[0]) for r in cohort_np))

    retention_per_bucket: dict[int, list[float]] = {b: [] for b in range(n_buckets)}
    for day in days_sorted:
        active_by_bucket = np.zeros(n_buckets, dtype=int)
        mask = cohort_np[:, 0] == day
        day_rows = cohort_np[mask]
        for pid, active in zip(day_rows[:, 1], day_rows[:, 2]):
            if active:
                b = id_to_bucket[int(pid)]
                active_by_bucket[b] += 1
        for b in range(n_buckets):
            total = bucket_counts[b]
            retention_per_bucket[b].append(
                active_by_bucket[b] / total if total > 0 else np.nan
            )

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_buckets))
    for b in range(n_buckets):
        ax.plot(
            days_sorted,
            retention_per_bucket[b],
            linewidth=1.5,
            color=colors[b],
            label=f"{labels[b]} (n={bucket_counts[b]})",
        )
    ax.set_xlabel("day")
    ax.set_ylabel("retention")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", title="day 1–7 win rate")


def _plot_rating_error_over_time(ax, aggregate: pl.DataFrame) -> None:
    if "rating_error_mean" not in aggregate.columns:
        ax.text(0.5, 0.5, "rating_error_mean not recorded", ha="center", va="center")
        return
    days = aggregate["day"].to_numpy()
    err = aggregate["rating_error_mean"].to_numpy()
    ax.plot(days, err, linewidth=2, color=COLOR_STAYED)
    ax.set_xlabel("day")
    ax.set_ylabel("mean |observed − true| skill")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def _plot_match_quality_over_time(ax, aggregate: pl.DataFrame) -> None:
    days = aggregate["day"].to_numpy()
    matches = aggregate["matches_played"].to_numpy().astype(float)
    blowouts = aggregate["blowouts"].to_numpy().astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(matches > 0, blowouts / matches, np.nan)
    ax.plot(days, share, linewidth=2, color=COLOR_LEFT)
    ax.set_xlabel("day")
    ax.set_ylabel("blowout share of matches")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)


def _plot_who_left_vs_stayed(ax, population: pl.DataFrame) -> None:
    """True-skill histograms for churned vs still-active players, taken
    from the final day snapshot."""
    last_day = int(population["day"].max())
    final = population.filter(pl.col("day") == last_day)
    if final.height == 0:
        ax.text(0.5, 0.5, "no final-day snapshot", ha="center", va="center")
        return
    stayed = final.filter(pl.col("active"))["true_skill"].to_numpy()
    left = final.filter(~pl.col("active"))["true_skill"].to_numpy()
    if stayed.size == 0 and left.size == 0:
        ax.text(0.5, 0.5, "no players", ha="center", va="center")
        return

    all_skills = np.concatenate([stayed, left]) if stayed.size and left.size else (
        stayed if stayed.size else left
    )
    lo = float(all_skills.min()) - 0.1
    hi = float(all_skills.max()) + 0.1
    bins = np.linspace(lo, hi, 40)

    if stayed.size:
        ax.hist(
            stayed,
            bins=bins,
            alpha=0.55,
            color=COLOR_STAYED,
            label=f"still active (n={stayed.size})",
        )
    if left.size:
        ax.hist(
            left,
            bins=bins,
            alpha=0.55,
            color=COLOR_LEFT,
            label=f"churned (n={left.size})",
        )
    ax.set_xlabel("true_skill")
    ax.set_ylabel("player count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_true_skill_percentiles(ax, aggregate: pl.DataFrame) -> None:
    days = aggregate["day"].to_numpy()
    p10 = aggregate["true_skill_p10"].to_numpy()
    p50 = aggregate["true_skill_p50"].to_numpy()
    p90 = aggregate["true_skill_p90"].to_numpy()
    ax.fill_between(days, p10, p90, alpha=0.2, color=COLOR_STAYED, label="p10–p90")
    ax.plot(days, p50, linewidth=2, color=COLOR_STAYED, label="p50")
    ax.set_xlabel("day")
    ax.set_ylabel("true_skill (active players)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _plot_cumulative_score_by_skill(ax, population: pl.DataFrame) -> None:
    """Cumulative (wins − losses) averaged per day-0 true-skill decile.

    Positive drift = cohort winning more than losing.
    Negative drift = cohort losing more than winning.
    Lines that fan apart show matchmaking selecting for skill; lines
    that overlap show no skill signal in outcomes.
    """
    if "recent_losses" not in population.columns:
        ax.text(0.5, 0.5, "recent_losses not recorded", ha="center", va="center")
        return

    day_zero = population.filter(pl.col("day") == 0)
    if day_zero.height == 0:
        ax.text(0.5, 0.5, "no day-0 players", ha="center", va="center")
        return

    skills = day_zero["true_skill"].to_numpy()
    ids = day_zero["player_id"].to_numpy()
    deciles = np.percentile(skills, np.arange(0, 101, 10))
    decile_idx = np.clip(
        np.searchsorted(deciles[1:-1], skills, side="right"), 0, 9
    )
    id_to_decile = {int(pid): int(d) for pid, d in zip(ids, decile_idx)}

    # Compute cumulative (wins - losses) per (player, day), then average
    # across each decile. recent_wins/recent_losses are per-tick counts,
    # so summing across days gives cumulative totals. Positive = winning
    # more than losing; negative = losing more than winning.
    sub = (
        population.select(["day", "player_id", "recent_losses", "recent_wins"])
        .sort(["player_id", "day"])
        .with_columns(
            (pl.col("recent_wins") - pl.col("recent_losses")).alias("diff")
        )
        .with_columns(
            pl.col("diff").cum_sum().over("player_id").alias("cum_diff")
        )
    )
    sub = sub.with_columns(
        pl.col("player_id")
        .map_elements(lambda pid: id_to_decile.get(int(pid), -1), return_dtype=pl.Int32)
        .alias("decile")
    ).filter(pl.col("decile") >= 0)

    by_decile = (
        sub.group_by(["day", "decile"])
        .agg(pl.col("cum_diff").mean().alias("mean_cum_diff"))
        .sort(["decile", "day"])
    )

    decile_labels = [
        "bottom 10%",
        "10–20%",
        "20–30%",
        "30–40%",
        "40–50%",
        "50–60%",
        "60–70%",
        "70–80%",
        "80–90%",
        "top 10%",
    ]
    colors = ORANGE_BLUE(np.linspace(0.05, 0.95, 10))
    for d in range(10):
        curve = by_decile.filter(pl.col("decile") == d)
        if curve.height == 0:
            continue
        ax.plot(
            curve["day"].to_numpy(),
            curve["mean_cum_diff"].to_numpy(),
            linewidth=1.8,
            color=colors[d],
            label=decile_labels[d],
        )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("day")
    ax.set_ylabel("mean cumulative (wins − losses)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2, title="day-0 true skill")


def generate_plots_for_experiment_dir(version_dir: Path) -> list[Path]:
    """Regenerate plots for an already-saved experiment version directory."""
    aggregate = pl.read_parquet(version_dir / "aggregate.parquet")
    pop_path = version_dir / "population.parquet"
    if not pop_path.exists():
        raise FileNotFoundError(
            f"population.parquet not found in {version_dir}; "
            "can't generate plots without per-day population snapshots"
        )
    population = pl.read_parquet(pop_path)
    label = f"{version_dir.parent.name}/{version_dir.name}"
    return generate_plots(
        population=population,
        aggregate=aggregate,
        out_dir=version_dir / "plots",
        experiment_name=label,
    )
