"""Plotly chart builders shared by Single Run and Compare pages.

Each function accepts `runs: list[tuple[str, pl.DataFrame]]` — a list
of (label, aggregate) pairs — and returns a `plotly.graph_objects.Figure`
with one trace per run. Single-run pages pass a 1-element list; compare
pages pass N.
"""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


RunList = list[tuple[str, pl.DataFrame]]


def _apply_retention(runs: RunList) -> RunList:
    out: RunList = []
    for label, df in runs:
        day0 = df["active_count"].item(0)
        factor = 1.0 / day0 if day0 else 0.0
        out.append(
            (label, df.with_columns((pl.col("active_count") * factor).alias("retention")))
        )
    return out


def _apply_blowout_share(runs: RunList) -> RunList:
    out: RunList = []
    for label, df in runs:
        out.append(
            (
                label,
                df.with_columns(
                    pl.when(pl.col("matches_played") > 0)
                    .then(pl.col("blowouts") / pl.col("matches_played"))
                    .otherwise(0.0)
                    .alias("blowout_share")
                ),
            )
        )
    return out


def _line_chart(
    runs: RunList,
    y_col: str,
    title: str,
    y_label: str,
    fmt: str = ".3f",
) -> go.Figure:
    fig = go.Figure()
    colors = _color_map([label for label, _ in runs])
    for label, df in runs:
        fig.add_trace(
            go.Scatter(
                x=df["day"].to_list(),
                y=df[y_col].to_list(),
                mode="lines",
                name=label,
                line=dict(color=colors[label]),
                hovertemplate=f"{label}: %{{y:{fmt}}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="day",
        yaxis_title=y_label,
        hovermode="x unified",
        yaxis=dict(tickformat=fmt),
    )
    return fig


def population_over_time(runs: RunList) -> go.Figure:
    return _line_chart(
        runs, "active_count", "Active population over time", "active players",
        fmt=",.0f",
    )


def retention_over_time(runs: RunList) -> go.Figure:
    """Active count normalized by each run's day-0 value."""
    return _line_chart(
        _apply_retention(runs),
        "retention",
        "Retention over time",
        "fraction of day-0 population",
        fmt=".1%",
    )


def match_quality_over_time(runs: RunList) -> go.Figure:
    """Extraction-mode match quality: mean pre-noise extract probability of
    the strongest team in each lobby. A value near 1/n_teams means balanced
    matchmaking; higher means one team is clearly favored.
    """
    fig = _line_chart(
        runs,
        "favorite_expected_extract_mean",
        "Match quality (favorite's expected extract, lower is better)",
        "mean E[extract] of strongest team",
    )
    # Reference line at 0.25 (chance baseline for 4 teams).
    fig.add_hline(
        y=0.25,
        line_dash="dot",
        line_color="#888",
        annotation_text="chance (0.25)",
        annotation_position="bottom right",
    )
    return fig


def rating_error_over_time(runs: RunList) -> go.Figure:
    return _line_chart(
        runs,
        "rating_error_mean",
        "Rating error over time",
        "mean |observed \u2212 true|",
    )


def mm_calibration_daily(match_teams: pl.DataFrame) -> pl.DataFrame:
    """Per-day mean |MM-view E[extract] - true-skill E[extract]|.

    Returns a polars DataFrame with columns:
      * ``day``
      * ``gap``       — |obs-view Elo E[extract] − true E[extract]|
      * ``floor``     — |true-view Elo E[extract] − true E[extract]|,
                        i.e. the irreducible gap between the pairwise-Elo
                        formula and the normal-CDF true expected_extract
                        even when observed_skill equals true_skill.
                        Subtracting this from ``gap`` isolates the MM's
                        actual rating-calibration error.

    MM-view expected extract is computed with the same pairwise Elo
    formula the rating updater uses (see ``elo_extract.py``):

        view_a = mean over opponents b of
            1 / (1 + 10^((skill_b - skill_a) / ELO_SCALE))

    with ELO_SCALE = 1.0. For ``gap`` the inputs are observed_skill;
    for ``floor`` the inputs are true_skill.
    """
    if match_teams.height == 0:
        return pl.DataFrame({"day": [], "gap": [], "floor": []})

    # Rank within match so we can join each team against its opponents.
    ranked = match_teams.with_columns(
        pl.int_range(pl.len()).over(["day", "match_idx"]).alias("_slot")
    )

    # Self-join on (day, match_idx), keep pairs where slots differ.
    pairs = ranked.join(
        ranked.select([
            "day", "match_idx", "_slot",
            pl.col("mean_observed_skill_before").alias("_opp_obs"),
            pl.col("mean_true_skill_before").alias("_opp_true"),
        ]),
        on=["day", "match_idx"],
    ).filter(pl.col("_slot") != pl.col("_slot_right"))

    pairs = pairs.with_columns(
        (
            1.0
            / (1.0 + (10.0 ** (pl.col("_opp_obs") - pl.col("mean_observed_skill_before"))))
        ).alias("_pair_obs"),
        (
            1.0
            / (1.0 + (10.0 ** (pl.col("_opp_true") - pl.col("mean_true_skill_before"))))
        ).alias("_pair_true"),
    )
    per_team = pairs.group_by(["day", "match_idx", "_slot"]).agg(
        pl.col("_pair_obs").mean().alias("_obs_view_expected"),
        pl.col("_pair_true").mean().alias("_true_view_expected"),
        pl.col("expected_extract").first().alias("_true_expected"),
    )
    per_team = per_team.with_columns(
        (pl.col("_obs_view_expected") - pl.col("_true_expected")).abs().alias("_gap"),
        (pl.col("_true_view_expected") - pl.col("_true_expected")).abs().alias("_floor"),
    )
    daily = (
        per_team.group_by("day")
        .agg(
            pl.col("_gap").mean().alias("gap"),
            pl.col("_floor").mean().alias("floor"),
        )
        .sort("day")
    )
    return daily


def mm_calibration_over_time(
    runs: list[tuple[str, pl.DataFrame]],
) -> go.Figure:
    """Daily mean gap between MM-view and true-skill expected extract.

    ``runs`` is a list of (label, match_teams) pairs. Each run is
    plotted as a solid line (observed-skill gap) plus a dashed
    "floor" line showing the irreducible formula-mismatch gap if
    ratings were perfect. Gap above floor = MM rating error.
    """
    fig = go.Figure()
    colors = _color_map([label for label, _ in runs])
    for label, mt in runs:
        daily = mm_calibration_daily(mt)
        fig.add_trace(
            go.Scatter(
                x=daily["day"].to_list(),
                y=daily["gap"].to_list(),
                mode="lines",
                name=label,
                line=dict(color=colors[label]),
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["day"].to_list(),
                y=daily["floor"].to_list(),
                mode="lines",
                name=f"{label} (floor)",
                line=dict(color=colors[label], dash="dot"),
                opacity=0.7,
                hovertemplate=f"{label} floor: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="MM rating calibration: |MM's E[extract] − actual E[extract]| (lower is better)",
        xaxis_title="day",
        yaxis_title="mean gap per team",
        hovermode="x unified",
        yaxis=dict(tickformat=".3f", rangemode="tozero"),
    )
    return fig


def extracts_per_match_over_time(
    runs: list[tuple[str, pl.DataFrame]],
) -> go.Figure:
    """Share of matches with k=0,1,2,... teams extracting, per day.

    ``runs`` is a list of (label, match_teams) pairs. One subplot per
    scenario, stacked area showing the daily share of matches by extract
    count. Matches the visual style of other over-time Compare charts.
    """
    from plotly.subplots import make_subplots

    labels = [label for label, _ in runs]
    colors = _color_map(labels)

    # Per-run daily distributions; also collect the global max k so all
    # subplots share the same stack categories (and legend colors).
    per_run_daily: list[tuple[str, pl.DataFrame]] = []
    max_k = 0
    for label, mt in runs:
        per_match = (
            mt.group_by(["day", "match_idx"])
            .agg(pl.col("extracted").cast(pl.Int32).sum().alias("k"))
        )
        daily = (
            per_match.group_by(["day", "k"])
            .agg(pl.len().alias("n"))
            .with_columns(
                (pl.col("n") / pl.col("n").sum().over("day")).alias("share")
            )
            .sort(["day", "k"])
        )
        if daily.height:
            max_k = max(max_k, int(daily["k"].max()))
        per_run_daily.append((label, daily))

    k_values = list(range(max_k + 1))
    # Sequential palette for stack layers (k=0 light -> k=max dark).
    import plotly.express as px
    stack_colors = px.colors.sample_colorscale(
        "Viridis",
        [i / max(1, max_k) for i in k_values],
    )

    fig = make_subplots(
        rows=len(runs),
        cols=1,
        shared_xaxes=True,
        subplot_titles=labels,
        vertical_spacing=0.05,
    )
    for row_idx, (label, daily) in enumerate(per_run_daily, start=1):
        days = sorted(daily["day"].unique().to_list())
        for k, color in zip(k_values, stack_colors):
            k_rows = daily.filter(pl.col("k") == k)
            share_lookup = dict(
                zip(k_rows["day"].to_list(), k_rows["share"].to_list())
            )
            ys = [share_lookup.get(d, 0.0) for d in days]
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=ys,
                    mode="lines",
                    stackgroup=f"r{row_idx}",
                    line=dict(width=0.5, color=color),
                    fillcolor=color,
                    name=f"{k} teams extracted",
                    legendgroup=f"k{k}",
                    showlegend=(row_idx == 1),
                    hovertemplate=(
                        f"{label}<br>day %{{x}} \u2014 {k} teams: "
                        f"%{{y:.1%}}<extra></extra>"
                    ),
                ),
                row=row_idx, col=1,
            )
        fig.update_yaxes(
            tickformat=".0%", range=[0, 1], row=row_idx, col=1,
        )

    fig.update_layout(
        title="Share of matches by teams extracted (per day)",
        height=max(260, 180 * len(runs) + 80),
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.update_xaxes(title_text="day", row=len(runs), col=1)
    return fig


def blowout_share_over_time(runs: RunList) -> go.Figure:
    """blowouts / matches_played per day, with div-by-zero guarded."""
    return _line_chart(
        _apply_blowout_share(runs),
        "blowout_share",
        "Blowout share over time",
        "blowouts / matches",
    )


def small_multiples(runs: RunList) -> go.Figure:
    """2x2 subplot grid with one shared legend across all panels.

    Panels: active population, retention, match quality, rating error.
    """
    panels = [
        ("Active population", "active_count", runs, ",.0f"),
        ("Retention", "retention", _apply_retention(runs), ".1%"),
        ("Match quality (favorite E[extract])", "favorite_expected_extract_mean", runs, ".3f"),
        ("Rating error", "rating_error_mean", runs, ".3f"),
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[p[0] for p in panels],
        shared_xaxes=False,
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = _color_map([label for label, _ in runs])
    for panel_idx, (_, y_col, panel_runs, fmt) in enumerate(panels):
        r, c = positions[panel_idx]
        for label, df in panel_runs:
            fig.add_trace(
                go.Scatter(
                    x=df["day"].to_list(),
                    y=df[y_col].to_list(),
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=(panel_idx == 0),
                    line=dict(color=colors[label]),
                    hovertemplate=f"{label}: %{{y:{fmt}}}<extra></extra>",
                ),
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="day", row=r, col=c)
        fig.update_yaxes(tickformat=fmt, row=r, col=c)
    fig.update_layout(
        height=750,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=60, r=40, b=100),
    )
    return fig


def skill_percentile_bands(aggregate: pl.DataFrame) -> go.Figure:
    """p10/p50/p90 of true_skill over time, shaded band."""
    d = aggregate["day"].to_list()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["true_skill_p90"].to_list(),
            mode="lines", line=dict(width=0), showlegend=False,
            hovertemplate="p90: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["true_skill_p10"].to_list(),
            mode="lines", fill="tonexty",
            fillcolor="rgba(100,149,237,0.25)",
            line=dict(width=0), name="p10–p90",
            hovertemplate="p10: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["true_skill_p50"].to_list(),
            mode="lines", name="p50", line=dict(color="royalblue"),
            hovertemplate="p50: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="True-skill percentiles over time",
        xaxis_title="day", yaxis_title="true skill",
        hovermode="x unified",
    )
    return fig


def experience_gear_over_time(aggregate: pl.DataFrame) -> go.Figure:
    """Mean experience and gear over time, dual-axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    d = aggregate["day"].to_list()
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["experience_mean"].to_list(),
            mode="lines", name="experience",
            hovertemplate="experience: %{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["gear_mean"].to_list(),
            mode="lines", name="gear",
            hovertemplate="gear: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Mean experience and gear over time",
        hovermode="x unified",
        xaxis_title="day",
    )
    fig.update_yaxes(title_text="experience", secondary_y=False)
    fig.update_yaxes(title_text="gear", secondary_y=True)
    return fig


def lobby_range_percentiles(aggregate: pl.DataFrame) -> go.Figure:
    """Lobby skill range over time: p50–p90 shaded band + mean line."""
    d = aggregate["day"].to_list()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["lobby_range_p90"].to_list(),
            mode="lines", line=dict(width=0), showlegend=False,
            hovertemplate="p90: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["lobby_range_p50"].to_list(),
            mode="lines", fill="tonexty",
            fillcolor="rgba(100,149,237,0.25)",
            line=dict(width=0), name="p50–p90",
            hovertemplate="p50: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d, y=aggregate["lobby_range_mean"].to_list(),
            mode="lines", name="mean", line=dict(color="royalblue"),
            hovertemplate="mean: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Lobby skill range over time",
        xaxis_title="day", yaxis_title="skill range",
        hovermode="x unified",
    )
    return fig


def retention_by_skill_decile(population: pl.DataFrame) -> go.Figure:
    """Retention (active fraction) per day, grouped by day-0 skill decile."""
    day0 = population.filter(pl.col("day") == 0).select(
        ["player_id", "true_skill"]
    )
    deciles = day0.with_columns(
        pl.col("true_skill").qcut(10, labels=[str(i) for i in range(10)]).alias("decile")
    ).select(["player_id", "decile"])
    merged = population.join(deciles, on="player_id", how="inner")
    base = (
        merged.filter(pl.col("day") == 0)
        .group_by("decile")
        .agg(pl.col("active").sum().alias("base"))
    )
    daily = (
        merged.group_by(["day", "decile"])
        .agg(pl.col("active").sum().alias("active"))
        .join(base, on="decile")
        .with_columns((pl.col("active") / pl.col("base")).alias("retention"))
        .sort(["decile", "day"])
    )
    fig = go.Figure()
    import plotly.express as px
    deciles_sorted = sorted(
        daily["decile"].unique().to_list(), key=lambda s: int(s)
    )
    colors = px.colors.sample_colorscale(
        "Viridis",
        [i / max(len(deciles_sorted) - 1, 1) for i in range(len(deciles_sorted))],
    )
    for dec, color in zip(deciles_sorted, colors):
        sub = daily.filter(pl.col("decile") == dec)
        fig.add_trace(
            go.Scatter(
                x=sub["day"].to_list(),
                y=sub["retention"].to_list(),
                mode="lines",
                name=f"d{dec}",
                line=dict(color=color),
                hovertemplate=f"decile {dec}: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Retention by day-0 skill decile",
        xaxis_title="day", yaxis_title="fraction still active",
        yaxis=dict(tickformat=".0%"),
        hovermode="x unified",
    )
    return fig


def retention_by_decile_faceted(
    runs: list[tuple[str, pl.DataFrame]],
) -> go.Figure:
    """2x5 grid of retention-over-time, one subplot per day-0 skill decile.

    ``runs`` is a list of ``(scenario_label, population)`` pairs. Each
    subplot shows one line per scenario, colored consistently across
    panels. Deciles are assigned per-run from that run's day-0 true_skill
    distribution (so a "decile 0" player in scenario A may have a
    different absolute skill than in scenario B — each run is qcut
    against its own population).
    """
    colors = _color_map([label for label, _ in runs])
    n_deciles = 10
    rows, cols = 2, 5
    decile_labels = [str(i) for i in range(n_deciles)]
    # Human-readable band labels: bottom 10%, 10-20%, ..., top 10%.
    def _band_label(i: int) -> str:
        if i == 0:
            return "bottom 10%"
        if i == n_deciles - 1:
            return "top 10%"
        return f"{i * 10}\u2013{(i + 1) * 10}%"
    band_titles = [_band_label(i) for i in range(n_deciles)]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=band_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    for run_idx, (label, population) in enumerate(runs):
        day0 = population.filter(pl.col("day") == 0).select(
            ["player_id", "true_skill"]
        )
        deciles = day0.with_columns(
            pl.col("true_skill")
            .qcut(n_deciles, labels=decile_labels)
            .alias("decile")
        ).select(["player_id", "decile"])
        merged = population.join(deciles, on="player_id", how="inner")
        base = (
            merged.filter(pl.col("day") == 0)
            .group_by("decile")
            .agg(pl.col("active").sum().alias("base"))
        )
        daily = (
            merged.group_by(["day", "decile"])
            .agg(pl.col("active").sum().alias("active"))
            .join(base, on="decile")
            .with_columns(
                (pl.col("active") / pl.col("base")).alias("retention")
            )
            .sort(["decile", "day"])
        )
        for dec_idx, dec in enumerate(decile_labels):
            sub = daily.filter(pl.col("decile") == dec)
            if sub.height == 0:
                continue
            r = dec_idx // cols + 1
            c = dec_idx % cols + 1
            fig.add_trace(
                go.Scatter(
                    x=sub["day"].to_list(),
                    y=sub["retention"].to_list(),
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=(dec_idx == 0),
                    line=dict(color=colors[label]),
                    hovertemplate=(
                        f"{label} ({band_titles[dec_idx]}): "
                        "%{y:.1%}<extra></extra>"
                    ),
                ),
                row=r,
                col=c,
            )
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=r, col=c)
        fig.update_xaxes(title_text="day", row=rows, col=c)
    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.08,
            xanchor="center", x=0.5,
        ),
        margin=dict(t=40, r=30, b=70),
    )
    return fig


def churn_rate_by_experience_cohort(
    runs: list[tuple[str, pl.DataFrame]],
) -> go.Figure:
    """1x3 grid: daily churn rate faceted by matches_played cohort.

    Cohorts:
      * new (<20 matches)
      * casual (20-49 matches)
      * experienced (>=50 matches)

    Each panel shows one line per scenario. Churn rate on day ``d`` is
    ``quits_on_d / eligible_on_d`` where a player is eligible if they
    were active on day d-1 and their matches_played on day d falls in
    the cohort's range.
    """
    cohorts = [
        ("new (<20 matches)", 0, 20),
        ("casual (20\u201349 matches)", 20, 50),
        ("experienced (\u226550 matches)", 50, None),
    ]
    colors = _color_map([label for label, _ in runs])
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[c[0] for c in cohorts],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    for run_idx, (label, population) in enumerate(runs):
        df = population.select(
            ["day", "player_id", "matches_played", "active"]
        ).sort(["player_id", "day"])
        df = df.with_columns(
            pl.col("active").shift(1).over("player_id").alias("prev_active"),
        )
        df = df.with_columns(
            (pl.col("prev_active") & ~pl.col("active"))
            .fill_null(False)
            .alias("quit")
        )
        for col_idx, (_title, lo, hi) in enumerate(cohorts):
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
            rate = (
                grouped["quits"].cast(pl.Float64)
                / grouped["cohort"].cast(pl.Float64)
            ).to_list()
            fig.add_trace(
                go.Scatter(
                    x=grouped["day"].to_list(),
                    y=rate,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=(col_idx == 0),
                    line=dict(color=colors[label]),
                    hovertemplate=f"{label}: %{{y:.2%}}<extra></extra>",
                ),
                row=1,
                col=col_idx + 1,
            )
    for c in range(1, 4):
        fig.update_xaxes(title_text="day", row=1, col=c)
        fig.update_yaxes(tickformat=".1%", rangemode="tozero", row=1, col=c)
    fig.update_yaxes(title_text="daily churn rate", row=1, col=1)
    fig.update_layout(
        height=380,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
        ),
        margin=dict(t=30, r=30, b=80),
    )
    return fig


def cohort_metric_by_skill_decile(
    population: pl.DataFrame, metric: str
) -> go.Figure:
    """Mean of `metric` per day, grouped by day-0 skill decile (active players only)."""
    day0 = population.filter(pl.col("day") == 0).select(
        ["player_id", "true_skill"]
    )
    deciles = day0.with_columns(
        pl.col("true_skill").qcut(10, labels=[str(i) for i in range(10)]).alias("decile")
    ).select(["player_id", "decile"])
    merged = population.join(deciles, on="player_id", how="inner")
    daily = (
        merged.filter(pl.col("active"))
        .group_by(["day", "decile"])
        .agg(pl.col(metric).mean().alias("v"))
        .sort(["decile", "day"])
    )
    fig = go.Figure()
    import plotly.express as px
    deciles_sorted = sorted(
        daily["decile"].unique().to_list(), key=lambda s: int(s)
    )
    colors = px.colors.sample_colorscale(
        "Viridis",
        [i / max(len(deciles_sorted) - 1, 1) for i in range(len(deciles_sorted))],
    )
    for dec, color in zip(deciles_sorted, colors):
        sub = daily.filter(pl.col("decile") == dec)
        fig.add_trace(
            go.Scatter(
                x=sub["day"].to_list(),
                y=sub["v"].to_list(),
                mode="lines",
                name=f"d{dec}",
                line=dict(color=color),
                hovertemplate=f"decile {dec}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Mean {metric} by day-0 skill decile (active players)",
        xaxis_title="day", yaxis_title=f"mean {metric}",
        hovermode="x unified",
    )
    return fig


def player_trajectories(
    population: pl.DataFrame,
    player_ids: list[int],
    metric: str,
    color_by: str | None = "day-0 true_skill",
    x_axis: str = "season",
) -> go.Figure:
    """Plot one metric over time per selected player.

    Lines end at each player's last active day. `color_by` is a named
    preset like "day-0 true_skill" or "final observed_skill"; None uses
    a single uniform color. Colors use the Viridis scale.
    """
    import plotly.express as px

    sub = (
        population.filter(pl.col("player_id").is_in(player_ids))
        .filter(pl.col("active"))
        .sort("day")
    )
    if x_axis == "player":
        sub = sub.with_columns(
            (pl.col("day") - pl.col("join_day")).alias("_x")
        )
        x_title = "player day (day \u2212 join_day)"
    else:
        sub = sub.with_columns(pl.col("day").alias("_x"))
        x_title = "day"

    color_lookup: dict[int, float] = {}
    if color_by is not None:
        when, _, col = color_by.partition(" ")
        if when == "day-0":
            snap = (
                population.filter(pl.col("day") == 0)
                .filter(pl.col("player_id").is_in(player_ids))
                .select(["player_id", col])
            )
        elif when == "final":
            snap = (
                sub.group_by("player_id")
                .agg(pl.col(col).last().alias(col))
            )
        else:
            snap = None
        if snap is not None and snap.height:
            color_lookup = dict(
                zip(snap["player_id"].to_list(), snap[col].to_list())
            )
    if color_lookup:
        c_min = min(color_lookup.values())
        c_max = max(color_lookup.values())
        c_range = (c_max - c_min) or 1.0
    else:
        c_min, c_max, c_range = 0.0, 1.0, 1.0

    fig = go.Figure()
    for pid in player_ids:
        ps = sub.filter(pl.col("player_id") == pid)
        if ps.height == 0:
            continue
        if color_by is not None and pid in color_lookup:
            val = color_lookup[pid]
            frac = (val - c_min) / c_range
            rgb = px.colors.sample_colorscale("Viridis", [frac])[0]
            hover_extra = f" ({color_by}: {val:.2f})"
        else:
            rgb = "#6fa8dc"
            hover_extra = ""
        fig.add_trace(
            go.Scatter(
                x=ps["_x"].to_list(),
                y=ps[metric].to_list(),
                mode="lines",
                line=dict(color=rgb, width=1.5),
                opacity=0.55,
                showlegend=False,
                name=f"player {pid}",
                hovertemplate=(
                    f"player {pid}{hover_extra}"
                    f"<br>{x_title} %{{x}} \u2014 {metric}: %{{y:.3f}}<extra></extra>"
                ),
            )
        )
    if color_by is not None and color_lookup:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(
                    colorscale="Viridis", cmin=c_min, cmax=c_max,
                    color=[c_min], showscale=True,
                    colorbar=dict(title=color_by),
                ),
                showlegend=False, hoverinfo="skip",
            )
        )
    color_note = f"color = {color_by} (Viridis)" if color_by else "uniform color"
    fig.update_layout(
        title=f"{metric} over time ({color_note}; line ends at churn)",
        xaxis_title=x_title, yaxis_title=metric,
        hovermode="closest",
    )
    return fig


def player_scatter(
    population: pl.DataFrame, day: int, x_col: str, y_col: str
) -> go.Figure:
    """Scatter of two per-player columns on a given day, with Pearson r."""
    snap = population.filter(pl.col("day") == day).filter(pl.col("active"))
    if snap.height == 0:
        fig = go.Figure()
        fig.update_layout(title=f"No active players on day {day}")
        return fig
    x = snap[x_col].to_list()
    y = snap[y_col].to_list()
    pids = snap["player_id"].to_list()
    r = snap.select(pl.corr(x_col, y_col)).item()
    fig = go.Figure(
        go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(size=4, opacity=0.4),
            customdata=pids,
            hovertemplate=(
                "player %{customdata}"
                f"<br>{x_col}: %{{x:.3f}}"
                f"<br>{y_col}: %{{y:.3f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{y_col} vs {x_col} on day {day}  (r = {r:.3f}, n = {snap.height:,})",
        xaxis_title=x_col, yaxis_title=y_col,
    )
    return fig


def matches_metric_distribution(matches: pl.DataFrame, column: str) -> go.Figure:
    """Histogram of a per-match metric (lobby_range, team_gap, win_prob_dev)."""
    fig = go.Figure(
        go.Histogram(
            x=matches[column].to_list(),
            nbinsx=50,
            hovertemplate=f"{column}: %{{x:.3f}}<br>count: %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column, yaxis_title="matches",
    )
    return fig


def matches_metric_over_time(matches: pl.DataFrame, column: str) -> go.Figure:
    """Per-day mean of a per-match metric."""
    daily = matches.group_by("day").agg(pl.col(column).mean().alias("v")).sort("day")
    fig = go.Figure(
        go.Scatter(
            x=daily["day"].to_list(),
            y=daily["v"].to_list(),
            mode="lines",
            hovertemplate=f"day %{{x}}: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Mean {column} per day",
        xaxis_title="day", yaxis_title=f"mean {column}",
        hovermode="x",
    )
    return fig


def player_trajectory_by_scenario(
    cohort_runs: list[tuple[str, pl.DataFrame]],
    n_players: int = 500,
    seed: int = 1999,
) -> go.Figure:
    """Compare per-player true_skill trajectories across scenarios.

    For each selected scenario ``cohort_runs`` has its population
    DataFrame. A shared random sample of player_ids (present at day 0
    in every scenario) is drawn, and each player's `(matches_played,
    true_skill)` trajectory is plotted. One line per (player, scenario),
    colored by scenario, so the same player's three trajectories can be
    visually compared as they diverge from the shared day-0 starting
    point.
    """
    import random as _rnd

    colors = _color_map([label for label, _ in cohort_runs])
    fig = go.Figure()

    if not cohort_runs:
        return fig

    # Intersection of day-0 player IDs across every scenario.
    day0_sets: list[set[int]] = []
    for _label, pop in cohort_runs:
        ids = (
            pop.filter(pl.col("day") == 0)["player_id"].unique().to_list()
        )
        day0_sets.append(set(int(x) for x in ids))
    common = set.intersection(*day0_sets) if day0_sets else set()
    if not common:
        return fig

    rng = _rnd.Random(seed)
    sampled = rng.sample(
        sorted(common), k=min(n_players, len(common))
    )
    sample_list = sorted(sampled)

    # Legend proxies: one solid, invisible-x trace per scenario so the
    # legend swatch shows the true color (not the 0.15-opacity player
    # lines).
    for label, _pop in cohort_runs:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=colors[label], width=3),
                name=label,
                legendgroup=label,
                showlegend=True,
                hoverinfo="skip",
            )
        )
    for label, pop in cohort_runs:
        sub = (
            pop.filter(pl.col("player_id").is_in(sample_list))
            .filter(pl.col("active"))
            .select(["player_id", "matches_played", "true_skill"])
            .sort(["player_id", "matches_played"])
        )
        color = colors[label]
        for pid, group in sub.group_by("player_id", maintain_order=True):
            xs = group["matches_played"].to_list()
            ys = group["true_skill"].to_list()
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.4,
                    legendgroup=label,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=f"Per-player true_skill trajectories ({len(sample_list)} players)",
        xaxis_title="matches_played",
        yaxis_title="true_skill",
        hovermode=False,
        legend=dict(orientation="h", y=1.08),
        height=520,
    )
    return fig


def _color_map(labels: list[str]) -> dict[str, str]:
    """Stable color per label using Plotly's default qualitative palette."""
    import plotly.express as px

    palette = px.colors.qualitative.Plotly
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def skill_distribution(population: pl.DataFrame, day: int) -> go.Figure:
    """Histogram of true_skill for a single run's population on one day."""
    snap = population.filter(pl.col("day") == day)
    fig = go.Figure(
        go.Histogram(x=snap["true_skill"].to_list(), nbinsx=40, name=f"day {day}")
    )
    fig.update_layout(
        title=f"True-skill distribution on day {day}",
        xaxis_title="true skill",
        yaxis_title="player count",
    )
    return fig
