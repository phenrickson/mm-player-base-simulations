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
    return _line_chart(
        runs,
        "win_prob_dev_mean",
        "Match quality (win-prob deviation, lower is better)",
        "mean |win prob \u2212 0.5|",
    )


def rating_error_over_time(runs: RunList) -> go.Figure:
    return _line_chart(
        runs,
        "rating_error_mean",
        "Rating error over time",
        "mean |observed \u2212 true|",
    )


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
        ("Match quality (win-prob dev)", "win_prob_dev_mean", runs, ".3f"),
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
