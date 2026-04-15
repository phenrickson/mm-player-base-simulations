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
    for label, df in runs:
        fig.add_trace(
            go.Scatter(
                x=df["day"].to_list(),
                y=df[y_col].to_list(),
                mode="lines",
                name=label,
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
