"""Plotly chart builders shared by Single Run and Compare pages.

Each function accepts `runs: list[tuple[str, pl.DataFrame]]` — a list
of (label, aggregate) pairs — and returns a `plotly.graph_objects.Figure`
with one trace per run. Single-run pages pass a 1-element list; compare
pages pass N.
"""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl


RunList = list[tuple[str, pl.DataFrame]]


def _line_chart(
    runs: RunList,
    y_col: str,
    title: str,
    y_label: str,
) -> go.Figure:
    fig = go.Figure()
    for label, df in runs:
        fig.add_trace(
            go.Scatter(
                x=df["day"].to_list(),
                y=df[y_col].to_list(),
                mode="lines",
                name=label,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="day",
        yaxis_title=y_label,
        hovermode="x unified",
    )
    return fig


def population_over_time(runs: RunList) -> go.Figure:
    return _line_chart(
        runs, "active_count", "Active population over time", "active players"
    )


def retention_over_time(runs: RunList) -> go.Figure:
    """Active count normalized by each run's day-0 value."""
    normalized: RunList = []
    for label, df in runs:
        day0 = df["active_count"].item(0)
        factor = 1.0 / day0 if day0 else 0.0
        normalized.append(
            (label, df.with_columns((pl.col("active_count") * factor).alias("retention")))
        )
    return _line_chart(
        normalized, "retention", "Retention over time", "fraction of day-0 population"
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
    shared: RunList = []
    for label, df in runs:
        shared.append(
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
    return _line_chart(
        shared, "blowout_share", "Blowout share over time", "blowouts / matches"
    )


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
