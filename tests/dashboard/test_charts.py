"""Tests for dashboard chart builders."""
from __future__ import annotations

import polars as pl
import plotly.graph_objects as go

from mm_sim.dashboard import charts


def _agg(days: int = 10, start_pop: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "day": list(range(days)),
            "active_count": [start_pop - i for i in range(days)],
            "matches_played": [50 for _ in range(days)],
            "blowouts": [5 for _ in range(days)],
            "rating_error_mean": [0.1 * i for i in range(days)],
            "win_prob_dev_mean": [0.2 for _ in range(days)],
            "favorite_expected_extract_mean": [0.35 for _ in range(days)],
            "true_skill_p10": [0.1 for _ in range(days)],
            "true_skill_p50": [0.5 for _ in range(days)],
            "true_skill_p90": [0.9 for _ in range(days)],
        }
    )


def test_population_over_time_single_run_returns_one_trace():
    fig = charts.population_over_time([("run-a", _agg())])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "run-a"


def test_population_over_time_multi_run_returns_one_trace_per_run():
    fig = charts.population_over_time(
        [("a", _agg()), ("b", _agg()), ("c", _agg())]
    )
    assert len(fig.data) == 3
    assert [t.name for t in fig.data] == ["a", "b", "c"]


def test_retention_over_time_normalizes_to_day_zero():
    df = _agg(days=5, start_pop=100)  # 100, 99, 98, 97, 96
    fig = charts.retention_over_time([("r", df)])
    ys = list(fig.data[0].y)
    assert ys[0] == 1.0
    assert abs(ys[-1] - 0.96) < 1e-9


def test_match_quality_over_time_uses_favorite_expected_extract():
    fig = charts.match_quality_over_time([("r", _agg())])
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [0.35] * 10


def test_rating_error_over_time_has_trace():
    fig = charts.rating_error_over_time([("r", _agg())])
    assert len(fig.data) == 1


def test_blowout_share_over_time_handles_zero_matches():
    df = _agg(days=3).with_columns(pl.Series("matches_played", [0, 50, 50]))
    fig = charts.blowout_share_over_time([("r", df)])
    assert list(fig.data[0].y)[0] == 0.0  # div-by-zero guarded


def test_skill_distribution_returns_histogram():
    pop = pl.DataFrame(
        {"day": [9] * 5, "true_skill": [0.1, 0.3, 0.5, 0.7, 0.9]}
    )
    fig = charts.skill_distribution(pop, day=9)
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"
