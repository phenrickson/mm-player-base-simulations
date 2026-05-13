"""Tests for DailySnapshotWriter — covering edge cases in record_match and
record_aggregate that arise with extraction-mode (4-team lobbies, winning_team=-1).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mm_sim.snapshot import DailySnapshotWriter


def _four_team_lobby():
    """Return lobby_true_skills and team_true_skills for a 4-team match."""
    teams = [
        np.array([1.0, 1.2]),
        np.array([0.9, 1.1]),
        np.array([1.3, 0.8]),
        np.array([1.0, 1.0]),
    ]
    lobby = np.concatenate(teams)
    return lobby, teams


def test_record_match_no_extractor_no_warning():
    """record_match with winning_team=-1 and a 4-team lobby must not raise
    RuntimeWarning.
    """
    writer = DailySnapshotWriter()
    lobby_true, team_trues = _four_team_lobby()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        writer.record_match(
            day=1,
            match_idx=0,
            lobby_true_skills=lobby_true,
            team_true_skills=team_trues,
            is_blowout=False,
            winning_team=-1,
        )

    rows = writer._match_rows
    assert len(rows) == 1
    row = rows[0]
    assert row["winning_team"] == -1
    assert row["is_blowout"] is False
    # win_prob_dev is NaN for >2-team lobbies — that is expected behaviour.
    assert np.isnan(row["win_prob_dev"])


def test_record_aggregate_all_nan_win_prob_dev_no_warning():
    """record_aggregate must not raise RuntimeWarning when every match in the
    day has win_prob_dev=NaN (i.e. all-NaN slice).  This happens in
    extraction mode with 4-team lobbies.
    """
    from mm_sim.population import Population
    from mm_sim.config import PopulationConfig

    writer = DailySnapshotWriter()
    lobby_true, team_trues = _four_team_lobby()

    # Record several matches — all 4-team, so win_prob_dev is always NaN.
    for i in range(5):
        writer.record_match(
            day=1,
            match_idx=i,
            lobby_true_skills=lobby_true,
            team_true_skills=team_trues,
            is_blowout=False,
            winning_team=-1,
        )

    pop = Population.create_initial(
        PopulationConfig(initial_size=10), rng=np.random.default_rng(0)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        writer.record_aggregate(day=1, pop=pop, matches_today=5, blowouts_today=0)

    df = writer.aggregate_dataframe()
    assert len(df) == 1
    row = df.row(0, named=True)
    # win_prob_dev columns should be NaN (not 0.0) when no 2-team matches exist.
    assert np.isnan(row["win_prob_dev_mean"])
    assert np.isnan(row["win_prob_dev_p50"])
    assert np.isnan(row["win_prob_dev_p90"])


def test_record_match_four_team_metrics():
    """team_gap is well-defined for 4-team lobbies (max - min of team means)."""
    writer = DailySnapshotWriter()
    lobby_true, team_trues = _four_team_lobby()

    writer.record_match(
        day=1,
        match_idx=0,
        lobby_true_skills=lobby_true,
        team_true_skills=team_trues,
        is_blowout=False,
        winning_team=2,  # team 2 extracted
    )

    row = writer._match_rows[0]
    team_means = [t.mean() for t in team_trues]
    expected_gap = max(team_means) - min(team_means)
    assert abs(row["team_gap"] - expected_gap) < 1e-9
    assert row["winning_team"] == 2
