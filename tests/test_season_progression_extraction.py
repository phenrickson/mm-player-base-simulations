"""Tests for extraction-mode season progression."""

from __future__ import annotations

import numpy as np

from mm_sim.config import PopulationConfig, SeasonProgressionConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population
from mm_sim.season_progression import apply_extraction_season_progression


def _pop(n: int) -> Population:
    cfg = PopulationConfig(initial_size=n)
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    pop.season_progress[:] = 0.0
    return pop


def test_base_per_match_derived_from_season_target():
    pop = _pop(3)
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.4]),
        kill_credits=[],
        team_strength=np.array([0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=1.0,
        extraction_weight=0.0,
        kill_weight=0.0,
    )
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )

    expected_per_match = 0.9 / (5.0 * 90)
    np.testing.assert_allclose(
        pop.season_progress[:3], [expected_per_match] * 3, atol=1e-6
    )


def test_extraction_weight_gates_on_extract():
    pop = _pop(6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.4, 0.4]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=0.0,
        extraction_weight=1.0,
        kill_weight=0.0,
    )
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )

    assert (pop.season_progress[:3] > 0).all()
    np.testing.assert_allclose(pop.season_progress[3:6], [0.0] * 3, atol=1e-7)


def test_concavity_diminishes_near_cap():
    pop = _pop(3)
    pop.season_progress[:] = 0.9
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.4]),
        kill_credits=[],
        team_strength=np.array([0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=1.0,
        extraction_weight=0.0,
        kill_weight=0.0,
    )
    before = pop.season_progress[:3].copy()
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )
    gain = pop.season_progress[:3] - before

    base_per_match = 0.9 / (5.0 * 90)
    expected_gain = base_per_match * 0.1
    np.testing.assert_allclose(gain, [expected_gain] * 3, atol=1e-6)


def test_disabled_is_noop():
    pop = _pop(3)
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.4]),
        kill_credits=[],
        team_strength=np.array([0.0]),
    )
    cfg = SeasonProgressionConfig(enabled=False)
    before = pop.season_progress[:3].copy()
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )
    np.testing.assert_array_equal(pop.season_progress[:3], before)


def test_kill_weight_scales_earn_by_kill_count():
    pop = _pop(6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.4, 0.4]),
        kill_credits=[(0, 1), (0, 1)],  # 2 credits to team 0
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=0.0,
        extraction_weight=0.0,
        kill_weight=1.0,
    )
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )
    assert (pop.season_progress[:3] > 0).all()
    np.testing.assert_allclose(pop.season_progress[3:6], [0.0] * 3, atol=1e-7)
