"""Tests for the extract-based Elo updater."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig, PopulationConfig, RatingUpdaterConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population
from mm_sim.rating_updaters.elo_extract import ExtractEloUpdater


def _pop(n: int) -> Population:
    cfg = PopulationConfig(initial_size=n)
    return Population.create_initial(cfg, np.random.default_rng(0))


def test_extracting_raises_rating_by_k_times_one_minus_expected():
    pop = _pop(3)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.7]),
        kill_credits=[],
    )
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    updater = ExtractEloUpdater(cfg)
    updater.update(result, pop)

    # delta = 32 * (1 - 0.7) / 3 = 3.2
    np.testing.assert_allclose(pop.observed_skill[:3], [3.2, 3.2, 3.2], atol=1e-5)


def test_dying_lowers_rating_by_k_times_expected():
    pop = _pop(3)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([False]),
        expected_extract=np.array([0.7]),
        kill_credits=[],
    )
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    updater = ExtractEloUpdater(cfg)
    updater.update(result, pop)

    np.testing.assert_allclose(pop.observed_skill[:3], [-32 * 0.7 / 3] * 3, atol=1e-4)


def test_multi_team_lobby_each_team_updated_independently():
    pop = _pop(12)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, True, False, False]),
        expected_extract=np.array([0.8, 0.5, 0.5, 0.2]),
        kill_credits=[],
    )
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=30.0)
    updater = ExtractEloUpdater(cfg)
    updater.update(result, pop)

    np.testing.assert_allclose(pop.observed_skill[:3], [30 * 0.2 / 3] * 3, atol=1e-5)
    np.testing.assert_allclose(pop.observed_skill[3:6], [30 * 0.5 / 3] * 3, atol=1e-5)
    np.testing.assert_allclose(pop.observed_skill[6:9], [-30 * 0.5 / 3] * 3, atol=1e-5)
    np.testing.assert_allclose(pop.observed_skill[9:12], [-30 * 0.2 / 3] * 3, atol=1e-5)
