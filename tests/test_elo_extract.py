"""Tests for the pairwise extract-based Elo updater."""

from __future__ import annotations

import numpy as np

from mm_sim.config import PopulationConfig, RatingUpdaterConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population
from mm_sim.rating_updaters.elo_extract import ELO_SCALE, ExtractEloUpdater


def _pop(n: int) -> Population:
    cfg = PopulationConfig(initial_size=n)
    return Population.create_initial(cfg, np.random.default_rng(0))


def _make_result(lobby: Lobby, extracted: list[bool]) -> MatchResult:
    return MatchResult(
        lobby=lobby,
        extracted=np.array(extracted),
        expected_extract=np.full(len(extracted), 0.5, dtype=np.float32),
        kill_credits=[],
    )


def test_single_team_lobby_no_update():
    """A lobby with only one team has no opponents; nothing to update."""
    pop = _pop(3)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2]])
    result = _make_result(lobby, [True])
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    ExtractEloUpdater(cfg).update(result, pop)
    np.testing.assert_array_equal(pop.observed_skill[:3], [0.0, 0.0, 0.0])


def test_two_team_equal_rating_winner_gains_half_k_half():
    """Two equal-rating teams, one extracts one dies. expected=0.5 for each.
    Winner gains k*0.5, loser loses k*0.5. Divided by n_opponents (1) and
    split across team_size (3).
    """
    pop = _pop(6)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = _make_result(lobby, [True, False])
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    ExtractEloUpdater(cfg).update(result, pop)
    # pair_delta for winning team = 32 * (1 - 0.5) = 16
    # per-player = 16 / 1 opp / 3 size (no scale divisor in new model)
    expected_win = 32.0 * 0.5 / 3.0
    np.testing.assert_allclose(
        pop.observed_skill[:3], [expected_win] * 3, atol=1e-5
    )
    np.testing.assert_allclose(
        pop.observed_skill[3:6], [-expected_win] * 3, atol=1e-5
    )


def test_both_teams_extract_is_draw_with_zero_delta_at_equal_rating():
    """Both extract => actual=0.5 for both. At equal rating, expected=0.5.
    Delta = 0 for everyone."""
    pop = _pop(6)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = _make_result(lobby, [True, True])
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    ExtractEloUpdater(cfg).update(result, pop)
    np.testing.assert_allclose(pop.observed_skill[:6], [0.0] * 6, atol=1e-6)


def test_higher_rated_team_gains_less_from_win():
    """Classic Elo: beating a weaker team earns less than beating a stronger one."""
    pop = _pop(6)
    pop.observed_skill[:3] = ELO_SCALE  # team A is one ELO_SCALE above B
    pop.observed_skill[3:6] = 0.0
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = _make_result(lobby, [True, False])  # favorite A wins
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    ExtractEloUpdater(cfg).update(result, pop)
    # expected_a vs b with ELO_SCALE diff: 1 / (1 + 10^(-1)) = 10/11
    # delta_a = 32 * (1 - 10/11) = 32/11 ≈ 2.909, per player ≈ 0.97
    gain_per_player_a = 32.0 * (1.0 - 10.0 / 11.0) / 3.0
    np.testing.assert_allclose(
        pop.observed_skill[:3] - ELO_SCALE, [gain_per_player_a] * 3, atol=1e-4
    )
    # b (weaker) lost as expected: actual=0, expected_b = 1/11
    loss_per_player_b = -32.0 * (1.0 / 11.0) / 3.0
    np.testing.assert_allclose(
        pop.observed_skill[3:6], [loss_per_player_b] * 3, atol=1e-4
    )


def test_four_team_lobby_pairs_summed_over_opponents():
    """4 teams, all equal rating. Teams 0 and 1 extract, teams 2 and 3 die.
    Each team has 3 opponents; delta = avg of 3 pairwise deltas.
    """
    pop = _pop(12)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    result = _make_result(lobby, [True, True, False, False])
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    ExtractEloUpdater(cfg).update(result, pop)
    # For team 0 (extracted): vs team1 draw(0.5), vs team2 win(1), vs team3 win(1)
    # At equal rating, expected=0.5 each. Sum of (actual-expected) = 0 + 0.5 + 0.5 = 1.0
    # total_pair_delta = 32 * 1.0 = 32; /3 opponents = 10.667; /3 players = 3.556
    per_player_extractor = 32.0 * 1.0 / 3.0 / 3.0
    per_player_dier = -per_player_extractor
    np.testing.assert_allclose(
        pop.observed_skill[:3], [per_player_extractor] * 3, atol=1e-5
    )
    np.testing.assert_allclose(
        pop.observed_skill[3:6], [per_player_extractor] * 3, atol=1e-5
    )
    np.testing.assert_allclose(
        pop.observed_skill[6:9], [per_player_dier] * 3, atol=1e-5
    )
    np.testing.assert_allclose(
        pop.observed_skill[9:12], [per_player_dier] * 3, atol=1e-5
    )
