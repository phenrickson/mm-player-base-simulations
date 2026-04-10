"""Math sanity for outcome generator and rating updaters.

These are the places where a subtle bug would silently corrupt the
simulation's central claims, so they get dedicated tests.
"""

import numpy as np

from mm_sim.config import OutcomeConfig, PopulationConfig, RatingUpdaterConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.outcomes.default import DefaultOutcomeGenerator
from mm_sim.population import Population
from mm_sim.rating_updaters.elo import EloUpdater
from mm_sim.rating_updaters.kpm import KPMUpdater
from mm_sim.seeding import make_rng


def _fixed_pop(n: int = 12) -> Population:
    pop = Population.create_initial(
        PopulationConfig(initial_size=n), make_rng(0)
    )
    return pop


def test_higher_skill_team_wins_most_of_the_time():
    pop = _fixed_pop(12)
    pop.true_skill[:] = np.array(
        [0.0] * 6 + [2.0] * 6, dtype=np.float32
    )
    lobby = Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    gen = DefaultOutcomeGenerator(OutcomeConfig(noise_std=0.2))
    rng = make_rng(1)
    wins_b = sum(
        1
        for _ in range(100)
        if gen.generate(lobby, pop, rng).winning_team == 1
    )
    assert wins_b > 85, f"expected >85/100 high-skill wins, got {wins_b}"


def test_default_outcome_contribution_fields():
    pop = _fixed_pop(12)
    lobby = Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    result = DefaultOutcomeGenerator(OutcomeConfig()).generate(
        lobby, pop, make_rng(1)
    )
    assert set(result.contributions.keys()) == {
        "kills",
        "deaths",
        "damage",
        "objective_score",
    }
    for arr in result.contributions.values():
        assert arr.shape == (12,)


def test_elo_winners_gain_losers_lose():
    pop = _fixed_pop(12)
    pop.observed_skill[:] = 0.0
    result = MatchResult(
        lobby=Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
        winning_team=0,
        score_margin=10.0,
        is_blowout=False,
        contributions={
            k: np.zeros(12, dtype=np.float32)
            for k in ("kills", "deaths", "damage", "objective_score")
        },
    )
    EloUpdater(RatingUpdaterConfig(k_factor=32.0)).update(result, pop)
    assert (pop.observed_skill[:6] > 0).all()
    assert (pop.observed_skill[6:] < 0).all()


def test_kpm_high_kill_players_gain_even_on_losing_team():
    pop = _fixed_pop(12)
    pop.observed_skill[:] = 0.0
    kills = np.array(
        [5, 5, 5, 5, 5, 5, 20, 1, 1, 1, 1, 1], dtype=np.float32
    )
    result = MatchResult(
        lobby=Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
        winning_team=0,  # player 6 is on the losing team
        score_margin=10.0,
        is_blowout=False,
        contributions={
            "kills": kills,
            "deaths": np.ones(12, dtype=np.float32),
            "damage": np.zeros(12, dtype=np.float32),
            "objective_score": np.zeros(12, dtype=np.float32),
        },
    )
    KPMUpdater(RatingUpdaterConfig(k_factor=32.0)).update(result, pop)
    assert pop.observed_skill[6] > 0  # high-kill loser gains
    assert (pop.observed_skill[7:] < 0).all()  # low-kill losers drop
