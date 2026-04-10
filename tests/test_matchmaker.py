"""Sanity tests for matchmakers: no double-booking, no split parties.

These are the invariants that must hold for any matchmaker. Beyond this,
the research question is best answered by running the full simulation
and looking at population dynamics.
"""

import numpy as np
import pytest

from mm_sim.config import MatchmakerConfig, PartyConfig, PopulationConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.matchmaker.composite_mm import CompositeRatingMatchmaker
from mm_sim.matchmaker.random_mm import RandomMatchmaker
from mm_sim.parties import assign_parties
from mm_sim.population import Population
from mm_sim.seeding import make_rng


def _fresh_pop(n: int = 2000) -> tuple[Population, np.random.Generator]:
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=n), rng)
    pop.observed_skill = pop.true_skill.copy()
    assign_parties(
        pop, PartyConfig(size_distribution={1: 0.5, 2: 0.3, 3: 0.2}), rng
    )
    return pop, rng


@pytest.mark.parametrize(
    "matchmaker_factory",
    [
        lambda: RandomMatchmaker(
            MatchmakerConfig(kind="random", lobby_size=12, teams_per_lobby=2)
        ),
        lambda: CompositeRatingMatchmaker(
            MatchmakerConfig(
                kind="composite",
                lobby_size=12,
                teams_per_lobby=2,
                composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
            )
        ),
    ],
)
def test_lobby_invariants(matchmaker_factory):
    pop, rng = _fresh_pop(2000)
    mm = matchmaker_factory()
    lobbies = list(mm.form_lobbies(pop.active_indices(), pop, rng))
    assert len(lobbies) > 0

    seen_players: set[int] = set()
    for lobby in lobbies:
        assert isinstance(lobby, Lobby)
        all_in_lobby: list[int] = []
        for team in lobby.teams:
            all_in_lobby.extend(team)
        # Full lobbies only
        assert len(all_in_lobby) == 12
        # No double-booking across lobbies
        for pid in all_in_lobby:
            assert pid not in seen_players, f"player {pid} in multiple lobbies"
            seen_players.add(pid)
        # Parties not split across teams
        for team_idx, team in enumerate(lobby.teams):
            team_parties = {int(pop.party_id[p]) for p in team}
            other_parties: set[int] = set()
            for other_idx, other_team in enumerate(lobby.teams):
                if other_idx == team_idx:
                    continue
                other_parties.update(int(pop.party_id[p]) for p in other_team)
            assert team_parties.isdisjoint(other_parties)
