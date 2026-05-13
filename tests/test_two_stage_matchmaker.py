"""Tests for the two-stage matchmaker."""

from __future__ import annotations

import numpy as np

from mm_sim.config import MatchmakerConfig, PopulationConfig, StageConfig
from mm_sim.matchmaker.two_stage import TwoStageMatchmaker
from mm_sim.population import Population


def _pop_with_parties(
    skills: list[float], party_ids: list[int]
) -> Population:
    cfg = PopulationConfig(initial_size=len(skills))
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    pop.true_skill[:] = np.array(skills, dtype=np.float32)
    pop.observed_skill[:] = np.array(skills, dtype=np.float32)
    pop.party_id[:] = np.array(party_ids, dtype=np.int32)
    return pop


def test_twelve_solos_form_four_teams_of_three():
    pop = _pop_with_parties(
        [float(i) / 12 for i in range(12)], list(range(12))
    )
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    assert len(lobbies[0].teams) == 4
    for team in lobbies[0].teams:
        assert len(team) == 3


def test_trio_stays_together_one_team():
    skills = [0.0] * 12
    party_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5]
    pop = _pop_with_parties(skills, party_ids)
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    trio_players = {0, 1, 2}
    assert any(trio_players.issubset(set(team)) for team in lobbies[0].teams)


def test_duo_plus_solo_teams_together():
    skills = [0.0] * 12
    party_ids = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8]
    pop = _pop_with_parties(skills, party_ids)
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    for duo in [{0, 1}, {3, 4}, {6, 7}]:
        assert any(duo.issubset(set(team)) for team in lobbies[0].teams)


def test_partial_lobby_dropped():
    pop = _pop_with_parties([0.0] * 11, list(range(11)))
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(11, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert lobbies == []


def test_stage1_groups_by_rating_proximity():
    skills = [0.0, 0.1, 0.2, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.8, 1.9, 2.0]
    pop = _pop_with_parties(skills, list(range(12)))
    cfg = MatchmakerConfig(
        kind="two_stage",
        lobby_size=12,
        teams_per_lobby=4,
        team_formation=StageConfig(
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
        ),
        lobby_assembly=StageConfig(
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
        ),
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    assert any({0, 1, 2}.issubset(set(team)) for team in lobbies[0].teams)
