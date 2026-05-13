"""Tests for extraction gear update."""

from __future__ import annotations

import numpy as np

from mm_sim.config import GearConfig, PopulationConfig
from mm_sim.gear import apply_extraction_gear_update
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


def _pop(gears: list[float]) -> Population:
    cfg = PopulationConfig(initial_size=len(gears))
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    pop.gear[:] = np.array(gears, dtype=np.float32)
    return pop


def test_extract_growth_only_for_extractors():
    pop = _pop([0.1] * 6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.5, 0.5]),
        kill_credits=[],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = GearConfig(extract_growth=0.02, transfer_rate=0.0)
    apply_extraction_gear_update(pop, result, cfg)

    np.testing.assert_allclose(pop.gear[:3], [0.12, 0.12, 0.12], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.1, 0.1, 0.1], atol=1e-5)


def test_killer_of_equal_strength_team_gets_floor_rate():
    pop = _pop([0.5] * 6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.5, 0.5]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = GearConfig(
        extract_growth=0.0,
        transfer_rate=0.1,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=0.9,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # delta = 0 -> rate = 0.1 * max(0.2, 1+0) = 0.1
    # losers strip 0.05 each, pool=0.15, *0.9=0.135, /3 = 0.045 per winner
    np.testing.assert_allclose(pop.gear[:3], [0.545, 0.545, 0.545], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.45, 0.45, 0.45], atol=1e-5)


def test_punching_down_uses_floor():
    pop = _pop([0.5] * 6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.9, 0.1]),
        kill_credits=[(0, 1)],
        team_strength=np.array([2.0, 0.0]),
    )
    cfg = GearConfig(
        extract_growth=0.0,
        transfer_rate=0.1,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=0.9,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # rate = 0.1 * max(0.2, 1 + 1*(-2)) = 0.1 * 0.2 = 0.02
    # losers lose 0.01 each, pool=0.03, winners +0.027/3 = 0.009 each
    np.testing.assert_allclose(pop.gear[:3], [0.509, 0.509, 0.509], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.49, 0.49, 0.49], atol=1e-5)


def test_upset_multiplies_transfer():
    pop = _pop([0.5] * 6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.1, 0.9]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 2.0]),
    )
    cfg = GearConfig(
        extract_growth=0.0,
        transfer_rate=0.05,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=0.9,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # rate = 0.05 * max(0.2, 1 + 2) = 0.15
    # losers lose 0.075 each, pool=0.225, winners +0.2025/3 = 0.0675 each
    np.testing.assert_allclose(pop.gear[:3], [0.5675, 0.5675, 0.5675], atol=1e-4)
    np.testing.assert_allclose(pop.gear[3:6], [0.425, 0.425, 0.425], atol=1e-4)


def test_no_extractors_skips_all_gear_updates():
    before = [0.5] * 6
    pop = _pop(before)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([False, False]),
        expected_extract=np.array([0.3, 0.3]),
        kill_credits=[],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = GearConfig(extract_growth=0.05, transfer_rate=0.1)
    apply_extraction_gear_update(pop, result, cfg)

    np.testing.assert_allclose(pop.gear, before, atol=1e-7)


def test_winner_cap_excess_goes_to_void():
    pop = _pop([0.95, 0.95, 0.95, 0.5, 0.5, 0.5])
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.9, 0.1]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = GearConfig(
        max_gear=1.0,
        extract_growth=0.0,
        transfer_rate=0.2,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=1.0,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # Winners capped at 1.0; losers still lose 0.2 * 0.5 = 0.1 each.
    np.testing.assert_allclose(pop.gear[:3], [1.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.4, 0.4, 0.4], atol=1e-5)
