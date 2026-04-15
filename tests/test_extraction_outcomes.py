"""Tests for the extraction outcome generator."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig, PopulationConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.extraction import ExtractionOutcomeGenerator
from mm_sim.population import Population


def _pop_with_skills(skills: list[float], gears: list[float] | None = None) -> Population:
    pop_cfg = PopulationConfig(initial_size=len(skills))
    pop = Population.create_initial(pop_cfg, np.random.default_rng(0))
    pop.true_skill[:] = np.array(skills, dtype=np.float32)
    if gears is not None:
        pop.gear[:] = np.array(gears, dtype=np.float32)
    return pop


def test_team_strength_includes_gear_weight():
    pop = _pop_with_skills([1.0, 0.0, -1.0, 2.0], gears=[0.5, 0.5, 0.5, 0.5])
    lobby = Lobby(teams=[[0, 1], [2, 3]])
    cfg = OutcomeConfig(kind="extraction", gear_weight=0.5, baseline_extract_prob=0.4)
    gen = ExtractionOutcomeGenerator(cfg)
    result = gen.generate(lobby, pop, np.random.default_rng(0))

    # Team 0: mean(1.0, 0.0) + 0.5*mean(0.5,0.5) = 0.5 + 0.25 = 0.75
    # Team 1: mean(-1.0, 2.0) + 0.5*0.5 = 0.5 + 0.25 = 0.75
    assert result.team_strength is not None
    np.testing.assert_allclose(result.team_strength, [0.75, 0.75], atol=1e-5)


def test_baseline_extract_prob_at_match_mean():
    """A team exactly at match_mean should extract with baseline_extract_prob."""
    skills = [0.0] * 12  # 4 teams of 3, all equal strength
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(kind="extraction", baseline_extract_prob=0.4)
    gen = ExtractionOutcomeGenerator(cfg)

    extracted_counts = np.zeros(4, dtype=int)
    trials = 2000
    for seed in range(trials):
        result = gen.generate(lobby, pop, np.random.default_rng(seed))
        extracted_counts += result.extracted.astype(int)

    rates = extracted_counts / trials
    # With all teams at same strength, expected rate for each = baseline.
    np.testing.assert_allclose(rates, [0.4] * 4, atol=0.04)


def test_stronger_team_extracts_more_often():
    skills = [2.0, 2.0, 2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(kind="extraction", baseline_extract_prob=0.4, strength_sensitivity=1.0)
    gen = ExtractionOutcomeGenerator(cfg)

    counts = np.zeros(4, dtype=int)
    for seed in range(1000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        counts += r.extracted.astype(int)

    # Strong team extracts most, weak least.
    assert counts[0] > counts[2] > counts[1]
    assert counts[0] > 700  # dominates
    assert counts[1] < 200  # rarely


def test_kill_attribution_closest_stronger_extractor():
    """Dead team is credited to the weakest extractor above them in strength."""
    # Force outcome by setting huge strength gaps and zero noise.
    skills = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -3.0, -3.0, -3.0]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    # Sensitivity huge so noise doesn't flip outcomes.
    cfg = OutcomeConfig(
        kind="extraction", baseline_extract_prob=0.4, strength_sensitivity=10.0
    )
    gen = ExtractionOutcomeGenerator(cfg)
    result = gen.generate(lobby, pop, np.random.default_rng(42))

    # Expect teams 0 and 1 extract (strongest two), teams 2 and 3 die.
    assert bool(result.extracted[0]) is True
    assert bool(result.extracted[1]) is True
    assert bool(result.extracted[2]) is False
    assert bool(result.extracted[3]) is False

    credits = set(result.kill_credits)
    # Team 3 (weakest) is killed by team 1 (weakest extractor above 3).
    # Team 2 is killed by team 1 (weakest extractor above 2).
    assert (1, 2) in credits
    assert (1, 3) in credits


def test_no_extractors_no_kill_credits():
    """If nobody extracts, kill_credits is empty."""
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(kind="extraction", baseline_extract_prob=0.05)  # rare extract
    gen = ExtractionOutcomeGenerator(cfg)

    for seed in range(2000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        if not r.extracted.any():
            assert r.kill_credits == []
            return
    raise AssertionError("expected at least one no-extractor match in 2000 trials")
