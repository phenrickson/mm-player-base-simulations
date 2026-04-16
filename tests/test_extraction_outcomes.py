"""Tests for the softmax-based extraction outcome generator."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig, PopulationConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.extraction import ExtractionOutcomeGenerator
from mm_sim.population import Population


def _pop_with_skills(
    skills: list[float], gears: list[float] | None = None
) -> Population:
    pop_cfg = PopulationConfig(initial_size=len(skills))
    pop = Population.create_initial(pop_cfg, np.random.default_rng(0))
    pop.true_skill[:] = np.array(skills, dtype=np.float32)
    if gears is not None:
        pop.gear[:] = np.array(gears, dtype=np.float32)
    return pop


def test_team_strength_includes_gear_weight():
    pop = _pop_with_skills(
        [1.0, 0.0, -1.0, 2.0], gears=[0.5, 0.5, 0.5, 0.5]
    )
    lobby = Lobby(teams=[[0, 1], [2, 3]])
    cfg = OutcomeConfig(kind="extraction", gear_weight=0.5)
    gen = ExtractionOutcomeGenerator(cfg)
    result = gen.generate(lobby, pop, np.random.default_rng(0))

    # Team 0: mean(1.0, 0.0) + 0.5*mean(0.5,0.5) = 0.75
    # Team 1: mean(-1.0, 2.0) + 0.5*0.5 = 0.75
    assert result.team_strength is not None
    np.testing.assert_allclose(result.team_strength, [0.75, 0.75], atol=1e-5)


def test_mean_extractor_count_matches_target():
    """Sample mean extractor count over many matches hits the config target."""
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)
    totals = []
    for seed in range(5000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        totals.append(int(r.extracted.sum()))
    totals_arr = np.array(totals)
    assert abs(totals_arr.mean() - 1.8) < 0.05


def test_p_zero_extract_tail_rate():
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)
    zeros = 0
    trials = 10_000
    for seed in range(trials):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        if not r.extracted.any():
            zeros += 1
    p0 = zeros / trials
    assert abs(p0 - 0.01) < 0.005


def test_stronger_team_extracts_more_often():
    skills = [
        2.0, 2.0, 2.0,
        -2.0, -2.0, -2.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
        strength_sensitivity=2.0,
    )
    gen = ExtractionOutcomeGenerator(cfg)

    counts = np.zeros(4, dtype=int)
    for seed in range(1000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        counts += r.extracted.astype(int)

    assert counts[0] > counts[2]
    assert counts[0] > counts[3]
    assert counts[1] < counts[2]
    assert counts[1] < counts[3]


def test_expected_extract_sums_to_mean_target():
    """expected_extract is a Plackett-Luce marginal; it sums to the k drawn
    for that match. Averaged across many matches it approaches the target
    mean number of extractors."""
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)

    sums = []
    for seed in range(2000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        sums.append(float(r.expected_extract.sum()))
    sums_arr = np.array(sums)
    assert abs(sums_arr.mean() - 1.8) < 0.05


def test_kill_attribution_only_when_extractors_exist():
    """If nobody extracts, kill_credits is empty; if some extract, credits
    only reference actual extractors as killers."""
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)

    saw_empty = False
    saw_populated = False
    for seed in range(2000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        if not r.extracted.any():
            assert r.kill_credits == []
            saw_empty = True
        else:
            for killer, dead in r.kill_credits:
                assert bool(r.extracted[killer]) is True
                assert bool(r.extracted[dead]) is False
            if r.kill_credits:
                saw_populated = True
    assert saw_empty is True
    assert saw_populated is True


def test_higher_beta_concentrates_wins_on_stronger_team():
    """Raising strength_sensitivity (beta) should push the strongest team's
    extract rate higher."""
    skills = [
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        -0.5, -0.5, -0.5,
        -1.0, -1.0, -1.0,
    ]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])

    def run_and_count(beta: float) -> np.ndarray:
        cfg = OutcomeConfig(
            kind="extraction",
            mean_extractors_per_match=1.8,
            p_zero_extract=0.01,
            p_all_extract=0.03,
            strength_sensitivity=beta,
        )
        gen = ExtractionOutcomeGenerator(cfg)
        counts = np.zeros(4, dtype=int)
        for seed in range(500):
            r = gen.generate(lobby, pop, np.random.default_rng(seed))
            counts += r.extracted.astype(int)
        return counts

    counts_low = run_and_count(0.5)
    counts_high = run_and_count(5.0)
    assert counts_high[0] > counts_low[0]
    assert counts_high[3] < counts_low[3]
