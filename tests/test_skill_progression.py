"""Tests for skill progression (per-tick true_skill drift toward talent ceiling)."""

from __future__ import annotations

import numpy as np

from mm_sim.config import PopulationConfig, SkillProgressionConfig
from mm_sim.population import Population
from mm_sim.skill_progression import apply_skill_progression_update


def _make_pop(n: int = 1000, seed: int = 0) -> Population:
    cfg = PopulationConfig(initial_size=n)
    return Population.create_initial(cfg, np.random.default_rng(seed))


def test_disabled_is_noop():
    pop = _make_pop()
    before = pop.true_skill.copy()
    cfg = SkillProgressionConfig(enabled=False)
    matches = np.ones(pop.size, dtype=np.int32) * 5
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(0))
    np.testing.assert_array_equal(pop.true_skill, before)


def test_enabled_moves_true_skill_toward_ceiling_zero_noise():
    pop = _make_pop(n=5000, seed=1)
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.0)
    matches = np.full(pop.size, 10, dtype=np.int32)
    gap_before = pop.talent_ceiling - pop.true_skill
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(2))
    gap_after = pop.talent_ceiling - pop.true_skill
    assert np.all(gap_after < gap_before)
    expected_delta = gap_before * 10.0 / 75.0
    actual_delta = (pop.true_skill - (pop.talent_ceiling - gap_before)).astype(np.float32)
    np.testing.assert_allclose(actual_delta, expected_delta, rtol=1e-4)


def test_players_with_zero_matches_do_not_change():
    pop = _make_pop()
    before = pop.true_skill.copy()
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.1)
    matches = np.zeros(pop.size, dtype=np.int32)
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(3))
    np.testing.assert_array_equal(pop.true_skill, before)


def test_true_skill_clipped_at_ceiling():
    pop = _make_pop(n=100, seed=4)
    pop.true_skill = pop.talent_ceiling.copy()
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.5)
    matches = np.full(pop.size, 100, dtype=np.int32)
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(5))
    assert np.all(pop.true_skill <= pop.talent_ceiling + 1e-6)


def test_noise_produces_variance_across_runs():
    pop_a = _make_pop(n=2000, seed=9)
    pop_b = _make_pop(n=2000, seed=9)
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.05)
    matches = np.full(pop_a.size, 5, dtype=np.int32)
    apply_skill_progression_update(pop_a, matches, cfg, np.random.default_rng(100))
    apply_skill_progression_update(pop_b, matches, cfg, np.random.default_rng(200))
    assert not np.allclose(pop_a.true_skill, pop_b.true_skill)
