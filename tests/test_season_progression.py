"""Tests for season progression: per-match earning and churn-pressure term."""

from __future__ import annotations

import numpy as np

from mm_sim.config import PopulationConfig, SeasonProgressionConfig
from mm_sim.population import Population
from mm_sim.season_progression import (
    apply_season_progression_update,
    expected_progress,
    season_churn_pressure,
)


def _make_pop(n: int = 100) -> Population:
    return Population.create_initial(
        PopulationConfig(initial_size=n), np.random.default_rng(0)
    )


def test_disabled_is_noop():
    pop = _make_pop()
    before = pop.season_progress.copy()
    cfg = SeasonProgressionConfig(enabled=False)
    matches = np.ones(pop.size, dtype=np.int32) * 5
    apply_season_progression_update(pop, matches, cfg)
    np.testing.assert_array_equal(pop.season_progress, before)


def test_earn_per_match_accumulates():
    pop = _make_pop(n=10)
    cfg = SeasonProgressionConfig(enabled=True, earn_per_match=0.05)
    matches = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    apply_season_progression_update(pop, matches, cfg)
    np.testing.assert_allclose(pop.season_progress, matches * 0.05, rtol=1e-5)


def test_earn_clipped_at_one():
    pop = _make_pop(n=3)
    cfg = SeasonProgressionConfig(enabled=True, earn_per_match=0.5)
    matches = np.full(3, 10, dtype=np.int32)
    apply_season_progression_update(pop, matches, cfg)
    assert np.all(pop.season_progress <= 1.0 + 1e-6)


def test_expected_progress_monotone_saturating():
    cfg = SeasonProgressionConfig(curve_steepness=3.0)
    values = [expected_progress(d, season_days=90, cfg=cfg) for d in range(0, 91, 10)]
    assert values[0] == 0.0
    for a, b in zip(values, values[1:]):
        assert b > a
    assert values[-1] < 1.0  # not fully saturated at d=90 unless steepness is huge
    assert values[-1] > 0.9


def test_churn_pressure_zero_when_disabled():
    cfg = SeasonProgressionConfig(enabled=False)
    progress = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    pressure = season_churn_pressure(progress, day=30, season_days=90, cfg=cfg)
    np.testing.assert_array_equal(pressure, np.zeros(3, dtype=np.float32))


def test_churn_pressure_behind_positive():
    cfg = SeasonProgressionConfig(
        enabled=True, behind_weight=0.02, boredom_weight=0.0, curve_steepness=3.0
    )
    expected = expected_progress(60, season_days=90, cfg=cfg)
    progress = np.array([expected - 0.2, expected, expected + 0.2], dtype=np.float32)
    pressure = season_churn_pressure(progress, day=60, season_days=90, cfg=cfg)
    assert pressure[0] > 0  # behind
    assert pressure[1] == 0  # on curve
    assert pressure[2] == 0  # ahead (but boredom_weight=0)


def test_churn_pressure_boredom_only_before_cutoff():
    cfg = SeasonProgressionConfig(
        enabled=True, behind_weight=0.0, boredom_weight=0.05,
        boredom_cutoff=0.7, curve_steepness=3.0,
    )
    # Day 10/90 = 0.11 (before cutoff) and day 80/90 = 0.89 (after cutoff).
    progress = np.array([1.0], dtype=np.float32)
    early = season_churn_pressure(progress, day=10, season_days=90, cfg=cfg)
    late = season_churn_pressure(progress, day=80, season_days=90, cfg=cfg)
    assert early[0] > 0
    assert late[0] == 0


def test_churn_uses_season_pressure_when_enabled():
    import numpy as np
    from mm_sim.churn import apply_churn
    from mm_sim.config import ChurnConfig, PopulationConfig, SeasonProgressionConfig
    from mm_sim.population import Population

    # Two identical populations; one with season pressure, one without.
    cfg_pop = PopulationConfig(initial_size=2000, starting_true_skill_fraction=1.0)
    pop_no_pressure = Population.create_initial(cfg_pop, np.random.default_rng(0))
    pop_with_pressure = Population.create_initial(cfg_pop, np.random.default_rng(0))

    # Make everyone have 0 progress while expected is ~0.6 — huge gap.
    pop_no_pressure.season_progress[:] = 0.0
    pop_with_pressure.season_progress[:] = 0.0

    churn_cfg = ChurnConfig(baseline_daily_quit_prob=0.0)
    season_cfg_on = SeasonProgressionConfig(
        enabled=True, behind_weight=0.5, curve_steepness=3.0
    )
    season_cfg_off = SeasonProgressionConfig(enabled=False)

    apply_churn(
        pop_no_pressure, churn_cfg, np.random.default_rng(1),
        day=30, season_days=90, season_cfg=season_cfg_off,
    )
    apply_churn(
        pop_with_pressure, churn_cfg, np.random.default_rng(1),
        day=30, season_days=90, season_cfg=season_cfg_on,
    )

    alive_no_pressure = int(pop_no_pressure.active.sum())
    alive_with_pressure = int(pop_with_pressure.active.sum())
    assert alive_with_pressure < alive_no_pressure
