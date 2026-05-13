"""Tests for per-match gear transfer from losers to winners."""

from __future__ import annotations

import numpy as np

from mm_sim.config import GearConfig, PopulationConfig
from mm_sim.gear import apply_gear_transfer_for_match
from mm_sim.population import Population


def _make_pop(n: int, gear_values: np.ndarray) -> Population:
    pop = Population.create_initial(
        PopulationConfig(initial_size=n, starting_gear=0.5),
        np.random.default_rng(0),
    )
    pop.gear = gear_values.astype(np.float32)
    return pop


def test_transfer_disabled_is_noop():
    pop = _make_pop(6, np.full(6, 0.5, dtype=np.float32))
    before = pop.gear.copy()
    cfg = GearConfig(transfer_enabled=False)
    apply_gear_transfer_for_match(
        pop, winners=np.array([0, 1, 2]), losers=np.array([3, 4, 5]),
        is_blowout=False, cfg=cfg,
    )
    np.testing.assert_array_equal(pop.gear, before)


def test_transfer_moves_gear_from_losers_to_winners():
    pop = _make_pop(6, np.array([0.3, 0.3, 0.3, 0.6, 0.6, 0.6], dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.1)
    total_before = pop.gear.sum()
    apply_gear_transfer_for_match(
        pop, winners=np.array([0, 1, 2]), losers=np.array([3, 4, 5]),
        is_blowout=False, cfg=cfg,
    )
    total_after = pop.gear.sum()
    # Transfer is internal; total (up to clipping) is preserved.
    np.testing.assert_allclose(total_after, total_before, atol=1e-5)
    # Losers went down, winners went up.
    assert np.all(pop.gear[:3] > 0.3)
    assert np.all(pop.gear[3:] < 0.6)


def test_blowout_transfers_more_than_regular():
    pop_a = _make_pop(6, np.full(6, 0.5, dtype=np.float32))
    pop_b = _make_pop(6, np.full(6, 0.5, dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.01, transfer_rate_blowout=0.04)
    apply_gear_transfer_for_match(
        pop_a, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=False, cfg=cfg
    )
    apply_gear_transfer_for_match(
        pop_b, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=True, cfg=cfg
    )
    assert pop_b.gear[0] > pop_a.gear[0]
    assert pop_b.gear[3] < pop_a.gear[3]


def test_transfer_clipped_to_max_gear():
    pop = _make_pop(6, np.array([0.98, 0.98, 0.98, 0.5, 0.5, 0.5], dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.5, max_gear=1.0)
    apply_gear_transfer_for_match(
        pop, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=False, cfg=cfg
    )
    assert np.all(pop.gear <= 1.0 + 1e-6)


def test_transfer_clipped_at_zero():
    pop = _make_pop(6, np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01], dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.5)
    apply_gear_transfer_for_match(
        pop, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=False, cfg=cfg
    )
    assert np.all(pop.gear >= 0.0 - 1e-6)
