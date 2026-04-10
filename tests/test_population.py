"""Sanity tests for Population: shapes, dtypes, and incremental growth.

These catch bugs where a downstream consumer would silently corrupt data
because an array had the wrong shape, dtype, or length.
"""

import numpy as np

from mm_sim.config import PopulationConfig
from mm_sim.population import Population
from mm_sim.seeding import make_rng


def test_initial_population_shapes_and_dtypes():
    pop = Population.create_initial(
        PopulationConfig(initial_size=1000), make_rng(42)
    )
    n = 1000
    assert pop.size == n
    for arr in (
        pop.true_skill,
        pop.observed_skill,
        pop.experience,
        pop.gear,
        pop.active,
        pop.party_id,
        pop.matches_played,
        pop.recent_wins,
        pop.recent_blowout_losses,
        pop.join_day,
    ):
        assert arr.shape == (n,)
    assert pop.true_skill.dtype == np.float32
    assert pop.active.dtype == bool
    assert pop.party_id.dtype == np.int32
    assert pop.active.all()
    # Normal(0, 1) sanity
    assert abs(pop.true_skill.mean()) < 0.1
    assert 0.9 < pop.true_skill.std() < 1.1


def test_add_new_players_extends_all_arrays():
    rng = make_rng(42)
    cfg = PopulationConfig(initial_size=100)
    pop = Population.create_initial(cfg, rng)
    pop.add_new_players(count=50, cfg=cfg, rng=rng, day=3)
    assert pop.size == 150
    assert pop.active[100:].all()
    assert (pop.join_day[100:] == 3).all()
    # All arrays should still line up
    assert pop.observed_skill.shape == (150,)
    assert pop.party_id.shape == (150,)
