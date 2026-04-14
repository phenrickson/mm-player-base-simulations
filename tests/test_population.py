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
        pop.talent_ceiling,
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
    # talent_ceiling carries the Normal(0, 1) draw; true_skill starts as a
    # fraction of it (default 0.3) so progression has room to grow toward it.
    assert abs(pop.talent_ceiling.mean()) < 0.1
    assert 0.9 < pop.talent_ceiling.std() < 1.1


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


def test_population_has_talent_ceiling():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=100)
    pop = Population.create_initial(cfg, np.random.default_rng(42))
    assert pop.talent_ceiling.shape == pop.true_skill.shape
    assert pop.talent_ceiling.dtype == np.float32
    assert pop.talent_ceiling.std() > 0


def test_initial_true_skill_is_fraction_of_ceiling():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=500, starting_true_skill_fraction=0.3)
    pop = Population.create_initial(cfg, np.random.default_rng(1))
    # true_skill = ceiling - |ceiling| * (1 - fraction), always <= ceiling
    expected = pop.talent_ceiling - np.abs(pop.talent_ceiling) * (1.0 - 0.3)
    np.testing.assert_allclose(pop.true_skill, expected.astype(np.float32), rtol=1e-5)
    assert np.all(pop.true_skill <= pop.talent_ceiling)


def test_add_new_players_extends_talent_ceiling():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=10)
    pop = Population.create_initial(cfg, np.random.default_rng(7))
    before = pop.talent_ceiling.shape[0]
    pop.add_new_players(5, cfg, np.random.default_rng(8), day=3)
    assert pop.talent_ceiling.shape[0] == before + 5
    assert pop.talent_ceiling.shape == pop.true_skill.shape
