"""Engine smoke test: short season, small population, verify it runs."""

import polars as pl

from mm_sim.config import PopulationConfig, SimulationConfig
from mm_sim.engine import SimulationEngine


def test_short_season_runs_end_to_end():
    cfg = SimulationConfig(
        seed=1,
        season_days=5,
        population=PopulationConfig(initial_size=500, daily_new_player_fraction=0.02),
    )
    df = SimulationEngine(cfg).run()
    assert isinstance(df, pl.DataFrame)
    # Day 0 pristine snapshot + 5 ticked days = 6 rows
    assert df.height == 6
    assert df["day"].to_list() == [0, 1, 2, 3, 4, 5]
    assert df["active_count"][0] > 0
    assert df["matches_played"].sum() > 0


def test_engine_is_deterministic():
    cfg = SimulationConfig(
        seed=7,
        season_days=3,
        population=PopulationConfig(initial_size=300),
    )
    df1 = SimulationEngine(cfg).run()
    df2 = SimulationEngine(cfg).run()
    assert df1.equals(df2)
