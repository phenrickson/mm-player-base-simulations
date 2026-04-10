"""Engine smoke test: short season, small population, verify it runs."""

import polars as pl

from mm_sim.config import PopulationConfig, SimulationConfig
from mm_sim.engine import SimulationEngine


def test_short_season_runs_end_to_end():
    cfg = SimulationConfig(
        seed=1,
        season_days=5,
        population=PopulationConfig(initial_size=500, daily_new_players=10),
    )
    df = SimulationEngine(cfg).run()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 5
    assert df["day"].to_list() == [0, 1, 2, 3, 4]
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
