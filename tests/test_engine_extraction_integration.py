"""End-to-end integration test: extraction mode runs without error."""

from __future__ import annotations

from mm_sim.config import (
    GearConfig,
    MatchmakerConfig,
    OutcomeConfig,
    PartyConfig,
    PopulationConfig,
    RatingUpdaterConfig,
    SeasonProgressionConfig,
    SimulationConfig,
    StageConfig,
)
from mm_sim.engine import SimulationEngine


def test_extraction_end_to_end_smoke():
    cfg = SimulationConfig(
        seed=42,
        season_days=5,
        population=PopulationConfig(initial_size=240),
        parties=PartyConfig(size_distribution={1: 0.5, 2: 0.2, 3: 0.3}),
        matchmaker=MatchmakerConfig(
            kind="two_stage",
            lobby_size=12,
            teams_per_lobby=4,
            team_formation=StageConfig(
                composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
            ),
            lobby_assembly=StageConfig(
                composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
            ),
        ),
        outcomes=OutcomeConfig(kind="extraction", gear_weight=0.5),
        gear=GearConfig(transfer_enabled=True),
        rating_updater=RatingUpdaterConfig(kind="elo_extract"),
        season_progression=SeasonProgressionConfig(enabled=True),
    )

    engine = SimulationEngine(cfg)
    df = engine.run()
    assert df.height >= 5
    pop = engine.population
    assert pop.season_progress.max() > 0.0
