"""End-to-end progression tests (enabled vs disabled affects mean metrics)."""

from __future__ import annotations

from mm_sim.config import (
    PopulationConfig,
    SimulationConfig,
    SkillProgressionConfig,
)
from mm_sim.engine import SimulationEngine


def _cfg(skill_enabled: bool) -> SimulationConfig:
    return SimulationConfig(
        seed=123,
        season_days=5,
        population=PopulationConfig(
            initial_size=500, daily_new_player_fraction=0.0
        ),
        skill_progression=SkillProgressionConfig(
            enabled=skill_enabled, tau=50.0, noise_std=0.0
        ),
    )


def test_skill_progression_enabled_raises_mean_true_skill():
    engine = SimulationEngine(_cfg(skill_enabled=True))
    initial = float(engine.population.true_skill.mean())
    engine.run()
    final = float(engine.population.true_skill.mean())
    assert final > initial


def test_skill_progression_disabled_leaves_mean_true_skill_static():
    engine = SimulationEngine(_cfg(skill_enabled=False))
    initial = float(engine.population.true_skill.mean())
    engine.run()
    final = float(engine.population.true_skill.mean())
    assert abs(final - initial) < 0.05
