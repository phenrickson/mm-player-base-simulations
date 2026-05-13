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


def test_snapshot_contains_season_progress_and_talent_ceiling():
    from mm_sim.config import SeasonProgressionConfig

    cfg = SimulationConfig(
        seed=7,
        season_days=3,
        population=PopulationConfig(initial_size=100),
        skill_progression=SkillProgressionConfig(enabled=True),
        season_progression=SeasonProgressionConfig(enabled=True),
    )
    engine = SimulationEngine(cfg)
    engine.run()
    frames = engine.snapshot_writer._pop_frames
    assert len(frames) >= 2
    for frame in frames:
        assert "talent_ceiling" in frame.columns
        assert "season_progress" in frame.columns
