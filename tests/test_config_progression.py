"""Tests for progression config schemas (skill, gear transfer, season)."""

from __future__ import annotations

import pytest

from mm_sim.config import (
    PopulationConfig,
    SimulationConfig,
    SkillProgressionConfig,
)


def test_skill_progression_defaults_disabled():
    cfg = SimulationConfig()
    assert cfg.skill_progression.enabled is False
    assert cfg.skill_progression.tau == 75.0
    assert cfg.skill_progression.noise_std == 0.02
    assert cfg.skill_progression.starting_true_skill_fraction == 0.3


def test_skill_progression_tau_must_be_positive():
    with pytest.raises(Exception):
        SkillProgressionConfig(tau=0.0)


def test_skill_progression_fraction_in_unit_interval():
    with pytest.raises(Exception):
        SkillProgressionConfig(starting_true_skill_fraction=-0.1)
    with pytest.raises(Exception):
        SkillProgressionConfig(starting_true_skill_fraction=1.5)


def test_population_starting_true_skill_fraction_default():
    cfg = PopulationConfig()
    assert cfg.starting_true_skill_fraction == 0.3
