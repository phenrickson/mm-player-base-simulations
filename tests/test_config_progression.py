"""Tests for progression config schemas (skill, gear transfer, season)."""

from __future__ import annotations

import pytest

from mm_sim.config import (
    GearConfig,
    PopulationConfig,
    SeasonProgressionConfig,
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


def test_gear_transfer_defaults_disabled():
    cfg = SimulationConfig()
    assert cfg.gear.transfer_enabled is False
    assert cfg.gear.transfer_rate == 0.01
    assert cfg.gear.transfer_rate_blowout == 0.04


def test_gear_transfer_rates_nonnegative():
    with pytest.raises(Exception):
        GearConfig(transfer_rate=-0.1)
    with pytest.raises(Exception):
        GearConfig(transfer_rate_blowout=-0.1)


def test_season_progression_defaults_disabled():
    cfg = SimulationConfig()
    assert cfg.season_progression.enabled is False
    assert cfg.season_progression.earn_per_match == 0.02
    assert cfg.season_progression.curve_steepness == 3.0
    assert cfg.season_progression.behind_weight == 0.02
    assert cfg.season_progression.boredom_weight == 0.01
    assert cfg.season_progression.boredom_cutoff == 0.7


def test_season_progression_earn_rate_nonnegative():
    with pytest.raises(Exception):
        SeasonProgressionConfig(earn_per_match=-0.01)


def test_season_progression_boredom_cutoff_in_unit():
    with pytest.raises(Exception):
        SeasonProgressionConfig(boredom_cutoff=1.5)
