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
    assert cfg.gear.transfer_rate == 0.005
    assert cfg.gear.transfer_rate_blowout == 0.04


def test_gear_transfer_rates_nonnegative():
    with pytest.raises(Exception):
        GearConfig(transfer_rate=-0.1)
    with pytest.raises(Exception):
        GearConfig(transfer_rate_blowout=-0.1)


def test_season_progression_defaults_disabled():
    cfg = SimulationConfig()
    assert cfg.season_progression.enabled is False
    assert cfg.season_progression.earn_per_match == 0.005
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


def test_outcome_config_extraction_defaults():
    from mm_sim.config import OutcomeConfig

    cfg = OutcomeConfig(kind="extraction")
    assert cfg.kind == "extraction"
    assert cfg.mean_extractors_per_match == 1.8
    assert cfg.p_zero_extract == 0.01
    assert cfg.p_all_extract == 0.03
    assert cfg.strength_sensitivity == 2.0


def test_matchmaker_config_two_stage_defaults():
    from mm_sim.config import MatchmakerConfig

    cfg = MatchmakerConfig()
    assert cfg.lobby_size == 12
    assert cfg.teams_per_lobby == 2
    assert cfg.team_formation.composite_weights == {
        "skill": 1.0, "experience": 0.0, "gear": 0.0
    }
    assert cfg.team_formation.max_rating_spread == 0.3
    assert cfg.lobby_assembly.max_rating_spread == 0.3


def test_gear_config_extraction_defaults():
    from mm_sim.config import GearConfig

    cfg = GearConfig()
    assert cfg.extract_growth == 0.003
    assert cfg.strength_bonus == 1.0
    assert cfg.punching_down_floor == 0.2
    assert cfg.transfer_efficiency == 0.9


def test_season_progression_extraction_defaults():
    from mm_sim.config import SeasonProgressionConfig

    cfg = SeasonProgressionConfig()
    assert cfg.base_earn_per_season == 0.8
    assert cfg.concavity == 1.0
    assert cfg.participation_weight == 0.3
    assert cfg.extraction_weight == 0.5
    assert cfg.kill_weight == 0.2


def test_rating_updater_config_elo_extract():
    from mm_sim.config import RatingUpdaterConfig

    cfg = RatingUpdaterConfig(kind="elo_extract")
    assert cfg.kind == "elo_extract"
