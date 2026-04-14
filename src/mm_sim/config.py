"""Pydantic configuration schema for a simulation run."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PopulationConfig(BaseModel):
    initial_size: int = Field(50_000, gt=0)
    true_skill_distribution: str = Field(
        "normal", pattern="^(normal|uniform|right_skewed)$"
    )
    true_skill_mean: float = 0.0
    true_skill_std: float = 1.0
    # Fraction of `initial_size` that joins each day. Frozen at startup
    # (arrivals don't follow the current population), so e.g. 0.03 with
    # initial_size=10000 means exactly 300 new players every day.
    daily_new_player_fraction: float = Field(0.0, ge=0.0)
    starting_observed_skill: float = 0.0
    starting_experience: float = 0.0
    starting_gear: float = 0.0


class PartyConfig(BaseModel):
    """Static party assignments at population creation."""

    # Map party size -> fraction of players in parties of that size.
    # {1: 0.5, 2: 0.25, 3: 0.25} means 50% solo, 25% duo, 25% trio.
    size_distribution: dict[int, float] = Field(
        default_factory=lambda: {1: 0.5, 2: 0.2, 3: 0.3}
    )
    # 0 = parties are random, 1 = parties are composed of identical-skill players.
    skill_homogeneity: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("size_distribution")
    @classmethod
    def _must_sum_to_one(cls, v: dict[int, float]) -> dict[int, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"size_distribution must sum to 1.0, got {total}")
        return v


class MatchmakerConfig(BaseModel):
    kind: str = Field("composite", pattern="^(random|composite)$")
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {"skill": 1.0, "experience": 0.0, "gear": 0.0}
    )
    lobby_size: int = Field(12, gt=1)
    teams_per_lobby: int = Field(2, gt=1)
    max_rating_spread: float = 0.3
    max_rating_spread_growth: float = 0.05

    @field_validator("composite_weights")
    @classmethod
    def _weights_nonnegative(cls, v: dict[str, float]) -> dict[str, float]:
        for k, val in v.items():
            if val < 0:
                raise ValueError(f"weight {k} must be >= 0, got {val}")
        return v


class OutcomeConfig(BaseModel):
    kind: str = "default"
    noise_std: float = 0.25
    blowout_threshold: float = 30.0


class RatingUpdaterConfig(BaseModel):
    kind: str = Field("elo", pattern="^(elo|kpm)$")
    k_factor: float = 32.0


class ChurnConfig(BaseModel):
    baseline_daily_quit_prob: float = 0.005
    loss_weight: float = 0.05
    blowout_loss_weight: float = 0.08
    win_streak_weight: float = -0.02
    rolling_window: int = 5
    max_daily_quit_prob: float = 0.5
    # New-player sensitivity: loss-driven churn terms get multiplied by
    # (1 + new_player_bonus * (1 - matches_played / new_player_threshold))
    # clipped so veterans (matches_played >= threshold) get a 1x multiplier.
    new_player_bonus: float = 1.0
    new_player_threshold: int = Field(20, gt=0)
    loss_streak_exp: float = 0.3
    max_loss_streak_multiplier: float = 4.0


class FrequencyConfig(BaseModel):
    mean_matches_per_day: float = 3.0
    win_modulation: float = 0.2
    loss_modulation: float = 0.15


class GearConfig(BaseModel):
    growth_per_match: float = 0.005
    drop_on_blowout_loss: float = 0.05
    max_gear: float = 1.0


class SimulationConfig(BaseModel):
    seed: int = 1999
    season_days: int = Field(90, gt=0)
    # Store full per-player snapshots this often. 1 = every day.
    population_snapshot_every_n_days: int = Field(1, gt=0)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    parties: PartyConfig = Field(default_factory=PartyConfig)
    matchmaker: MatchmakerConfig = Field(default_factory=MatchmakerConfig)
    outcomes: OutcomeConfig = Field(default_factory=OutcomeConfig)
    rating_updater: RatingUpdaterConfig = Field(default_factory=RatingUpdaterConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)
    frequency: FrequencyConfig = Field(default_factory=FrequencyConfig)
    gear: GearConfig = Field(default_factory=GearConfig)
