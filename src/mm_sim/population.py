"""Population: struct-of-numpy-arrays for all player state.

Every per-player attribute is a dedicated numpy array so the simulation
can operate on the whole population at once without Python-object overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_sim.config import PopulationConfig


@dataclass
class Population:
    true_skill: np.ndarray              # hidden ground-truth skill
    observed_skill: np.ndarray          # matchmaker's estimate
    experience: np.ndarray              # normalized [0, 1]
    gear: np.ndarray                    # normalized [0, 1]
    active: np.ndarray                  # False once churned
    party_id: np.ndarray                # -1 means unassigned
    matches_played: np.ndarray
    recent_wins: np.ndarray             # rolling count inside churn window
    recent_blowout_losses: np.ndarray
    join_day: np.ndarray

    @property
    def size(self) -> int:
        return int(self.true_skill.shape[0])

    @classmethod
    def create_initial(
        cls, cfg: PopulationConfig, rng: np.random.Generator
    ) -> "Population":
        n = cfg.initial_size
        true_skill = _sample_skill(n, cfg, rng).astype(np.float32)
        return cls(
            true_skill=true_skill,
            observed_skill=np.full(n, cfg.starting_observed_skill, dtype=np.float32),
            experience=np.full(n, cfg.starting_experience, dtype=np.float32),
            gear=np.full(n, cfg.starting_gear, dtype=np.float32),
            active=np.ones(n, dtype=bool),
            party_id=np.full(n, -1, dtype=np.int32),
            matches_played=np.zeros(n, dtype=np.int32),
            recent_wins=np.zeros(n, dtype=np.int8),
            recent_blowout_losses=np.zeros(n, dtype=np.int8),
            join_day=np.zeros(n, dtype=np.int32),
        )

    def add_new_players(
        self,
        count: int,
        cfg: PopulationConfig,
        rng: np.random.Generator,
        day: int = 0,
    ) -> np.ndarray:
        if count <= 0:
            return np.array([], dtype=np.int32)
        new_true = _sample_skill(count, cfg, rng).astype(np.float32)
        start = self.size
        self.true_skill = np.concatenate([self.true_skill, new_true])
        self.observed_skill = np.concatenate(
            [
                self.observed_skill,
                np.full(count, cfg.starting_observed_skill, dtype=np.float32),
            ]
        )
        self.experience = np.concatenate(
            [
                self.experience,
                np.full(count, cfg.starting_experience, dtype=np.float32),
            ]
        )
        self.gear = np.concatenate(
            [self.gear, np.full(count, cfg.starting_gear, dtype=np.float32)]
        )
        self.active = np.concatenate([self.active, np.ones(count, dtype=bool)])
        self.party_id = np.concatenate(
            [self.party_id, np.full(count, -1, dtype=np.int32)]
        )
        self.matches_played = np.concatenate(
            [self.matches_played, np.zeros(count, dtype=np.int32)]
        )
        self.recent_wins = np.concatenate(
            [self.recent_wins, np.zeros(count, dtype=np.int8)]
        )
        self.recent_blowout_losses = np.concatenate(
            [self.recent_blowout_losses, np.zeros(count, dtype=np.int8)]
        )
        self.join_day = np.concatenate(
            [self.join_day, np.full(count, day, dtype=np.int32)]
        )
        return np.arange(start, start + count, dtype=np.int32)

    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active).astype(np.int32)


def _sample_skill(
    n: int, cfg: PopulationConfig, rng: np.random.Generator
) -> np.ndarray:
    if cfg.true_skill_distribution == "normal":
        return rng.normal(cfg.true_skill_mean, cfg.true_skill_std, size=n)
    if cfg.true_skill_distribution == "uniform":
        half = cfg.true_skill_std * 1.732  # roughly matches target variance
        return rng.uniform(
            cfg.true_skill_mean - half, cfg.true_skill_mean + half, size=n
        )
    if cfg.true_skill_distribution == "right_skewed":
        raw = rng.lognormal(mean=0.0, sigma=0.8, size=n)
        raw = (raw - raw.mean()) / raw.std()
        return raw * cfg.true_skill_std + cfg.true_skill_mean
    raise ValueError(f"unknown distribution: {cfg.true_skill_distribution}")
