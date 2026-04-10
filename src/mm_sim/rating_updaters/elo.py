"""Elo rating updater: win/loss only, k-factor scaled."""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class EloUpdater:
    def __init__(self, cfg: RatingUpdaterConfig) -> None:
        self.cfg = cfg

    def update(self, result: MatchResult, pop: Population) -> None:
        if len(result.lobby.teams) != 2:
            raise NotImplementedError(
                "EloUpdater only supports 2-team lobbies for v1"
            )

        team_a = np.array(result.lobby.teams[0], dtype=np.int32)
        team_b = np.array(result.lobby.teams[1], dtype=np.int32)
        r_a = float(pop.observed_skill[team_a].mean())
        r_b = float(pop.observed_skill[team_b].mean())

        # Expected scores (Elo with scale=1.0 on observed_skill space)
        expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 1.0))
        expected_b = 1.0 - expected_a

        actual_a = 1.0 if result.winning_team == 0 else 0.0
        actual_b = 1.0 - actual_a

        # Rescale k_factor (classic Elo uses 400; our skill space is ~[-3, 3])
        k = self.cfg.k_factor / 400.0
        pop.observed_skill[team_a] += k * (actual_a - expected_a)
        pop.observed_skill[team_b] += k * (actual_b - expected_b)
