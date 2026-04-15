"""Expected-vs-actual extract Elo updater."""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class ExtractEloUpdater:
    def __init__(self, cfg: RatingUpdaterConfig) -> None:
        self.cfg = cfg

    def update(self, result: MatchResult, pop: Population) -> None:
        if result.extracted is None or result.expected_extract is None:
            raise ValueError(
                "ExtractEloUpdater requires a MatchResult with extracted + "
                "expected_extract set (use the extraction outcome generator)"
            )
        for team_idx, team in enumerate(result.lobby.teams):
            actual = 1.0 if bool(result.extracted[team_idx]) else 0.0
            expected = float(result.expected_extract[team_idx])
            team_size = len(team)
            delta = self.cfg.k_factor * (actual - expected) / team_size
            for pid in team:
                pop.observed_skill[pid] += np.float32(delta)
