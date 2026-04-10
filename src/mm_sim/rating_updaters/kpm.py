"""KPM rating updater: updates on per-player kills vs lobby average."""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class KPMUpdater:
    """Rating updates on kills z-score within the lobby.

    A player far above the lobby mean in kills gains rating regardless of
    whether their team won. This is the "what if the rating signal is a
    bad proxy for actual contribution?" experiment.
    """

    def __init__(self, cfg: RatingUpdaterConfig) -> None:
        self.cfg = cfg

    def update(self, result: MatchResult, pop: Population) -> None:
        flat_ids = result.flat_player_ids()
        kills = result.contributions["kills"]
        mean_kills = float(kills.mean())
        std = float(kills.std()) or 1.0
        z = (kills - mean_kills) / std
        k = self.cfg.k_factor / 400.0
        pop.observed_skill[flat_ids] += (k * z).astype(np.float32)
