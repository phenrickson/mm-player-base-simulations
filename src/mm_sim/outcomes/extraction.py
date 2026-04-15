"""Extraction outcome generator: each team independently extracts or dies."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from mm_sim.config import OutcomeConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class ExtractionOutcomeGenerator:
    def __init__(self, cfg: OutcomeConfig) -> None:
        self.cfg = cfg
        # Noise sigma=1; threshold chosen so a team at match_mean extracts
        # with probability baseline_extract_prob.
        self._sigma = 1.0
        self._threshold = float(norm.ppf(1.0 - cfg.baseline_extract_prob))

    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult:
        n_teams = len(lobby.teams)
        strengths = np.zeros(n_teams, dtype=np.float32)
        for i, team in enumerate(lobby.teams):
            arr = np.array(team, dtype=np.int32)
            s = pop.true_skill[arr].astype(np.float32)
            if self.cfg.gear_weight > 0:
                s = s + self.cfg.gear_weight * pop.gear[arr].astype(np.float32)
            strengths[i] = s.mean()

        match_mean = float(strengths.mean())
        deltas = strengths - match_mean
        noise = rng.normal(0.0, self._sigma, size=n_teams).astype(np.float32)
        rolls = self.cfg.strength_sensitivity * deltas + noise
        extracted = rolls > self._threshold

        # Expected extract: P(roll > threshold | delta) under N(0, sigma) noise.
        z = (self._threshold - self.cfg.strength_sensitivity * deltas) / self._sigma
        expected_extract = 1.0 - norm.cdf(z)

        # Attribute kills.
        kill_credits: list[tuple[int, int]] = []
        extractor_idxs = np.flatnonzero(extracted)
        if extractor_idxs.size > 0:
            for dead in np.flatnonzero(~extracted):
                dead_strength = strengths[dead]
                above = [
                    i for i in extractor_idxs if strengths[i] > dead_strength
                ]
                if above:
                    killer = int(min(above, key=lambda i: strengths[i]))
                else:
                    killer = int(max(extractor_idxs, key=lambda i: strengths[i]))
                kill_credits.append((killer, int(dead)))

        return MatchResult(
            lobby=lobby,
            extracted=extracted,
            kill_credits=kill_credits,
            expected_extract=expected_extract.astype(np.float32),
            team_strength=strengths,
            winning_team=-1,
            contributions={},
        )
