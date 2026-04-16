"""Extraction outcome generator: each team independently extracts or dies."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from mm_sim.config import OutcomeConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population

_STD_NORMAL = NormalDist()


class ExtractionOutcomeGenerator:
    def __init__(self, cfg: OutcomeConfig) -> None:
        self.cfg = cfg
        # Noise sigma=1; threshold chosen so a team at match_mean extracts
        # with probability baseline_extract_prob.
        self._sigma = 1.0
        self._threshold = float(_STD_NORMAL.inv_cdf(1.0 - cfg.baseline_extract_prob))

    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult:
        n_teams = len(lobby.teams)
        strengths = np.zeros(n_teams, dtype=np.float32)
        # Per-player skill-based performance (without noise). Used for
        # within-team contribution shares; per-player noise is added on top.
        player_perfs: list[np.ndarray] = []
        for i, team in enumerate(lobby.teams):
            arr = np.array(team, dtype=np.int32)
            s = pop.true_skill[arr].astype(np.float32)
            if self.cfg.gear_weight > 0:
                s = s + self.cfg.gear_weight * pop.gear[arr].astype(np.float32)
            strengths[i] = s.mean()  # noise-free team strength
            perf_noise = rng.normal(
                0.0, self.cfg.noise_std, size=len(arr)
            ).astype(np.float32)
            player_perfs.append(s + perf_noise)

        match_mean = float(strengths.mean())
        deltas = strengths - match_mean
        noise = rng.normal(0.0, self._sigma, size=n_teams).astype(np.float32)
        rolls = self.cfg.strength_sensitivity * deltas + noise
        extracted = rolls > self._threshold

        # Expected extract: P(roll > threshold | delta) under N(0, sigma) noise.
        z = (self._threshold - self.cfg.strength_sensitivity * deltas) / self._sigma
        expected_extract = np.array(
            [1.0 - _STD_NORMAL.cdf(float(zi)) for zi in z],
            dtype=np.float32,
        )

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

        # Per-player contributions, ordered to match result.flat_player_ids().
        # `player_perf` is each player's effective performance this match;
        # within a team we normalize to a positive share used by the rating
        # updater to weight per-player rating deltas.
        flat_perf = np.concatenate(player_perfs).astype(np.float32)
        # Within-team normalized share (positive, sums to team_size per team).
        shares = np.zeros_like(flat_perf)
        cursor = 0
        for i, team in enumerate(lobby.teams):
            n = len(team)
            team_perf = flat_perf[cursor : cursor + n]
            # Shift to strictly positive so shares make sense.
            shifted = team_perf - team_perf.min() + 0.1
            shares[cursor : cursor + n] = shifted / shifted.mean()
            cursor += n

        return MatchResult(
            lobby=lobby,
            extracted=extracted,
            kill_credits=kill_credits,
            expected_extract=expected_extract.astype(np.float32),
            team_strength=strengths,
            winning_team=-1,
            contributions={
                "player_perf": flat_perf,
                "share": shares,
            },
        )
