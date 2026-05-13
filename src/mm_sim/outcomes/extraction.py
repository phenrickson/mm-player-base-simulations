"""Extraction outcome generator: softmax-over-strength winner sampling."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.outcomes.softmax_winners import (
    plackett_luce_marginals,
    sample_extractor_count,
)
from mm_sim.population import Population


class ExtractionOutcomeGenerator:
    """Each match draws k (number of extractors) from a mixture, then samples
    k winners without replacement from ``softmax(beta * strengths)``.

    ``expected_extract[i]`` is the Plackett-Luce marginal probability that
    team i is among the top-k under that same softmax — the quantity the
    calibration chart compares against the rating-updater's Elo view.
    """

    def __init__(self, cfg: OutcomeConfig) -> None:
        self.cfg = cfg

    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult:
        n_teams = len(lobby.teams)
        strengths = np.zeros(n_teams, dtype=np.float32)
        player_perfs: list[np.ndarray] = []
        for i, team in enumerate(lobby.teams):
            arr = np.array(team, dtype=np.int32)
            s = pop.true_skill[arr].astype(np.float32)
            if self.cfg.gear_weight > 0:
                s = s + self.cfg.gear_weight * pop.gear[arr].astype(np.float32)
            strengths[i] = s.mean()
            perf_noise = rng.normal(
                0.0, self.cfg.noise_std, size=len(arr)
            ).astype(np.float32)
            player_perfs.append(s + perf_noise)

        k = sample_extractor_count(
            n_teams=n_teams,
            mean_extractors=self.cfg.mean_extractors_per_match,
            p_zero=self.cfg.p_zero_extract,
            p_all=self.cfg.p_all_extract,
            rng=rng,
        )
        beta = self.cfg.strength_sensitivity
        extracted = np.zeros(n_teams, dtype=bool)
        if k >= n_teams:
            extracted[:] = True
        elif k > 0:
            shifted = beta * strengths.astype(np.float64)
            shifted = shifted - shifted.max()
            w = np.exp(shifted)
            remaining = np.ones(n_teams, dtype=bool)
            for _ in range(k):
                pool = np.flatnonzero(remaining)
                probs = w[pool] / w[pool].sum()
                pick = int(rng.choice(pool, p=probs))
                extracted[pick] = True
                remaining[pick] = False

        expected_extract = plackett_luce_marginals(
            strengths=strengths, k=k, beta=beta
        ).astype(np.float32)

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

        flat_perf = np.concatenate(player_perfs).astype(np.float32)
        shares = np.zeros_like(flat_perf)
        cursor = 0
        for i, team in enumerate(lobby.teams):
            n = len(team)
            team_perf = flat_perf[cursor : cursor + n]
            shifted = team_perf - team_perf.min() + 0.1
            shares[cursor : cursor + n] = shifted / shifted.mean()
            cursor += n

        return MatchResult(
            lobby=lobby,
            extracted=extracted,
            kill_credits=kill_credits,
            expected_extract=expected_extract,
            team_strength=strengths,
            winning_team=-1,
            contributions={
                "player_perf": flat_perf,
                "share": shares,
            },
        )
