"""Pairwise extract Elo updater.

For each pair of teams (A, B) in a lobby, run a classic Elo update:
  actual = 1 if A extracted and B died
         = 0 if A died and B extracted
         = 0.5 otherwise
  expected = 1 / (1 + 10^((rating_b - rating_a) / ELO_DIVISOR))
  delta_pair = k_factor * (actual - expected)

Each team's total delta is the sum of pairwise deltas over its opponents,
divided by the number of opponents (so k stays calibrated regardless of
team count). Split equally across the team's players, and scaled by
1/ELO_DIVISOR so observed_skill stays in the same range as true_skill
(~[-3, 3]) rather than the classical Elo scale (0-3000).
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population

# Classical Elo's logistic denominator: a 400-point rating gap implies
# 10:1 win odds. We also use 1/ELO_DIVISOR as a display scale so ratings
# live in true_skill units rather than Elo units.
ELO_DIVISOR = 400.0


class ExtractEloUpdater:
    def __init__(self, cfg: RatingUpdaterConfig) -> None:
        self.cfg = cfg

    def update(self, result: MatchResult, pop: Population) -> None:
        if result.extracted is None:
            raise ValueError(
                "ExtractEloUpdater requires a MatchResult with extracted set "
                "(use the extraction outcome generator)"
            )
        n_teams = len(result.lobby.teams)
        if n_teams < 2:
            return

        team_arrays = [np.array(t, dtype=np.int32) for t in result.lobby.teams]
        team_ratings = np.array(
            [float(pop.observed_skill[arr].mean()) for arr in team_arrays]
        )
        extracted = result.extracted.astype(bool)

        total_pair_delta = np.zeros(n_teams, dtype=np.float64)
        for a in range(n_teams):
            for b in range(n_teams):
                if a == b:
                    continue
                if extracted[a] and not extracted[b]:
                    actual = 1.0
                elif not extracted[a] and extracted[b]:
                    actual = 0.0
                else:
                    actual = 0.5
                expected_a = 1.0 / (
                    1.0
                    + 10.0 ** ((team_ratings[b] - team_ratings[a]) / ELO_DIVISOR)
                )
                total_pair_delta[a] += self.cfg.k_factor * (actual - expected_a)

        # Average over opponents so k_factor stays calibrated.
        n_opponents = n_teams - 1
        team_delta = total_pair_delta / n_opponents

        # Display scale: keep observed_skill in true_skill units.
        per_player_scale = 1.0 / ELO_DIVISOR
        for team_idx, arr in enumerate(team_arrays):
            per_player = team_delta[team_idx] * per_player_scale / len(arr)
            pop.observed_skill[arr] = (
                pop.observed_skill[arr] + np.float32(per_player)
            ).astype(np.float32)
