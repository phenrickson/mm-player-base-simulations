"""Pairwise extract Elo updater.

For each pair of teams (A, B) in a lobby, run a classic Elo update:
  actual = 1 if A extracted and B died
         = 0 if A died and B extracted
         = 0.5 otherwise
  expected = 1 / (1 + 10^((rating_b - rating_a) / ELO_SCALE))
  delta_pair = k_factor * (actual - expected)

Each team's total delta is the sum of pairwise deltas over its opponents,
divided by the number of opponents (so k stays calibrated regardless of
team count), then split equally across the team's players.

ELO_SCALE = 1.0 puts observed_skill in the same units as true_skill:
a 1-unit rating gap implies 10:1 win odds. k_factor is used directly
(no 1/400 rescaling), so its numerical value controls max per-match
rating movement in observed_skill units. Typical values: 0.05-0.2.
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population

# Rating gap that implies 10:1 win odds. Chosen to keep observed_skill
# in the same unit space as true_skill (~[-3, 3]) rather than classical
# Elo units (0-3000). Both the expected-score formula and the display
# range use this value, so the self-regulation dynamics are preserved.
ELO_SCALE = 1.0


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
                    + 10.0 ** ((team_ratings[b] - team_ratings[a]) / ELO_SCALE)
                )
                total_pair_delta[a] += self.cfg.k_factor * (actual - expected_a)

        # Average over opponents so k_factor stays calibrated.
        n_opponents = n_teams - 1
        team_delta = total_pair_delta / n_opponents

        for team_idx, arr in enumerate(team_arrays):
            per_player = team_delta[team_idx] / len(arr)
            pop.observed_skill[arr] = (
                pop.observed_skill[arr] + np.float32(per_player)
            ).astype(np.float32)
