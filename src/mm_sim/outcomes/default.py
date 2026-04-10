"""Default outcome generator: true_skill + noise drives match performance."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class DefaultOutcomeGenerator:
    def __init__(self, cfg: OutcomeConfig) -> None:
        self.cfg = cfg

    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult:
        team_performances: list[float] = []
        per_player_perf: list[np.ndarray] = []
        for team in lobby.teams:
            team_arr = np.array(team, dtype=np.int32)
            base = pop.true_skill[team_arr].astype(np.float32)
            noise = rng.normal(
                0.0, self.cfg.noise_std, size=len(team_arr)
            ).astype(np.float32)
            player_perf = base + noise
            per_player_perf.append(player_perf)
            team_performances.append(float(player_perf.sum()))

        winning_team = int(np.argmax(team_performances))
        best = team_performances[winning_team]
        worst = min(team_performances)
        score_margin = (best - worst) * 5.0
        is_blowout = score_margin >= self.cfg.blowout_threshold

        flat_perf = np.concatenate(per_player_perf)
        # Shift so everything is positive for the kill/death breakdown
        flat_perf_pos = np.clip(flat_perf - flat_perf.min() + 0.1, 0.1, None)
        total = flat_perf_pos.sum()
        scale = max(score_margin + 40.0, 40.0)
        half_n = len(flat_perf_pos) / 2

        kills = np.floor(flat_perf_pos / total * scale * half_n).astype(np.float32)
        inv = 1.0 / flat_perf_pos
        deaths = np.floor(inv / inv.sum() * scale * half_n).astype(np.float32)
        damage = kills * float(rng.uniform(80.0, 120.0))
        objective_score = kills * 100.0 + rng.uniform(
            0.0, 50.0, size=len(kills)
        ).astype(np.float32)

        return MatchResult(
            lobby=lobby,
            winning_team=winning_team,
            score_margin=float(score_margin),
            is_blowout=bool(is_blowout),
            contributions={
                "kills": kills,
                "deaths": deaths,
                "damage": damage,
                "objective_score": objective_score,
            },
        )
