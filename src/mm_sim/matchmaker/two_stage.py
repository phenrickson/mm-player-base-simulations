"""Two-stage matchmaker: form teams of 3, then assemble lobbies of 4 teams."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mm_sim.config import MatchmakerConfig, StageConfig
from mm_sim.matchmaker.base import Lobby, group_by_party
from mm_sim.matchmaker.composite_mm import compute_composite_rating
from mm_sim.population import Population


class TwoStageMatchmaker:
    """Team formation + lobby assembly with independent policies per stage."""

    def __init__(self, cfg: MatchmakerConfig) -> None:
        self.cfg = cfg
        if cfg.lobby_size % cfg.teams_per_lobby != 0:
            raise ValueError(
                "lobby_size must be divisible by teams_per_lobby"
            )
        self.team_size = cfg.lobby_size // cfg.teams_per_lobby

    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]:
        parties = group_by_party(searching_player_ids, pop)
        teams = self._form_teams(parties, pop, self.cfg.team_formation, rng)
        return self._assemble_lobbies(teams, pop, self.cfg.lobby_assembly, rng)

    def _form_teams(
        self,
        parties: list[list[int]],
        pop: Population,
        stage_cfg: StageConfig,
        rng: np.random.Generator,
    ) -> list[list[int]]:
        """Combine solos/duos into teams of exactly team_size; trios pass through."""
        rating = compute_composite_rating(pop, stage_cfg.composite_weights)

        full_teams: list[list[int]] = []
        partial: list[list[int]] = []
        for party in parties:
            if len(party) == self.team_size:
                full_teams.append(list(party))
            elif len(party) < self.team_size:
                partial.append(list(party))
            # Parties larger than team_size are invalid per design.

        keys = [float(np.mean(rating[p])) for p in partial]
        if stage_cfg.sort_jitter > 0 and len(partial) > 0:
            noise = rng.normal(0.0, stage_cfg.sort_jitter, size=len(partial))
            keys = [k + float(noise[i]) for i, k in enumerate(keys)]
        partial_sorted = [p for _, p in sorted(zip(keys, partial), key=lambda kp: kp[0])]

        i = 0
        while i < len(partial_sorted):
            team = list(partial_sorted[i])
            i += 1
            while len(team) < self.team_size and i < len(partial_sorted):
                candidate = partial_sorted[i]
                if len(team) + len(candidate) <= self.team_size:
                    team.extend(candidate)
                    i += 1
                else:
                    i += 1
            if len(team) == self.team_size:
                full_teams.append(team)

        return full_teams

    def _assemble_lobbies(
        self,
        teams: list[list[int]],
        pop: Population,
        stage_cfg: StageConfig,
        rng: np.random.Generator,
    ) -> list[Lobby]:
        if len(teams) < self.cfg.teams_per_lobby:
            return []

        rating = compute_composite_rating(pop, stage_cfg.composite_weights)
        team_keys = np.array([float(rating[t].mean()) for t in teams])
        if stage_cfg.sort_jitter > 0:
            team_keys = team_keys + rng.normal(
                0.0, stage_cfg.sort_jitter, size=len(team_keys)
            )
        order = np.argsort(team_keys)
        teams_sorted = [teams[int(i)] for i in order]

        lobbies: list[Lobby] = []
        step = self.cfg.teams_per_lobby
        for i in range(0, len(teams_sorted) - step + 1, step):
            lobbies.append(Lobby(teams=teams_sorted[i : i + step]))
        return lobbies
