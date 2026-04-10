"""Matchmaker protocol, Lobby dataclass, and shared party-packing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from mm_sim.config import MatchmakerConfig
from mm_sim.population import Population


@dataclass
class Lobby:
    teams: list[list[int]]  # each team is a list of player ids


class Matchmaker(Protocol):
    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]: ...


def group_by_party(
    player_ids: np.ndarray, pop: Population
) -> list[list[int]]:
    """Group a list of searching player ids into their parties."""
    party_to_members: dict[int, list[int]] = {}
    for pid in player_ids:
        p = int(pop.party_id[pid])
        party_to_members.setdefault(p, []).append(int(pid))
    return list(party_to_members.values())


def pack_parties_into_lobbies(
    parties: list[list[int]], cfg: MatchmakerConfig
) -> list[Lobby]:
    """Walk parties in given order and fill lobbies.

    Caller controls ordering (random for RandomMatchmaker, sorted by rating
    for CompositeRatingMatchmaker). Within a lobby, snake-assign parties to
    teams so both sides get a representative slice of the window.

    Partial lobbies (fewer than lobby_size players left) are dropped.
    """
    lobby_size = cfg.lobby_size
    teams_per_lobby = cfg.teams_per_lobby
    team_capacity = lobby_size // teams_per_lobby

    lobbies: list[Lobby] = []
    available = [True] * len(parties)
    cursor = 0
    n = len(parties)

    while cursor < n:
        # Pull parties from cursor forward into a pool of exactly lobby_size
        # players. Skip any party that would overflow.
        pool_indices: list[int] = []
        pool_player_count = 0
        scan = cursor
        while scan < n and pool_player_count < lobby_size:
            if available[scan]:
                party = parties[scan]
                if pool_player_count + len(party) <= lobby_size:
                    pool_indices.append(scan)
                    pool_player_count += len(party)
            scan += 1

        if pool_player_count < lobby_size:
            break

        for idx in pool_indices:
            available[idx] = False
        while cursor < n and not available[cursor]:
            cursor += 1

        # Snake-assign to teams
        teams: list[list[int]] = [[] for _ in range(teams_per_lobby)]
        team_cursor = 0
        direction = 1
        failed = False
        for idx in pool_indices:
            party = parties[idx]
            placed = False
            for _ in range(teams_per_lobby):
                if len(teams[team_cursor]) + len(party) <= team_capacity:
                    teams[team_cursor].extend(party)
                    placed = True
                    team_cursor += direction
                    if team_cursor >= teams_per_lobby:
                        direction = -1
                        team_cursor = teams_per_lobby - 1
                    elif team_cursor < 0:
                        direction = 1
                        team_cursor = 0
                    break
                team_cursor += direction
                if team_cursor >= teams_per_lobby:
                    direction = -1
                    team_cursor = teams_per_lobby - 1
                elif team_cursor < 0:
                    direction = 1
                    team_cursor = 0
            if not placed:
                failed = True
                break

        if not failed and sum(len(t) for t in teams) == lobby_size:
            lobbies.append(Lobby(teams=teams))

    return lobbies
