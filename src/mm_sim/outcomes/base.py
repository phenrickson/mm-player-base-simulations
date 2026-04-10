"""Outcome generator protocol and MatchResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mm_sim.matchmaker.base import Lobby
from mm_sim.population import Population


@dataclass
class MatchResult:
    lobby: Lobby
    winning_team: int
    score_margin: float
    is_blowout: bool
    # contributions: dict of field name -> array indexed by position in the
    # flattened list of players (team 0 first, then team 1...).
    contributions: dict[str, np.ndarray]

    def flat_player_ids(self) -> np.ndarray:
        return np.array(
            [pid for team in self.lobby.teams for pid in team], dtype=np.int32
        )


class OutcomeGenerator(Protocol):
    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult: ...
