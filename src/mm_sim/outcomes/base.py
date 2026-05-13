"""Outcome generator protocol and MatchResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mm_sim.matchmaker.base import Lobby
from mm_sim.population import Population


@dataclass
class MatchResult:
    lobby: Lobby
    # --- Legacy 2-team fields (set for outcomes.kind == "default") ---
    winning_team: int = -1
    score_margin: float = 0.0
    is_blowout: bool = False
    contributions: dict[str, np.ndarray] = field(default_factory=dict)
    # --- Extraction fields (set for outcomes.kind == "extraction") ---
    # extracted[team_idx] -> True if that team extracted this match.
    extracted: np.ndarray | None = None
    # kill_credits: list of (killer_team_idx, victim_team_idx) tuples.
    kill_credits: list[tuple[int, int]] = field(default_factory=list)
    # expected_extract[team_idx] -> pre-noise extract probability (for Elo).
    expected_extract: np.ndarray | None = None
    # team_strength[team_idx] -> mean strength used in rolls (for gear transfer).
    team_strength: np.ndarray | None = None

    def flat_player_ids(self) -> np.ndarray:
        return np.array(
            [pid for team in self.lobby.teams for pid in team], dtype=np.int32
        )


class OutcomeGenerator(Protocol):
    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult: ...
