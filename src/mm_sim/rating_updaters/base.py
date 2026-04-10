"""Rating updater protocol."""

from __future__ import annotations

from typing import Protocol

from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class RatingUpdater(Protocol):
    def update(self, result: MatchResult, pop: Population) -> None: ...
