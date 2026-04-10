"""Random matchmaker: party-aware, no skill consideration."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mm_sim.config import MatchmakerConfig
from mm_sim.matchmaker.base import Lobby, group_by_party, pack_parties_into_lobbies
from mm_sim.population import Population


class RandomMatchmaker:
    def __init__(self, cfg: MatchmakerConfig) -> None:
        self.cfg = cfg

    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]:
        parties = group_by_party(searching_player_ids, pop)
        rng.shuffle(parties)
        return pack_parties_into_lobbies(parties, self.cfg)
