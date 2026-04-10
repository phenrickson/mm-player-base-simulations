"""Composite rating matchmaker: skill/experience/gear weighted composite.

This is the core research tool. Different `composite_weights` let you
compare pure-skill matchmaking vs. level-based vs. anything in between.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mm_sim.config import MatchmakerConfig
from mm_sim.matchmaker.base import Lobby, group_by_party, pack_parties_into_lobbies
from mm_sim.population import Population


def compute_composite_rating(
    pop: Population, weights: dict[str, float]
) -> np.ndarray:
    """Per-player composite rating: weighted sum of normalized components.

    Skill is converted to a population percentile so it lives in [0, 1]
    alongside experience and gear.
    """
    active_mask = pop.active
    skill_component = np.zeros(pop.size, dtype=np.float32)
    if active_mask.any():
        active_obs = pop.observed_skill[active_mask]
        ranks = active_obs.argsort().argsort().astype(np.float32)
        percentiles = ranks / max(len(active_obs) - 1, 1)
        skill_component[active_mask] = percentiles
    return (
        weights.get("skill", 0.0) * skill_component
        + weights.get("experience", 0.0) * pop.experience
        + weights.get("gear", 0.0) * pop.gear
    )


class CompositeRatingMatchmaker:
    def __init__(self, cfg: MatchmakerConfig) -> None:
        self.cfg = cfg

    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]:
        rating = compute_composite_rating(pop, self.cfg.composite_weights)
        parties = group_by_party(searching_player_ids, pop)
        party_ratings = [float(rating[members].mean()) for members in parties]
        order = np.argsort(party_ratings)
        parties_sorted = [parties[i] for i in order]
        return pack_parties_into_lobbies(parties_sorted, self.cfg)
