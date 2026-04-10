"""Static party assignment at population creation.

Parties are assigned once, at the start of the season, with two knobs:
- size distribution (how many solo vs duo vs trio parties)
- skill homogeneity (0 = random friends, 1 = parties of identical skill)
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import PartyConfig
from mm_sim.population import Population


def assign_parties(
    pop: Population, cfg: PartyConfig, rng: np.random.Generator
) -> None:
    n = pop.size
    sizes = list(cfg.size_distribution.keys())
    probs = list(cfg.size_distribution.values())

    # Draw party sizes until we've placed everyone.
    party_sizes: list[int] = []
    remaining = n
    while remaining > 0:
        s = int(rng.choice(sizes, p=probs))
        s = min(s, remaining)
        party_sizes.append(s)
        remaining -= s

    # Two candidate orderings: by true_skill (homogeneous) and shuffled (random).
    sorted_idx = np.argsort(pop.true_skill)
    shuffled_idx = sorted_idx.copy()
    rng.shuffle(shuffled_idx)

    # Blend them: each player slot picks from sorted with prob = homogeneity,
    # otherwise from shuffled. Deduplicate while preserving order.
    h = cfg.skill_homogeneity
    blend = np.where(rng.random(n) < h, sorted_idx, shuffled_idx)
    _, first_positions = np.unique(blend, return_index=True)
    order = blend[np.sort(first_positions)]
    missing = np.setdiff1d(np.arange(n), order, assume_unique=False)
    order = np.concatenate([order, missing])

    next_pid = 0
    cursor = 0
    for s in party_sizes:
        group = order[cursor : cursor + s]
        pop.party_id[group] = next_pid
        next_pid += 1
        cursor += s
