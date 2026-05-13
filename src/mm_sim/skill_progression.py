"""True-skill progression: per-match drift toward talent_ceiling.

Discrete analogue of dx/dn = (ceiling - x) / tau, which integrates to
x(n) = ceiling - (ceiling - x0) * exp(-n / tau). Applied per tick so noise
accumulates naturally. No-op when disabled.
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import SkillProgressionConfig
from mm_sim.population import Population


def apply_skill_progression_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    cfg: SkillProgressionConfig,
    rng: np.random.Generator,
) -> None:
    if not cfg.enabled:
        return

    matches = matches_played_this_tick.astype(np.float32)
    gap = np.maximum(pop.talent_ceiling - pop.true_skill, 0.0).astype(np.float32)
    deterministic = gap * (matches / cfg.tau)

    if cfg.noise_std > 0.0:
        noise = rng.normal(
            loc=0.0,
            scale=cfg.noise_std * np.sqrt(np.maximum(matches, 0.0)),
            size=pop.size,
        ).astype(np.float32)
    else:
        noise = np.zeros(pop.size, dtype=np.float32)

    played_mask = matches > 0
    eligible_mask = played_mask & (pop.true_skill < pop.talent_ceiling)
    delta = np.where(eligible_mask, deterministic + noise, 0.0).astype(np.float32)
    new_skill = pop.true_skill + delta
    clipped = np.minimum(new_skill, pop.talent_ceiling).astype(np.float32)
    pop.true_skill = np.where(eligible_mask, clipped, pop.true_skill).astype(np.float32)
