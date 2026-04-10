"""Per-player matches-per-day sampling, modulated by recent experience."""

from __future__ import annotations

import numpy as np

from mm_sim.config import FrequencyConfig
from mm_sim.population import Population


def sample_matches_per_day(
    pop: Population, cfg: FrequencyConfig, rng: np.random.Generator
) -> np.ndarray:
    """Return an int32 array: how many matches each player plays today.

    Winners play more, losers play less — a second feedback loop on top
    of churn. Inactive players get zero.
    """
    window = max(
        int(pop.recent_wins.max(initial=0)),
        int(pop.recent_blowout_losses.max(initial=0)),
        1,
    )
    win_rate = pop.recent_wins.astype(np.float32) / float(window)
    loss_rate = pop.recent_blowout_losses.astype(np.float32) / float(window)
    multiplier = np.clip(
        1.0 + cfg.win_modulation * win_rate - cfg.loss_modulation * loss_rate,
        0.0,
        None,
    )
    lam = cfg.mean_matches_per_day * multiplier
    draws = rng.poisson(lam).astype(np.int32)
    draws[~pop.active] = 0
    return draws
