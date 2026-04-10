"""Churn: daily per-player quit probability driven by recent experience.

Making churn a function of recent experience (rather than a flat rate) is
what lets the Activision feedback loop emerge in the simulation.
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import ChurnConfig
from mm_sim.population import Population


def apply_churn(
    pop: Population, cfg: ChurnConfig, rng: np.random.Generator
) -> None:
    window = float(cfg.rolling_window)
    blowout_rate = pop.recent_blowout_losses.astype(np.float32) / window
    win_rate = pop.recent_wins.astype(np.float32) / window

    quit_prob = np.clip(
        cfg.baseline_daily_quit_prob
        + cfg.blowout_loss_weight * blowout_rate
        + cfg.win_streak_weight * win_rate,
        0.0,
        cfg.max_daily_quit_prob,
    ).astype(np.float32)

    draws = rng.random(size=pop.size).astype(np.float32)
    quits = (draws < quit_prob) & pop.active
    pop.active[quits] = False
