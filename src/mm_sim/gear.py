"""Gear: grows with matches, drops on blowout losses, clipped to [0, max]."""

from __future__ import annotations

import numpy as np

from mm_sim.config import GearConfig
from mm_sim.population import Population


def apply_gear_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    blowout_losses_this_tick: np.ndarray,
    cfg: GearConfig,
) -> None:
    growth = matches_played_this_tick.astype(np.float32) * cfg.growth_per_match
    drop = blowout_losses_this_tick.astype(np.float32) * cfg.drop_on_blowout_loss
    pop.gear = np.clip(
        pop.gear + growth - drop, 0.0, cfg.max_gear
    ).astype(np.float32)
