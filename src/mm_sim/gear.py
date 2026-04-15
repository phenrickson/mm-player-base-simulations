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
    # Legacy drop only applies when transfer is disabled. When transfer is on,
    # blowout gear effects are handled per-match inside the transfer function.
    if cfg.transfer_enabled:
        drop = np.zeros_like(growth)
    else:
        drop = blowout_losses_this_tick.astype(np.float32) * cfg.drop_on_blowout_loss
    pop.gear = np.clip(
        pop.gear + growth - drop, 0.0, cfg.max_gear
    ).astype(np.float32)


def apply_gear_transfer_for_match(
    pop: Population,
    winners: np.ndarray,
    losers: np.ndarray,
    is_blowout: bool,
    cfg: GearConfig,
) -> None:
    """Transfer a fraction of each loser's gear to the winners (split equally).

    No-op when transfer_enabled is False. Blowouts use the higher rate.
    """
    if not cfg.transfer_enabled:
        return
    if len(winners) == 0 or len(losers) == 0:
        return

    rate = cfg.transfer_rate_blowout if is_blowout else cfg.transfer_rate
    if rate <= 0.0:
        return

    loser_gear = pop.gear[losers]
    loss = (loser_gear * rate).astype(np.float32)
    total_transferred = float(loss.sum())

    pop.gear[losers] = np.clip(loser_gear - loss, 0.0, cfg.max_gear).astype(np.float32)
    per_winner = total_transferred / float(len(winners))
    pop.gear[winners] = np.clip(
        pop.gear[winners] + per_winner, 0.0, cfg.max_gear
    ).astype(np.float32)
