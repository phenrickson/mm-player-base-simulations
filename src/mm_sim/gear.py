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


def apply_extraction_gear_update(
    pop: Population,
    result: "MatchResult",
    cfg: GearConfig,
) -> None:
    """Gear update for a single extraction match.

    - If no team extracts: skip entirely.
    - Each extracting player gains `extract_growth`.
    - For each kill credit (killer, victim):
        delta = victim_strength - killer_strength
        rate  = transfer_rate * max(punching_down_floor, 1 + strength_bonus*delta)
      Victim players lose `their_gear * rate`; pool of stripped gear is
      multiplied by `transfer_efficiency` and split equally among killer
      team members (capped at max_gear).
    """
    from mm_sim.outcomes.base import MatchResult  # noqa: F401

    if result.extracted is None or result.team_strength is None:
        raise ValueError("extraction gear update requires extracted + team_strength")

    if not result.extracted.any():
        return

    # 1. Extract growth for extractors.
    for team_idx, team in enumerate(result.lobby.teams):
        if not bool(result.extracted[team_idx]):
            continue
        for pid in team:
            pop.gear[pid] = min(
                cfg.max_gear, float(pop.gear[pid]) + cfg.extract_growth
            )

    # 2. Kill-credited transfers.
    for killer_idx, victim_idx in result.kill_credits:
        killer_team = result.lobby.teams[killer_idx]
        victim_team = result.lobby.teams[victim_idx]
        delta = float(
            result.team_strength[victim_idx] - result.team_strength[killer_idx]
        )
        scale = max(
            cfg.punching_down_floor, 1.0 + cfg.strength_bonus * delta
        )
        rate = cfg.transfer_rate * scale
        if rate <= 0.0:
            continue

        pool = 0.0
        for vid in victim_team:
            loss = float(pop.gear[vid]) * rate
            pop.gear[vid] = max(0.0, float(pop.gear[vid]) - loss)
            pool += loss

        gain_per_killer = (pool * cfg.transfer_efficiency) / len(killer_team)
        for kid in killer_team:
            pop.gear[kid] = min(
                cfg.max_gear, float(pop.gear[kid]) + gain_per_killer
            )
