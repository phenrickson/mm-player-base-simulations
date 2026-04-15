"""Churn: daily per-player quit probability driven by recent experience.

Quit probability is a combination of four signals:

    baseline
    + loss_weight * recent_loss_rate^2       (squared — streaks hurt)
    + blowout_loss_weight * recent_blowout_rate
    + win_streak_weight * recent_win_rate    (negative = wins make you stick)

The loss term is squared so a 50% loss rate barely registers (0.25 of
the linear value) but a 100% loss rate feels catastrophic. The two
loss-driven terms are scaled by a new-player sensitivity multiplier so
players who have played very few matches get hit harder by losses than
veterans. Skill itself is never an input — churn emerges from what
matches the player got and how they did in them.
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import ChurnConfig, SeasonProgressionConfig
from mm_sim.population import Population
from mm_sim.season_progression import season_churn_pressure


def apply_churn(
    pop: Population,
    cfg: ChurnConfig,
    rng: np.random.Generator,
    *,
    day: int = 0,
    season_days: int = 1,
    season_cfg: SeasonProgressionConfig | None = None,
) -> None:
    window = float(cfg.rolling_window)
    loss_rate = pop.recent_losses.astype(np.float32) / window
    loss_streak = pop.loss_streak.astype(np.float32)
    blowout_rate = pop.recent_blowout_losses.astype(np.float32) / window
    win_rate = pop.recent_wins.astype(np.float32) / window

    # New-player sensitivity: 1 + bonus * max(0, 1 - matches/threshold)
    # Veterans (matches >= threshold) get a 1x multiplier.
    threshold = float(cfg.new_player_threshold)
    newness = np.clip(
        1.0 - pop.matches_played.astype(np.float32) / threshold, 0.0, 1.0
    )
    sensitivity = (1.0 + cfg.new_player_bonus * newness).astype(np.float32)

    loss_streak_multiplier = np.exp(cfg.loss_streak_exp * loss_streak) - 1.0
    loss_streak_multiplier = np.clip(
        loss_streak_multiplier, 0.0, cfg.max_loss_streak_multiplier
    ).astype(np.float32)
    loss_streak_factor = 1.0 + loss_streak_multiplier

    loss_term = cfg.loss_weight * (loss_rate ** 2) * loss_streak_factor
    blowout_term = cfg.blowout_loss_weight * blowout_rate

    if season_cfg is not None:
        season_term = season_churn_pressure(
            pop.season_progress, day=day, season_days=season_days, cfg=season_cfg
        )
    else:
        season_term = np.zeros(pop.size, dtype=np.float32)

    quit_prob = np.clip(
        cfg.baseline_daily_quit_prob
        + sensitivity * (loss_term + blowout_term)
        + cfg.win_streak_weight * win_rate
        + season_term,
        0.0,
        cfg.max_daily_quit_prob,
    ).astype(np.float32)

    draws = rng.random(size=pop.size).astype(np.float32)
    quits = (draws < quit_prob) & pop.active
    pop.active[quits] = False
