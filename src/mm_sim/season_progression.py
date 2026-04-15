"""Season progression: per-player [0,1] progress, earned by playing matches.

Churn pressure has two sides:
  - behind: gap = expected(day) - progress > 0 adds to quit prob.
  - boredom: if day/season_days < boredom_cutoff and progress > expected(day),
    the gap magnitude (weighted by boredom_weight) adds to quit prob.

No-op when disabled.
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import SeasonProgressionConfig
from mm_sim.population import Population


def expected_progress(
    day: int, season_days: int, cfg: SeasonProgressionConfig
) -> float:
    if season_days <= 0:
        return 0.0
    frac = day / float(season_days)
    return float(1.0 - np.exp(-cfg.curve_steepness * frac))


def apply_season_progression_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    cfg: SeasonProgressionConfig,
) -> None:
    if not cfg.enabled:
        return
    earned = matches_played_this_tick.astype(np.float32) * cfg.earn_per_match
    pop.season_progress = np.clip(
        pop.season_progress + earned, 0.0, 1.0
    ).astype(np.float32)


def season_churn_pressure(
    progress: np.ndarray,
    day: int,
    season_days: int,
    cfg: SeasonProgressionConfig,
) -> np.ndarray:
    """Return additional quit probability per player from season-progress gap.

    Returns zeros when disabled or when weights are zero.
    """
    if not cfg.enabled:
        return np.zeros(progress.shape, dtype=np.float32)

    expected = expected_progress(day, season_days, cfg)
    gap = expected - progress.astype(np.float32)

    behind = np.clip(gap, 0.0, None) * cfg.behind_weight

    season_frac = day / float(season_days) if season_days > 0 else 1.0
    if season_frac < cfg.boredom_cutoff:
        ahead = np.clip(-gap, 0.0, None) * cfg.boredom_weight
    else:
        ahead = np.zeros_like(behind)

    return (behind + ahead).astype(np.float32)
