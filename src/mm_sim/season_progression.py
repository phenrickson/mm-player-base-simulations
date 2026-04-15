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


def apply_extraction_season_progression(
    pop: "Population",
    result: "MatchResult",
    cfg: SeasonProgressionConfig,
    *,
    mean_matches_per_day: float,
    season_days: int,
    expected_extract_rate: float = 0.4,
    expected_kills_per_extract: float = 0.75,
) -> None:
    """Outcome-weighted season progress update for one extraction match.

    Per-team earn:
      raw = base_per_match * (
          participation_weight
        + extraction_weight * (1 if extracted else 0) / expected_extract_rate
        + kill_weight * kills / (expected_extract_rate * expected_kills_per_extract)
      )

    Then scaled by concavity:
      gain = raw * (1 - current_progress) ** concavity
    """
    from mm_sim.outcomes.base import MatchResult  # noqa: F401

    if not cfg.enabled:
        return
    if result.extracted is None:
        raise ValueError("extraction season progression requires extracted array")

    expected_matches = max(mean_matches_per_day * season_days, 1.0)
    base_per_match = cfg.base_earn_per_season / expected_matches

    n_teams = len(result.lobby.teams)
    kills_per_team = np.zeros(n_teams, dtype=np.int32)
    for killer_idx, _ in result.kill_credits:
        kills_per_team[killer_idx] += 1

    kill_denom = max(
        expected_extract_rate * expected_kills_per_extract, 1e-6
    )

    for team_idx, team in enumerate(result.lobby.teams):
        extracted = 1.0 if bool(result.extracted[team_idx]) else 0.0
        kills = float(kills_per_team[team_idx])
        raw = base_per_match * (
            cfg.participation_weight
            + cfg.extraction_weight * extracted / max(expected_extract_rate, 1e-6)
            + cfg.kill_weight * kills / kill_denom
        )
        if raw <= 0:
            continue
        for pid in team:
            current = float(pop.season_progress[pid])
            scale = max(1.0 - current, 0.0) ** cfg.concavity
            pop.season_progress[pid] = np.float32(
                min(1.0, current + raw * scale)
            )
