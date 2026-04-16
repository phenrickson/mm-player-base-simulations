"""Daily snapshots of population state.

Two kinds of snapshots are written:
- Aggregate metrics per day (small, always written): one row per day with
  active_count, skill percentiles, etc.
- Full per-player state per day (larger, optional): one row per
  (day, player_id) tuple with every field on Population.

Both return polars DataFrames; the experiment writer stores them as parquet.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mm_sim.population import Population


class DailySnapshotWriter:
    """Writes both aggregate and (optional) full per-player daily snapshots."""

    def __init__(self) -> None:
        self._agg_rows: list[dict] = []
        self._pop_frames: list[pl.DataFrame] = []
        self._match_rows: list[dict] = []
        self._match_team_rows: list[dict] = []

    def record_aggregate(
        self,
        day: int,
        pop: Population,
        matches_today: int,
        blowouts_today: int,
    ) -> None:
        # Match-quality daily aggregates from any matches recorded this day.
        day_matches = [m for m in self._match_rows if m["day"] == day]
        if day_matches:
            lobby_range = np.array([m["lobby_range"] for m in day_matches])
            lobby_std_arr = np.array([m["lobby_std"] for m in day_matches])
            team_gap_arr = np.array([m["team_gap"] for m in day_matches])
            win_prob_dev_arr = np.array([m["win_prob_dev"] for m in day_matches])
            fee_arr = np.array(
                [m.get("favorite_expected_extract", float("nan")) for m in day_matches]
            )

            def _nan_safe_stats(arr: np.ndarray) -> tuple[float, float, float]:
                valid = arr[~np.isnan(arr)]
                if valid.size == 0:
                    return float("nan"), float("nan"), float("nan")
                return (
                    float(valid.mean()),
                    float(np.percentile(valid, 50)),
                    float(np.percentile(valid, 90)),
                )

            _wpd_mean, _wpd_p50, _wpd_p90 = _nan_safe_stats(win_prob_dev_arr)
            _fee_mean, _fee_p50, _fee_p90 = _nan_safe_stats(fee_arr)
            mq = {
                "lobby_range_mean": float(lobby_range.mean()),
                "lobby_range_p50": float(np.percentile(lobby_range, 50)),
                "lobby_range_p90": float(np.percentile(lobby_range, 90)),
                "lobby_std_mean": float(lobby_std_arr.mean()),
                "team_gap_mean": float(team_gap_arr.mean()),
                "team_gap_p50": float(np.percentile(team_gap_arr, 50)),
                "team_gap_p90": float(np.percentile(team_gap_arr, 90)),
                "win_prob_dev_mean": _wpd_mean,
                "win_prob_dev_p50": _wpd_p50,
                "win_prob_dev_p90": _wpd_p90,
                "favorite_expected_extract_mean": _fee_mean,
                "favorite_expected_extract_p50": _fee_p50,
                "favorite_expected_extract_p90": _fee_p90,
            }
        else:
            mq = {
                "lobby_range_mean": 0.0,
                "lobby_range_p50": 0.0,
                "lobby_range_p90": 0.0,
                "lobby_std_mean": 0.0,
                "team_gap_mean": 0.0,
                "team_gap_p50": 0.0,
                "team_gap_p90": 0.0,
                "win_prob_dev_mean": 0.0,
                "win_prob_dev_p50": 0.0,
                "win_prob_dev_p90": 0.0,
                "favorite_expected_extract_mean": float("nan"),
                "favorite_expected_extract_p50": float("nan"),
                "favorite_expected_extract_p90": float("nan"),
            }

        active_mask = pop.active
        active_count = int(active_mask.sum())
        if active_count > 0:
            ts = pop.true_skill[active_mask]
            obs = pop.observed_skill[active_mask]
            exp = pop.experience[active_mask]
            gear = pop.gear[active_mask]
            self._agg_rows.append(
                {
                    "day": day,
                    "active_count": active_count,
                    "matches_played": matches_today,
                    "blowouts": blowouts_today,
                    "true_skill_mean": float(ts.mean()),
                    "true_skill_p10": float(np.percentile(ts, 10)),
                    "true_skill_p50": float(np.percentile(ts, 50)),
                    "true_skill_p90": float(np.percentile(ts, 90)),
                    "observed_skill_mean": float(obs.mean()),
                    "rating_error_mean": float(np.abs(obs - ts).mean()),
                    "experience_mean": float(exp.mean()),
                    "gear_mean": float(gear.mean()),
                    **mq,
                }
            )
        else:
            self._agg_rows.append(
                {
                    "day": day,
                    "active_count": 0,
                    "matches_played": matches_today,
                    "blowouts": blowouts_today,
                    "true_skill_mean": 0.0,
                    "true_skill_p10": 0.0,
                    "true_skill_p50": 0.0,
                    "true_skill_p90": 0.0,
                    "observed_skill_mean": 0.0,
                    "rating_error_mean": 0.0,
                    "experience_mean": 0.0,
                    "gear_mean": 0.0,
                    **mq,
                }
            )

    def record_match(
        self,
        day: int,
        match_idx: int,
        lobby_true_skills: np.ndarray,  # flat array of true_skill for every player in the lobby
        team_true_skills: list[np.ndarray],  # one array per team
        is_blowout: bool,
        winning_team: int,
        favorite_expected_extract: float | None = None,
    ) -> None:
        """Append one row describing a single match's quality metrics.

        `favorite_expected_extract` is the extraction-mode analog of
        `win_prob_dev`: the pre-noise extract probability of the strongest
        team in the lobby. None/NaN in 2-team mode.
        """
        lobby_min = float(lobby_true_skills.min())
        lobby_max = float(lobby_true_skills.max())
        lobby_std = float(lobby_true_skills.std())
        lobby_range = lobby_max - lobby_min

        team_means = np.array([float(t.mean()) for t in team_true_skills])
        team_gap = float(team_means.max() - team_means.min())

        if len(team_means) == 2:
            r_a, r_b = team_means[0], team_means[1]
            expected_a = 1.0 / (1.0 + 10.0 ** (r_b - r_a))
            win_prob_dev = abs(expected_a - 0.5)
        else:
            win_prob_dev = float("nan")

        self._match_rows.append(
            {
                "day": day,
                "match_idx": match_idx,
                "lobby_range": lobby_range,
                "lobby_std": lobby_std,
                "team_gap": team_gap,
                "win_prob_dev": win_prob_dev,
                "favorite_expected_extract": (
                    float(favorite_expected_extract)
                    if favorite_expected_extract is not None
                    else float("nan")
                ),
                "is_blowout": bool(is_blowout),
                "winning_team": int(winning_team),
            }
        )

    def record_population(self, day: int, pop: Population) -> None:
        """Append a full per-player snapshot for this day (all players, incl.
        churned, so departed-player analysis is possible).

        Every mutable array is copied: polars stores numpy arrays by
        reference, so without copies later in-place mutations would
        retroactively change earlier snapshots.
        """
        n = pop.size
        df = pl.DataFrame(
            {
                "day": np.full(n, day, dtype=np.int32),
                "player_id": np.arange(n, dtype=np.int32),
                "true_skill": pop.true_skill.copy(),
                "talent_ceiling": pop.talent_ceiling.copy(),
                "observed_skill": pop.observed_skill.copy(),
                "experience": pop.experience.copy(),
                "gear": pop.gear.copy(),
                "active": pop.active.copy(),
                "party_id": pop.party_id.copy(),
                "matches_played": pop.matches_played.copy(),
                "recent_wins": pop.recent_wins.astype(np.int16),
                "recent_losses": pop.recent_losses.astype(np.int16),
                "recent_blowout_losses": pop.recent_blowout_losses.astype(np.int16),
                "loss_streak": pop.loss_streak.copy(),
                "join_day": pop.join_day.copy(),
                "season_progress": pop.season_progress.copy(),
            }
        )
        self._pop_frames.append(df)

    def aggregate_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(self._agg_rows)

    def population_dataframe(self) -> pl.DataFrame | None:
        if not self._pop_frames:
            return None
        return pl.concat(self._pop_frames)

    def match_dataframe(self) -> pl.DataFrame | None:
        if not self._match_rows:
            return None
        return pl.DataFrame(self._match_rows)

    def record_match_team_detail(
        self,
        day: int,
        match_idx: int,
        team_idx: int,
        player_ids: list[int],
        mean_true_skill_before: float,
        mean_observed_skill_before: float,
        mean_gear_before: float,
        team_strength: float,
        expected_extract: float,
        extracted: bool,
        kills: int,
        killed_by_team: int,  # -1 if none
    ) -> None:
        """One row per team per match. Captures what the team looked like
        going into the match and what happened."""
        self._match_team_rows.append(
            {
                "day": day,
                "match_idx": match_idx,
                "team_idx": team_idx,
                "player_ids": list(player_ids),
                "mean_true_skill_before": float(mean_true_skill_before),
                "mean_observed_skill_before": float(mean_observed_skill_before),
                "mean_gear_before": float(mean_gear_before),
                "team_strength": float(team_strength),
                "expected_extract": float(expected_extract),
                "extracted": bool(extracted),
                "kills": int(kills),
                "killed_by_team": int(killed_by_team),
            }
        )

    def match_team_dataframe(self) -> pl.DataFrame | None:
        if not self._match_team_rows:
            return None
        return pl.DataFrame(self._match_team_rows)

    # Back-compat alias used by older callers during refactor
    def to_dataframe(self) -> pl.DataFrame:
        return self.aggregate_dataframe()
