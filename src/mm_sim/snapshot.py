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

    def record_aggregate(
        self,
        day: int,
        pop: Population,
        matches_today: int,
        blowouts_today: int,
    ) -> None:
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
                "observed_skill": pop.observed_skill.copy(),
                "experience": pop.experience.copy(),
                "gear": pop.gear.copy(),
                "active": pop.active.copy(),
                "party_id": pop.party_id.copy(),
                "matches_played": pop.matches_played.copy(),
                "recent_wins": pop.recent_wins.astype(np.int16),
                "recent_losses": pop.recent_losses.astype(np.int16),
                "recent_blowout_losses": pop.recent_blowout_losses.astype(np.int16),
                "join_day": pop.join_day.copy(),
            }
        )
        self._pop_frames.append(df)

    def aggregate_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(self._agg_rows)

    def population_dataframe(self) -> pl.DataFrame | None:
        if not self._pop_frames:
            return None
        return pl.concat(self._pop_frames)

    # Back-compat alias used by older callers during refactor
    def to_dataframe(self) -> pl.DataFrame:
        return self.aggregate_dataframe()
