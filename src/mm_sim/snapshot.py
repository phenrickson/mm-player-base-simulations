"""Daily snapshot of population metrics -> polars DataFrame.

This is the main output of a simulation run: one row per day with
aggregate metrics we care about for answering the research questions.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from mm_sim.population import Population


class DailySnapshotWriter:
    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(
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
            self._rows.append(
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
            self._rows.append(
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

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(self._rows)
