"""Integration test: verify the Activision feedback loop can emerge.

Runs a named experiment (`feedback_loop_test`) so the artifact remains on
disk after the test and can be loaded with `load_experiment(...)` for
further inspection. Asserts only coarse sanity invariants; the real
analysis happens by looking at the saved population snapshots.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from mm_sim.config import (
    ChurnConfig,
    FrequencyConfig,
    MatchmakerConfig,
    PartyConfig,
    PopulationConfig,
    SimulationConfig,
)
from mm_sim.experiments import ExperimentRunner, load_experiment


def test_feedback_loop_experiment_runs_and_produces_artifacts(tmp_path: Path):
    """Runs a 30-day season with skill-based MM and churn sensitive to
    blowout losses, then verifies the resulting artifact is well-formed
    and contains the data needed to investigate the feedback loop."""
    cfg = SimulationConfig(
        seed=1999,
        season_days=30,
        population=PopulationConfig(
            initial_size=5000,
            true_skill_distribution="normal",
            daily_new_player_fraction=0.01,
        ),
        parties=PartyConfig(size_distribution={1: 1.0}),  # solo-only for clarity
        matchmaker=MatchmakerConfig(
            kind="composite",
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
        ),
        churn=ChurnConfig(
            baseline_daily_quit_prob=0.002,
            blowout_loss_weight=0.15,
            win_streak_weight=-0.02,
            rolling_window=5,
        ),
        frequency=FrequencyConfig(mean_matches_per_day=4.0),
    )
    runner = ExperimentRunner(experiments_dir=tmp_path)
    exp = runner.run(cfg, name="feedback_loop_test")

    # Structural sanity: day 0 pristine + 30 ticked days = 31 rows
    assert exp.aggregate.height == 31
    assert exp.population is not None
    assert exp.population.height > 0
    assert {"day", "player_id", "true_skill", "active"}.issubset(
        exp.population.columns
    )

    # The experiment should have matched some games and churned some players
    assert exp.aggregate["matches_played"].sum() > 0
    first_day_active = exp.aggregate["active_count"][0]
    last_day_active = exp.aggregate["active_count"][-1]
    assert last_day_active > 0  # population didn't collapse entirely

    # Sanity: rating error drops as Elo learns (monotonic-ish)
    first_week_error = exp.aggregate.filter(pl.col("day") < 7)[
        "rating_error_mean"
    ].mean()
    last_week_error = exp.aggregate.filter(pl.col("day") >= 23)[
        "rating_error_mean"
    ].mean()
    assert last_week_error < first_week_error, (
        f"rating error should decrease; first_week={first_week_error:.3f}, "
        f"last_week={last_week_error:.3f}"
    )

    # Confirm round-trip load works
    reloaded = load_experiment("feedback_loop_test", experiments_dir=tmp_path)
    assert reloaded.metadata.name == "feedback_loop_test"
    assert reloaded.aggregate.equals(exp.aggregate)
