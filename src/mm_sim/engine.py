"""SimulationEngine: the daily-tick orchestrator.

The engine is deliberately thin. All logic lives in the individual
components (matchmaker, outcome generator, rating updater, churn, etc.).
This module just wires them together and runs the loop.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from mm_sim.churn import apply_churn
from mm_sim.config import SimulationConfig
from mm_sim.experience import apply_experience_update
from mm_sim.frequency import sample_matches_per_day
from mm_sim.gear import apply_gear_update
from mm_sim.matchmaker.base import Matchmaker
from mm_sim.matchmaker.composite_mm import CompositeRatingMatchmaker
from mm_sim.matchmaker.random_mm import RandomMatchmaker
from mm_sim.outcomes.base import OutcomeGenerator
from mm_sim.outcomes.default import DefaultOutcomeGenerator
from mm_sim.parties import assign_parties
from mm_sim.population import Population
from mm_sim.rating_updaters.base import RatingUpdater
from mm_sim.rating_updaters.elo import EloUpdater
from mm_sim.rating_updaters.kpm import KPMUpdater
from mm_sim.seeding import make_rng, spawn_child
from mm_sim.snapshot import DailySnapshotWriter


def _make_matchmaker(cfg: SimulationConfig) -> Matchmaker:
    kind = cfg.matchmaker.kind
    if kind == "random":
        return RandomMatchmaker(cfg.matchmaker)
    if kind == "composite":
        return CompositeRatingMatchmaker(cfg.matchmaker)
    raise ValueError(f"unknown matchmaker kind: {kind}")


def _make_outcome_generator(cfg: SimulationConfig) -> OutcomeGenerator:
    if cfg.outcomes.kind == "default":
        return DefaultOutcomeGenerator(cfg.outcomes)
    raise ValueError(f"unknown outcome kind: {cfg.outcomes.kind}")


def _make_rating_updater(cfg: SimulationConfig) -> RatingUpdater:
    kind = cfg.rating_updater.kind
    if kind == "elo":
        return EloUpdater(cfg.rating_updater)
    if kind == "kpm":
        return KPMUpdater(cfg.rating_updater)
    raise ValueError(f"unknown rating updater kind: {kind}")


class SimulationEngine:
    def __init__(
        self, cfg: SimulationConfig, *, progress_label: str | None = None
    ) -> None:
        self.cfg = cfg
        self.progress_label = progress_label
        self.master_rng = make_rng(cfg.seed)
        self.population = Population.create_initial(
            cfg.population, spawn_child(self.master_rng, "population_init")
        )
        assign_parties(
            self.population,
            cfg.parties,
            spawn_child(self.master_rng, "parties_init"),
        )
        self.matchmaker = _make_matchmaker(cfg)
        self.outcome_generator = _make_outcome_generator(cfg)
        self.rating_updater = _make_rating_updater(cfg)
        self.snapshot_writer = DailySnapshotWriter()
        # Frozen at startup: arrivals per day keyed off the initial cohort
        # size, not the current (bleeding) population.
        self.daily_new_players = int(
            round(cfg.population.initial_size * cfg.population.daily_new_player_fraction)
        )

    def run(self) -> pl.DataFrame:
        """Run the full season. Returns the aggregate daily snapshot.

        A pristine day-0 snapshot (pre-matches) is recorded first so the
        initial observed_skill state is preserved. Then day 1..season_days
        are the simulated ticks.
        """
        self.snapshot_writer.record_aggregate(
            day=0, pop=self.population, matches_today=0, blowouts_today=0
        )
        self.snapshot_writer.record_population(day=0, pop=self.population)
        total = self.cfg.season_days
        for day in range(1, total + 1):
            self._tick(day)
            self._emit_progress(day, total)
        self._finish_progress()
        return self.snapshot_writer.aggregate_dataframe()

    def _emit_progress(self, day: int, total: int) -> None:
        if self.progress_label is None or not sys.stdout.isatty():
            return
        pct = int(day * 100 / total)
        sys.stdout.write(
            f"\r  {self.progress_label}: day {day}/{total} ({pct}%)"
        )
        sys.stdout.flush()

    def _finish_progress(self) -> None:
        if self.progress_label is None or not sys.stdout.isatty():
            return
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _tick(self, day: int) -> None:
        day_rng = spawn_child(self.master_rng, f"day_{day}")

        matches_per_player = sample_matches_per_day(
            self.population, self.cfg.frequency, spawn_child(day_rng, "frequency")
        )

        total_matches = np.zeros_like(matches_per_player)
        total_wins = np.zeros_like(matches_per_player)
        total_losses = np.zeros_like(matches_per_player)
        total_blowout_losses = np.zeros_like(matches_per_player)

        matches_today = 0
        blowouts_today = 0
        day_match_idx = 0

        max_rounds = int(matches_per_player.max(initial=0))
        for round_idx in range(max_rounds):
            remaining = matches_per_player - total_matches
            still_playing = (remaining > 0) & self.population.active
            searching = np.flatnonzero(still_playing).astype(np.int32)
            if len(searching) < self.cfg.matchmaker.lobby_size:
                break

            round_rng = spawn_child(day_rng, f"round_{round_idx}")
            lobbies = self.matchmaker.form_lobbies(
                searching, self.population, round_rng
            )

            for lobby_idx, lobby in enumerate(lobbies):
                result = self.outcome_generator.generate(
                    lobby,
                    self.population,
                    spawn_child(round_rng, f"lobby_{lobby_idx}"),
                )
                self.rating_updater.update(result, self.population)

                matches_today += 1
                if result.is_blowout:
                    blowouts_today += 1

                flat_ids = result.flat_player_ids()
                total_matches[flat_ids] += 1

                # Per-match quality metrics based on true_skill.
                lobby_true = self.population.true_skill[flat_ids]
                team_trues = [
                    self.population.true_skill[np.array(team, dtype=np.int32)]
                    for team in lobby.teams
                ]
                self.snapshot_writer.record_match(
                    day=day,
                    match_idx=day_match_idx,
                    lobby_true_skills=lobby_true,
                    team_true_skills=team_trues,
                    is_blowout=bool(result.is_blowout),
                    winning_team=int(result.winning_team),
                )
                day_match_idx += 1

                winning_team_ids = np.array(
                    lobby.teams[result.winning_team], dtype=np.int32
                )
                total_wins[winning_team_ids] += 1

                # RESET streak on win
                self.population.loss_streak[winning_team_ids] = 0

                for team_idx, team in enumerate(lobby.teams):
                    if team_idx == result.winning_team:
                        continue
                    losing_team_ids = np.array(team, dtype=np.int32)
                    total_losses[losing_team_ids] += 1
                    self.population.loss_streak[losing_team_ids] += 1
                    if result.is_blowout:
                        total_blowout_losses[losing_team_ids] += 1

        # Update rolling windows (last-tick values)
        window = self.cfg.churn.rolling_window
        self.population.recent_wins = np.clip(
            total_wins, 0, window
        ).astype(np.int8)
        self.population.recent_losses = np.clip(
            total_losses, 0, window
        ).astype(np.int8)
        self.population.recent_blowout_losses = np.clip(
            total_blowout_losses, 0, window
        ).astype(np.int8)

        apply_experience_update(
            self.population,
            total_matches,
            normalization_max_matches=max(self.cfg.season_days * 5, 1),
        )
        apply_gear_update(
            self.population,
            total_matches,
            total_blowout_losses,
            self.cfg.gear,
        )

        apply_churn(
            self.population,
            self.cfg.churn,
            spawn_child(day_rng, "churn"),
        )

        # New players arrive (assigned as solo parties for simplicity)
        new_ids = self.population.add_new_players(
            self.daily_new_players,
            self.cfg.population,
            spawn_child(day_rng, "new_players"),
            day=day,
        )
        if len(new_ids) > 0:
            next_pid = (
                int(self.population.party_id.max()) + 1
                if self.population.size > 0
                else 0
            )
            for offset, nid in enumerate(new_ids):
                self.population.party_id[nid] = next_pid + offset

        self.snapshot_writer.record_aggregate(
            day=day,
            pop=self.population,
            matches_today=matches_today,
            blowouts_today=blowouts_today,
        )
        if day % self.cfg.population_snapshot_every_n_days == 0:
            self.snapshot_writer.record_population(day=day, pop=self.population)
