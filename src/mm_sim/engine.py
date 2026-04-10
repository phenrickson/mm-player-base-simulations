"""SimulationEngine: the daily-tick orchestrator.

The engine is deliberately thin. All logic lives in the individual
components (matchmaker, outcome generator, rating updater, churn, etc.).
This module just wires them together and runs the loop.
"""

from __future__ import annotations

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
    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
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

    def run(self) -> pl.DataFrame:
        for day in range(self.cfg.season_days):
            self._tick(day)
        return self.snapshot_writer.to_dataframe()

    def _tick(self, day: int) -> None:
        day_rng = spawn_child(self.master_rng, f"day_{day}")

        matches_per_player = sample_matches_per_day(
            self.population, self.cfg.frequency, spawn_child(day_rng, "frequency")
        )

        total_matches = np.zeros_like(matches_per_player)
        total_wins = np.zeros_like(matches_per_player)
        total_blowout_losses = np.zeros_like(matches_per_player)

        matches_today = 0
        blowouts_today = 0

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

                winning_team_ids = np.array(
                    lobby.teams[result.winning_team], dtype=np.int32
                )
                total_wins[winning_team_ids] += 1
                if result.is_blowout:
                    for team_idx, team in enumerate(lobby.teams):
                        if team_idx != result.winning_team:
                            total_blowout_losses[
                                np.array(team, dtype=np.int32)
                            ] += 1

        # Update rolling windows (last-tick values)
        window = self.cfg.churn.rolling_window
        self.population.recent_wins = np.clip(
            total_wins, 0, window
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
            self.cfg.population.daily_new_players,
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

        self.snapshot_writer.record(
            day=day,
            pop=self.population,
            matches_today=matches_today,
            blowouts_today=blowouts_today,
        )
