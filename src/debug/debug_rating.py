"""Tiny reproducer: run ~20 matches with 24 players, print everything.

Goal: see why observed_skill has ~0 correlation with true_skill.
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import (
    GearConfig,
    MatchmakerConfig,
    OutcomeConfig,
    PartyConfig,
    PopulationConfig,
    RatingUpdaterConfig,
    SeasonProgressionConfig,
    SimulationConfig,
    SkillProgressionConfig,
    StageConfig,
)
from mm_sim.matchmaker.two_stage import TwoStageMatchmaker
from mm_sim.outcomes.extraction import ExtractionOutcomeGenerator
from mm_sim.population import Population
from mm_sim.rating_updaters.elo_extract import ExtractEloUpdater


def main() -> None:
    N = 240  # 240 players -> 20 lobbies of 12 each round
    pop_cfg = PopulationConfig(
        initial_size=N,
        true_skill_distribution="normal",
        true_skill_std=1.0,
    )
    pop = Population.create_initial(pop_cfg, np.random.default_rng(42))
    # All solos:
    pop.party_id[:] = np.arange(N, dtype=np.int32)

    mm_cfg = MatchmakerConfig(
        kind="two_stage",
        lobby_size=12,
        teams_per_lobby=4,
        team_formation=StageConfig(
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
            sort_jitter=0.2,
        ),
        lobby_assembly=StageConfig(
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
            sort_jitter=0.2,
        ),
    )
    oc_cfg = OutcomeConfig(
        kind="extraction",
        gear_weight=0.0,  # ignore gear for now — focus on skill signal
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
        strength_sensitivity=2.0,
    )
    ru_cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=0.1)

    mm = TwoStageMatchmaker(mm_cfg)
    oc = ExtractionOutcomeGenerator(oc_cfg)
    ru = ExtractEloUpdater(ru_cfg)

    rng = np.random.default_rng(1)

    def print_table() -> None:
        order = np.argsort(-pop.true_skill)
        print(f"{'pid':>4} {'true':>7} {'obs':>7} {'diff':>7}")
        for pid in order:
            print(
                f"{int(pid):>4} "
                f"{float(pop.true_skill[pid]):>7.3f} "
                f"{float(pop.observed_skill[pid]):>7.3f} "
                f"{float(pop.observed_skill[pid] - pop.true_skill[pid]):>+7.3f}"
            )

    print("=== INITIAL ===")
    print_table()
    print()

    # Print first 5 matches in full detail, then corr every 50 rounds.
    verbose_rounds = {0, 1, 2}

    def k_rescaled() -> float:
        return ru_cfg.k_factor / 400.0

    total_rounds = 1000
    for round_idx in range(total_rounds):
        searching = np.arange(N, dtype=np.int32)
        lobbies = mm.form_lobbies(searching, pop, np.random.default_rng(round_idx))
        for lobby_idx, lobby in enumerate(lobbies):
            result = oc.generate(
                lobby, pop, np.random.default_rng(round_idx * 100 + lobby_idx)
            )
            # Snapshot BEFORE update
            obs_before = pop.observed_skill.copy()
            ru.update(result, pop)
            obs_after = pop.observed_skill

            if round_idx in verbose_rounds:
                print(f"\n--- round {round_idx} lobby {lobby_idx} ---")
                share = result.contributions.get("share")
                cursor = 0
                for t_idx, team in enumerate(lobby.teams):
                    n = len(team)
                    arr = np.array(team, dtype=np.int32)
                    extracted = bool(result.extracted[t_idx])
                    expected = float(result.expected_extract[t_idx])
                    team_delta = k_rescaled() * (
                        (1.0 if extracted else 0.0) - expected
                    )
                    print(
                        f"  team {t_idx}: extracted={extracted} "
                        f"E[extract]={expected:.2f} team_delta={team_delta:+.5f}"
                    )
                    print(f"    {'pid':>4} {'true':>7} {'share':>6} {'delta':>8}")
                    for j, pid in enumerate(team):
                        s = float(share[cursor + j]) if share is not None else 1.0
                        d = float(obs_after[pid] - obs_before[pid])
                        print(
                            f"    {int(pid):>4} {float(pop.true_skill[pid]):>+7.3f} "
                            f"{s:>6.3f} {d:>+8.5f}"
                        )
                    cursor += n

        if (round_idx + 1) % 50 == 0:
            corr = float(np.corrcoef(pop.true_skill, pop.observed_skill)[0, 1])
            print(
                f"[round {round_idx+1}] corr={corr:+.4f} "
                f"obs_std={float(pop.observed_skill.std()):.4f}"
            )

    print("\n=== FINAL TABLE ===")
    print_table()


if __name__ == "__main__":
    main()
