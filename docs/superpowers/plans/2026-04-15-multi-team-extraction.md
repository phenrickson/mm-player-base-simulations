# Multi-Team Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-team extraction match mode (4 teams × 3 players, independent per-team extract/die outcomes, directed kill attribution, strength-responsive gear transfer with decay, season-length-invariant outcome-weighted progression, expected-vs-actual extract Elo) while keeping the existing 2-team mode working.

**Architecture:** New `extraction` outcome generator, new `two_stage` matchmaker, new `elo_extract` rating updater, extended `gear` and `season_progression` modules. The engine's match loop is refactored to handle N teams per match with kill credits. Legacy 2-team paths stay intact — scenario config selects behavior via `outcomes.kind`, `matchmaker.kind`, `rating_updater.kind`.

**Tech Stack:** Python 3.12, numpy, polars, pydantic v2 (for config), pytest.

See [design spec](../specs/2026-04-15-multi-team-extraction-design.md).

---

## File Structure

**New files:**

- `src/mm_sim/outcomes/extraction.py` — extract-roll outcome generator
- `src/mm_sim/matchmaker/two_stage.py` — team formation + lobby assembly
- `src/mm_sim/rating_updaters/elo_extract.py` — expected-vs-actual extract Elo
- `tests/test_extraction_outcomes.py`
- `tests/test_two_stage_matchmaker.py`
- `tests/test_elo_extract.py`
- `tests/test_gear_extraction.py`
- `tests/test_season_progression_extraction.py`

**Modified files:**

- `src/mm_sim/config.py` — add/extend `OutcomeConfig`, `MatchmakerConfig`, `GearConfig`, `SeasonProgressionConfig`, `RatingUpdaterConfig`
- `src/mm_sim/outcomes/base.py` — extend `MatchResult` with `extracted` array and `kill_credits`
- `src/mm_sim/gear.py` — add `apply_gear_update_for_extraction_match`
- `src/mm_sim/season_progression.py` — add outcome-weighted earn with concavity
- `src/mm_sim/engine.py` — dispatch to extraction path when `outcomes.kind == "extraction"`
- `scenarios/defaults.toml` — opt in to extraction mode
- `tests/test_config_progression.py` — bump defaults
- `tests/test_gear_transfer.py` — ensure legacy still passes

---

## Task 1: Extend config for extraction mode (pydantic)

**Files:**
- Modify: `src/mm_sim/config.py`
- Modify: `tests/test_config_progression.py`

- [ ] **Step 1: Write failing tests for the new config fields**

Append to `tests/test_config_progression.py`:

```python
def test_outcome_config_extraction_defaults():
    from mm_sim.config import OutcomeConfig

    cfg = OutcomeConfig(kind="extraction")
    assert cfg.kind == "extraction"
    assert cfg.baseline_extract_prob == 0.4
    assert cfg.strength_sensitivity == 1.0


def test_matchmaker_config_two_stage_defaults():
    from mm_sim.config import MatchmakerConfig

    cfg = MatchmakerConfig()
    # Legacy single-stage fields still present:
    assert cfg.lobby_size == 12
    assert cfg.teams_per_lobby == 2
    # New two-stage substructure present:
    assert cfg.team_formation.composite_weights == {
        "skill": 1.0, "experience": 0.0, "gear": 0.0
    }
    assert cfg.team_formation.max_rating_spread == 0.3
    assert cfg.lobby_assembly.max_rating_spread == 0.3


def test_gear_config_extraction_defaults():
    from mm_sim.config import GearConfig

    cfg = GearConfig()
    assert cfg.extract_growth == 0.003
    assert cfg.strength_bonus == 1.0
    assert cfg.punching_down_floor == 0.2
    assert cfg.transfer_efficiency == 0.9


def test_season_progression_extraction_defaults():
    from mm_sim.config import SeasonProgressionConfig

    cfg = SeasonProgressionConfig()
    assert cfg.base_earn_per_season == 0.8
    assert cfg.concavity == 1.0
    assert cfg.participation_weight == 0.3
    assert cfg.extraction_weight == 0.5
    assert cfg.kill_weight == 0.2


def test_rating_updater_config_elo_extract():
    from mm_sim.config import RatingUpdaterConfig

    cfg = RatingUpdaterConfig(kind="elo_extract")
    assert cfg.kind == "elo_extract"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_config_progression.py -v
```

Expected: FAILs on the five new tests (new fields missing).

- [ ] **Step 3: Extend config.py**

Replace the relevant classes in `src/mm_sim/config.py` with:

```python
class StageConfig(BaseModel):
    """Policy for one stage of the two-stage matchmaker."""

    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {"skill": 1.0, "experience": 0.0, "gear": 0.0}
    )
    max_rating_spread: float = 0.3
    max_rating_spread_growth: float = 0.05

    @field_validator("composite_weights")
    @classmethod
    def _weights_nonnegative(cls, v: dict[str, float]) -> dict[str, float]:
        for k, val in v.items():
            if val < 0:
                raise ValueError(f"weight {k} must be >= 0, got {val}")
        return v


class MatchmakerConfig(BaseModel):
    kind: str = Field("composite", pattern="^(random|composite|two_stage)$")
    # Legacy single-stage fields (still used by random/composite paths).
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {"skill": 1.0, "experience": 0.0, "gear": 0.0}
    )
    lobby_size: int = Field(12, gt=1)
    teams_per_lobby: int = Field(2, gt=1)
    max_rating_spread: float = 0.3
    max_rating_spread_growth: float = 0.05
    # Two-stage fields (used when kind == "two_stage").
    team_formation: StageConfig = Field(default_factory=StageConfig)
    lobby_assembly: StageConfig = Field(default_factory=StageConfig)

    @field_validator("composite_weights")
    @classmethod
    def _weights_nonnegative(cls, v: dict[str, float]) -> dict[str, float]:
        for k, val in v.items():
            if val < 0:
                raise ValueError(f"weight {k} must be >= 0, got {val}")
        return v


class OutcomeConfig(BaseModel):
    kind: str = Field("default", pattern="^(default|extraction)$")
    noise_std: float = 0.25
    blowout_threshold: float = 30.0
    gear_weight: float = Field(0.0, ge=0.0)
    # Extraction-specific:
    baseline_extract_prob: float = Field(0.4, ge=0.0, le=1.0)
    strength_sensitivity: float = Field(1.0, gt=0.0)


class RatingUpdaterConfig(BaseModel):
    kind: str = Field("elo", pattern="^(elo|kpm|elo_extract)$")
    k_factor: float = 32.0


class GearConfig(BaseModel):
    growth_per_match: float = Field(0.0015, ge=0.0)
    max_gear: float = Field(1.0, gt=0.0)
    transfer_enabled: bool = False
    transfer_rate: float = Field(0.005, ge=0.0)
    transfer_rate_blowout: float = Field(0.04, ge=0.0)
    drop_on_blowout_loss: float = Field(0.05, ge=0.0)
    # Extraction-specific:
    extract_growth: float = Field(0.003, ge=0.0)
    strength_bonus: float = Field(1.0, ge=0.0)
    punching_down_floor: float = Field(0.2, ge=0.0, le=1.0)
    transfer_efficiency: float = Field(0.9, ge=0.0, le=1.0)


class SeasonProgressionConfig(BaseModel):
    enabled: bool = False
    # Legacy linear earn (used when outcomes.kind == "default"):
    earn_per_match: float = Field(0.005, ge=0.0)
    curve_steepness: float = Field(3.0, gt=0.0)
    behind_weight: float = Field(0.02, ge=0.0)
    boredom_weight: float = Field(0.01, ge=0.0)
    boredom_cutoff: float = Field(0.7, ge=0.0, le=1.0)
    # Extraction outcome-weighted earn:
    base_earn_per_season: float = Field(0.8, gt=0.0)
    concavity: float = Field(1.0, gt=0.0)
    participation_weight: float = Field(0.3, ge=0.0)
    extraction_weight: float = Field(0.5, ge=0.0)
    kill_weight: float = Field(0.2, ge=0.0)
```

Keep the other classes (`PopulationConfig`, `PartyConfig`, `ChurnConfig`, `FrequencyConfig`, `SkillProgressionConfig`, `SimulationConfig`) unchanged.

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_config_progression.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/config.py tests/test_config_progression.py
git commit -m "feat(config): extraction mode fields and two-stage matchmaker config"
```

---

## Task 2: Extend MatchResult for multi-team extraction outcomes

**Files:**
- Modify: `src/mm_sim/outcomes/base.py`
- Test: covered in Task 3

- [ ] **Step 1: Add extraction fields to MatchResult**

Replace `src/mm_sim/outcomes/base.py` with:

```python
"""Outcome generator protocol and MatchResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mm_sim.matchmaker.base import Lobby
from mm_sim.population import Population


@dataclass
class MatchResult:
    lobby: Lobby
    # --- Legacy 2-team fields (set for outcomes.kind == "default") ---
    winning_team: int = -1
    score_margin: float = 0.0
    is_blowout: bool = False
    contributions: dict[str, np.ndarray] = field(default_factory=dict)
    # --- Extraction fields (set for outcomes.kind == "extraction") ---
    # extracted[team_idx] -> True if that team extracted this match.
    extracted: np.ndarray | None = None
    # kill_credits: list of (killer_team_idx, victim_team_idx) tuples.
    kill_credits: list[tuple[int, int]] = field(default_factory=list)
    # expected_extract[team_idx] -> pre-noise extract probability (for Elo).
    expected_extract: np.ndarray | None = None
    # team_strength[team_idx] -> mean strength used in rolls (for gear transfer).
    team_strength: np.ndarray | None = None

    def flat_player_ids(self) -> np.ndarray:
        return np.array(
            [pid for team in self.lobby.teams for pid in team], dtype=np.int32
        )


class OutcomeGenerator(Protocol):
    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult: ...
```

- [ ] **Step 2: Run existing tests to confirm no regression**

```bash
uv run pytest tests/test_outcomes_and_updaters.py -v
```

Expected: all pass (legacy fields still present, just with defaults).

- [ ] **Step 3: Commit**

```bash
git add src/mm_sim/outcomes/base.py
git commit -m "refactor(outcomes): MatchResult carries extraction + legacy fields"
```

---

## Task 3: Extraction outcome generator

**Files:**
- Create: `src/mm_sim/outcomes/extraction.py`
- Create: `tests/test_extraction_outcomes.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the extraction outcome generator."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig, PopulationConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.extraction import ExtractionOutcomeGenerator
from mm_sim.population import Population


def _pop_with_skills(skills: list[float], gears: list[float] | None = None) -> Population:
    pop_cfg = PopulationConfig(initial_size=len(skills))
    pop = Population.create_initial(pop_cfg, np.random.default_rng(0))
    pop.true_skill[:] = np.array(skills, dtype=np.float32)
    if gears is not None:
        pop.gear[:] = np.array(gears, dtype=np.float32)
    return pop


def test_team_strength_includes_gear_weight():
    pop = _pop_with_skills([1.0, 0.0, -1.0, 2.0], gears=[0.5, 0.5, 0.5, 0.5])
    lobby = Lobby(teams=[[0, 1], [2, 3]])
    cfg = OutcomeConfig(kind="extraction", gear_weight=0.5, baseline_extract_prob=0.4)
    gen = ExtractionOutcomeGenerator(cfg)
    result = gen.generate(lobby, pop, np.random.default_rng(0))

    # Team 0: mean(1.0, 0.0) + 0.5*mean(0.5,0.5) = 0.5 + 0.25 = 0.75
    # Team 1: mean(-1.0, 2.0) + 0.5*0.5 = 0.5 + 0.25 = 0.75
    assert result.team_strength is not None
    np.testing.assert_allclose(result.team_strength, [0.75, 0.75], atol=1e-5)


def test_baseline_extract_prob_at_match_mean():
    """A team exactly at match_mean should extract with baseline_extract_prob."""
    skills = [0.0] * 12  # 4 teams of 3, all equal strength
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(kind="extraction", baseline_extract_prob=0.4)
    gen = ExtractionOutcomeGenerator(cfg)

    extracted_counts = np.zeros(4, dtype=int)
    trials = 2000
    for seed in range(trials):
        result = gen.generate(lobby, pop, np.random.default_rng(seed))
        extracted_counts += result.extracted.astype(int)

    rates = extracted_counts / trials
    # With all teams at same strength, expected rate for each = baseline.
    np.testing.assert_allclose(rates, [0.4] * 4, atol=0.04)


def test_stronger_team_extracts_more_often():
    skills = [2.0, 2.0, 2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(kind="extraction", baseline_extract_prob=0.4, strength_sensitivity=1.0)
    gen = ExtractionOutcomeGenerator(cfg)

    counts = np.zeros(4, dtype=int)
    for seed in range(1000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        counts += r.extracted.astype(int)

    # Strong team extracts most, weak least.
    assert counts[0] > counts[2] > counts[1]
    assert counts[0] > 700  # dominates
    assert counts[1] < 200  # rarely


def test_kill_attribution_closest_stronger_extractor():
    """Dead team is credited to the weakest extractor above them in strength."""
    # Force outcome by setting huge strength gaps and zero noise.
    skills = [3.0, 3.0, 3.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -3.0, -3.0, -3.0]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    # Sensitivity huge so noise doesn't flip outcomes.
    cfg = OutcomeConfig(
        kind="extraction", baseline_extract_prob=0.4, strength_sensitivity=10.0
    )
    gen = ExtractionOutcomeGenerator(cfg)
    result = gen.generate(lobby, pop, np.random.default_rng(42))

    # Expect teams 0 and 1 extract (strongest two), teams 2 and 3 die.
    assert bool(result.extracted[0]) is True
    assert bool(result.extracted[1]) is True
    assert bool(result.extracted[2]) is False
    assert bool(result.extracted[3]) is False

    credits = set(result.kill_credits)
    # Team 3 (weakest) is killed by team 1 (weakest extractor above 3).
    # Team 2 is killed by team 1 (weakest extractor above 2).
    assert (1, 2) in credits
    assert (1, 3) in credits


def test_no_extractors_no_kill_credits():
    """If nobody extracts, kill_credits is empty."""
    # Trick: use sensitivity so small that noise dominates, run until we see
    # a no-extractor match.
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(kind="extraction", baseline_extract_prob=0.05)  # rare extract
    gen = ExtractionOutcomeGenerator(cfg)

    for seed in range(2000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        if not r.extracted.any():
            assert r.kill_credits == []
            return
    raise AssertionError("expected at least one no-extractor match in 2000 trials")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_extraction_outcomes.py -v
```

Expected: import error, `ExtractionOutcomeGenerator` does not exist.

- [ ] **Step 3: Implement the generator**

Create `src/mm_sim/outcomes/extraction.py`:

```python
"""Extraction outcome generator: each team independently extracts or dies."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from mm_sim.config import OutcomeConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class ExtractionOutcomeGenerator:
    def __init__(self, cfg: OutcomeConfig) -> None:
        self.cfg = cfg
        # Internally derive noise std from baseline_extract_prob:
        # We want P(roll > 0 | delta=0) = baseline_extract_prob.
        # roll = strength_sensitivity * delta + N(0, sigma).
        # At delta=0, P(noise > 0) is always 0.5 regardless of sigma. So we
        # instead shift the threshold: extract iff strength_sensitivity*delta
        # + noise > threshold, where threshold is chosen so P(N(0, sigma) >
        # threshold) = baseline_extract_prob.
        # Pick sigma=1 by convention; threshold = inverse-CDF at
        # (1 - baseline_extract_prob).
        self._sigma = 1.0
        self._threshold = float(norm.ppf(1.0 - cfg.baseline_extract_prob))

    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult:
        n_teams = len(lobby.teams)
        strengths = np.zeros(n_teams, dtype=np.float32)
        for i, team in enumerate(lobby.teams):
            arr = np.array(team, dtype=np.int32)
            s = pop.true_skill[arr].astype(np.float32)
            if self.cfg.gear_weight > 0:
                s = s + self.cfg.gear_weight * pop.gear[arr].astype(np.float32)
            strengths[i] = s.mean()

        match_mean = float(strengths.mean())
        deltas = strengths - match_mean
        noise = rng.normal(0.0, self._sigma, size=n_teams).astype(np.float32)
        rolls = self.cfg.strength_sensitivity * deltas + noise
        extracted = rolls > self._threshold

        # Expected extract: P(roll > threshold | delta) under N(0, sigma) noise.
        # = 1 - Phi((threshold - sens*delta) / sigma)
        z = (self._threshold - self.cfg.strength_sensitivity * deltas) / self._sigma
        expected_extract = 1.0 - norm.cdf(z)

        # Attribute kills.
        kill_credits: list[tuple[int, int]] = []
        extractor_idxs = np.flatnonzero(extracted)
        if extractor_idxs.size > 0:
            for dead in np.flatnonzero(~extracted):
                dead_strength = strengths[dead]
                above = [
                    i for i in extractor_idxs if strengths[i] > dead_strength
                ]
                if above:
                    # Weakest extractor above the dead team takes credit.
                    killer = int(min(above, key=lambda i: strengths[i]))
                else:
                    # No stronger extractor: strongest extractor takes credit.
                    killer = int(max(extractor_idxs, key=lambda i: strengths[i]))
                kill_credits.append((killer, int(dead)))

        return MatchResult(
            lobby=lobby,
            extracted=extracted,
            kill_credits=kill_credits,
            expected_extract=expected_extract.astype(np.float32),
            team_strength=strengths,
            # Legacy fields left at defaults.
            winning_team=-1,
            contributions={},
        )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_extraction_outcomes.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/outcomes/extraction.py tests/test_extraction_outcomes.py
git commit -m "feat(outcomes): extraction outcome generator with per-team rolls + kill attribution"
```

---

## Task 4: Elo-extract rating updater

**Files:**
- Create: `src/mm_sim/rating_updaters/elo_extract.py`
- Create: `tests/test_elo_extract.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the extract-based Elo updater."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig, PopulationConfig, RatingUpdaterConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population
from mm_sim.rating_updaters.elo_extract import ExtractEloUpdater


def _pop(n: int) -> Population:
    cfg = PopulationConfig(initial_size=n)
    return Population.create_initial(cfg, np.random.default_rng(0))


def test_extracting_raises_rating_by_k_times_one_minus_expected():
    pop = _pop(3)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.7]),
        kill_credits=[],
    )
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    updater = ExtractEloUpdater(cfg)
    updater.update(result, pop)

    # delta = 32 * (1 - 0.7) / 3 = 3.2
    np.testing.assert_allclose(pop.observed_skill[:3], [3.2, 3.2, 3.2], atol=1e-5)


def test_dying_lowers_rating_by_k_times_expected():
    pop = _pop(3)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([False]),
        expected_extract=np.array([0.7]),
        kill_credits=[],
    )
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=32.0)
    updater = ExtractEloUpdater(cfg)
    updater.update(result, pop)

    # delta = 32 * (0 - 0.7) / 3 = -7.466...
    np.testing.assert_allclose(pop.observed_skill[:3], [-32 * 0.7 / 3] * 3, atol=1e-4)


def test_multi_team_lobby_each_team_updated_independently():
    pop = _pop(12)
    pop.observed_skill[:] = 0.0
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, True, False, False]),
        expected_extract=np.array([0.8, 0.5, 0.5, 0.2]),
        kill_credits=[],
    )
    cfg = RatingUpdaterConfig(kind="elo_extract", k_factor=30.0)
    updater = ExtractEloUpdater(cfg)
    updater.update(result, pop)

    np.testing.assert_allclose(pop.observed_skill[:3], [30 * 0.2 / 3] * 3, atol=1e-5)
    np.testing.assert_allclose(pop.observed_skill[3:6], [30 * 0.5 / 3] * 3, atol=1e-5)
    np.testing.assert_allclose(pop.observed_skill[6:9], [-30 * 0.5 / 3] * 3, atol=1e-5)
    np.testing.assert_allclose(pop.observed_skill[9:12], [-30 * 0.2 / 3] * 3, atol=1e-5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_elo_extract.py -v
```

Expected: import error, `ExtractEloUpdater` does not exist.

- [ ] **Step 3: Implement the updater**

Create `src/mm_sim/rating_updaters/elo_extract.py`:

```python
"""Expected-vs-actual extract Elo updater."""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class ExtractEloUpdater:
    def __init__(self, cfg: RatingUpdaterConfig) -> None:
        self.cfg = cfg

    def update(self, result: MatchResult, pop: Population) -> None:
        if result.extracted is None or result.expected_extract is None:
            raise ValueError(
                "ExtractEloUpdater requires a MatchResult with extracted + "
                "expected_extract set (use the extraction outcome generator)"
            )
        for team_idx, team in enumerate(result.lobby.teams):
            actual = 1.0 if bool(result.extracted[team_idx]) else 0.0
            expected = float(result.expected_extract[team_idx])
            team_size = len(team)
            delta = self.cfg.k_factor * (actual - expected) / team_size
            for pid in team:
                pop.observed_skill[pid] += np.float32(delta)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_elo_extract.py -v
```

Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/rating_updaters/elo_extract.py tests/test_elo_extract.py
git commit -m "feat(rating): extract-based Elo updater"
```

---

## Task 5: Gear transfer for extraction matches

**Files:**
- Modify: `src/mm_sim/gear.py`
- Create: `tests/test_gear_extraction.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for extraction gear update."""

from __future__ import annotations

import numpy as np

from mm_sim.config import GearConfig, PopulationConfig
from mm_sim.gear import apply_extraction_gear_update
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


def _pop(gears: list[float]) -> Population:
    cfg = PopulationConfig(initial_size=len(gears))
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    pop.gear[:] = np.array(gears, dtype=np.float32)
    return pop


def test_extract_growth_only_for_extractors():
    pop = _pop([0.1] * 6)  # 2 teams of 3
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.5, 0.5]),
        kill_credits=[],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = GearConfig(
        extract_growth=0.02, transfer_rate=0.0
    )  # transfer_rate=0 so only growth applies
    apply_extraction_gear_update(pop, result, cfg)

    np.testing.assert_allclose(pop.gear[:3], [0.12, 0.12, 0.12], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.1, 0.1, 0.1], atol=1e-5)


def test_killer_of_equal_strength_team_gets_floor_rate():
    pop = _pop([0.5] * 6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.5, 0.5]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 0.0]),  # delta = 0
    )
    # extract_growth=0 so we isolate transfer math.
    cfg = GearConfig(
        extract_growth=0.0,
        transfer_rate=0.1,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=0.9,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # delta = 0 -> rate = 0.1 * max(0.2, 1+0) = 0.1
    # losers' gear 0.5 each, strip 0.05 each, pool=0.15
    # efficiency 0.9 -> winners share 0.135 / 3 = 0.045 each
    np.testing.assert_allclose(pop.gear[:3], [0.545, 0.545, 0.545], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.45, 0.45, 0.45], atol=1e-5)


def test_punching_down_uses_floor():
    pop = _pop([0.5] * 6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.9, 0.1]),
        kill_credits=[(0, 1)],
        team_strength=np.array([2.0, 0.0]),  # delta = -2 (punching down)
    )
    cfg = GearConfig(
        extract_growth=0.0,
        transfer_rate=0.1,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=0.9,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # rate = 0.1 * max(0.2, 1 + 1*(-2)) = 0.1 * 0.2 = 0.02
    # losers lose 0.01 each, pool=0.03, winners +0.027/3 = 0.009 each
    np.testing.assert_allclose(pop.gear[:3], [0.509, 0.509, 0.509], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.49, 0.49, 0.49], atol=1e-5)


def test_upset_multiplies_transfer():
    pop = _pop([0.5] * 6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.1, 0.9]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 2.0]),  # delta = +2 (upset)
    )
    cfg = GearConfig(
        extract_growth=0.0,
        transfer_rate=0.05,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=0.9,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # rate = 0.05 * max(0.2, 1 + 1*2) = 0.05 * 3 = 0.15
    # losers lose 0.075 each, pool=0.225, winners +0.2025/3 = 0.0675 each
    np.testing.assert_allclose(pop.gear[:3], [0.5675, 0.5675, 0.5675], atol=1e-4)
    np.testing.assert_allclose(pop.gear[3:6], [0.425, 0.425, 0.425], atol=1e-4)


def test_no_extractors_skips_all_gear_updates():
    before = [0.5] * 6
    pop = _pop(before)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([False, False]),
        expected_extract=np.array([0.3, 0.3]),
        kill_credits=[],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = GearConfig(extract_growth=0.05, transfer_rate=0.1)
    apply_extraction_gear_update(pop, result, cfg)

    np.testing.assert_allclose(pop.gear, before, atol=1e-7)


def test_winner_cap_excess_goes_to_void():
    pop = _pop([0.95, 0.95, 0.95, 0.5, 0.5, 0.5])
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.9, 0.1]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = GearConfig(
        max_gear=1.0,
        extract_growth=0.0,
        transfer_rate=0.2,
        strength_bonus=1.0,
        punching_down_floor=0.2,
        transfer_efficiency=1.0,
    )
    apply_extraction_gear_update(pop, result, cfg)

    # Winners capped at 1.0; losers still lose 0.2 * 0.5 = 0.1 each.
    np.testing.assert_allclose(pop.gear[:3], [1.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(pop.gear[3:6], [0.4, 0.4, 0.4], atol=1e-5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_gear_extraction.py -v
```

Expected: import error, `apply_extraction_gear_update` missing.

- [ ] **Step 3: Add the implementation**

Append to `src/mm_sim/gear.py`:

```python
def apply_extraction_gear_update(
    pop: "Population",
    result: "MatchResult",
    cfg: GearConfig,
) -> None:
    """Gear update for a single extraction match.

    Rules (see design spec for full details):
      - If no team extracted, skip entirely (no growth, no transfer).
      - Every extracting player gains `extract_growth`.
      - For each (killer, victim) credit:
          delta = victim_strength - killer_strength
          rate  = transfer_rate * max(punching_down_floor, 1 + strength_bonus*delta)
        Victim players each lose `their_gear * rate`. Total stripped is
        multiplied by `transfer_efficiency` and split equally among killer's
        players (capped at max_gear; excess -> void).
    """
    from mm_sim.outcomes.base import MatchResult  # noqa: F401 — for runtime import order

    if result.extracted is None or result.team_strength is None:
        raise ValueError("extraction gear update requires extracted + team_strength")

    if not result.extracted.any():
        return

    # 1. Extract growth.
    for team_idx, team in enumerate(result.lobby.teams):
        if not bool(result.extracted[team_idx]):
            continue
        for pid in team:
            pop.gear[pid] = min(
                cfg.max_gear, float(pop.gear[pid]) + cfg.extract_growth
            )

    # 2. Kill-credited transfers.
    for killer_idx, victim_idx in result.kill_credits:
        killer_team = result.lobby.teams[killer_idx]
        victim_team = result.lobby.teams[victim_idx]
        delta = float(
            result.team_strength[victim_idx] - result.team_strength[killer_idx]
        )
        scale = max(
            cfg.punching_down_floor, 1.0 + cfg.strength_bonus * delta
        )
        rate = cfg.transfer_rate * scale
        if rate <= 0.0:
            continue

        pool = 0.0
        for vid in victim_team:
            loss = float(pop.gear[vid]) * rate
            pop.gear[vid] = max(0.0, float(pop.gear[vid]) - loss)
            pool += loss

        gain_per_killer = (pool * cfg.transfer_efficiency) / len(killer_team)
        for kid in killer_team:
            pop.gear[kid] = min(
                cfg.max_gear, float(pop.gear[kid]) + gain_per_killer
            )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_gear_extraction.py tests/test_gear_transfer.py -v
```

Expected: all pass (legacy 2-team tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/gear.py tests/test_gear_extraction.py
git commit -m "feat(gear): extraction-mode gear update (extract_growth + directed transfer)"
```

---

## Task 6: Outcome-weighted season progression

**Files:**
- Modify: `src/mm_sim/season_progression.py`
- Create: `tests/test_season_progression_extraction.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for extraction-mode season progression."""

from __future__ import annotations

import numpy as np

from mm_sim.config import PopulationConfig, SeasonProgressionConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population
from mm_sim.season_progression import apply_extraction_season_progression


def _pop(n: int) -> Population:
    cfg = PopulationConfig(initial_size=n)
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    pop.season_progress[:] = 0.0
    return pop


def test_base_per_match_derived_from_season_target():
    pop = _pop(3)
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.4]),
        kill_credits=[],
        team_strength=np.array([0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=1.0,
        extraction_weight=0.0,
        kill_weight=0.0,
    )
    # With participation_weight=1 and the others 0, each participating match
    # should add base_earn_per_season / expected_matches.
    apply_extraction_season_progression(
        pop,
        result,
        cfg,
        mean_matches_per_day=5.0,
        season_days=90,
    )

    expected_per_match = 0.9 / (5.0 * 90)
    np.testing.assert_allclose(
        pop.season_progress[:3],
        [expected_per_match] * 3,
        atol=1e-6,
    )


def test_extraction_weight_gates_on_extract():
    pop = _pop(6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.4, 0.4]),
        kill_credits=[(0, 1)],
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=0.0,
        extraction_weight=1.0,
        kill_weight=0.0,
    )
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )

    # Extractors gain; dead team does not.
    assert (pop.season_progress[:3] > 0).all()
    np.testing.assert_allclose(pop.season_progress[3:6], [0.0] * 3, atol=1e-7)


def test_concavity_diminishes_near_cap():
    pop = _pop(3)
    pop.season_progress[:] = 0.9  # already near cap
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.4]),
        kill_credits=[],
        team_strength=np.array([0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=1.0,
        extraction_weight=0.0,
        kill_weight=0.0,
    )
    before = pop.season_progress[:3].copy()
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )
    gain = pop.season_progress[:3] - before

    # Concavity 1 with progress=0.9 scales earn by (1 - 0.9) = 0.1
    base_per_match = 0.9 / (5.0 * 90)
    expected_gain = base_per_match * 0.1
    np.testing.assert_allclose(gain, [expected_gain] * 3, atol=1e-6)


def test_disabled_is_noop():
    pop = _pop(3)
    lobby = Lobby(teams=[[0, 1, 2]])
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True]),
        expected_extract=np.array([0.4]),
        kill_credits=[],
        team_strength=np.array([0.0]),
    )
    cfg = SeasonProgressionConfig(enabled=False)
    before = pop.season_progress[:3].copy()
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )
    np.testing.assert_array_equal(pop.season_progress[:3], before)


def test_kill_weight_scales_earn_by_kill_count():
    pop = _pop(6)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5]])
    # Team 0 extracts and gets 2 kills (over some hypothetical victims)
    result = MatchResult(
        lobby=lobby,
        extracted=np.array([True, False]),
        expected_extract=np.array([0.4, 0.4]),
        kill_credits=[(0, 1), (0, 1)],  # 2 credits attributed to team 0
        team_strength=np.array([0.0, 0.0]),
    )
    cfg = SeasonProgressionConfig(
        enabled=True,
        base_earn_per_season=0.9,
        concavity=1.0,
        participation_weight=0.0,
        extraction_weight=0.0,
        kill_weight=1.0,
    )
    apply_extraction_season_progression(
        pop, result, cfg, mean_matches_per_day=5.0, season_days=90
    )
    # Team 0 gets normalized kill earn for 2 kills.
    assert (pop.season_progress[:3] > 0).all()
    # Team 1 got no kills.
    np.testing.assert_allclose(pop.season_progress[3:6], [0.0] * 3, atol=1e-7)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_season_progression_extraction.py -v
```

Expected: import error, `apply_extraction_season_progression` missing.

- [ ] **Step 3: Add the implementation**

Append to `src/mm_sim/season_progression.py`:

```python
def apply_extraction_season_progression(
    pop: "Population",
    result: "MatchResult",
    cfg: SeasonProgressionConfig,
    *,
    mean_matches_per_day: float,
    season_days: int,
    # Reference values used to normalize outcome weights. Hardcoded for v1;
    # tunable later if lobby shape changes.
    expected_extract_rate: float = 0.4,
    expected_kills_per_extract: float = 0.75,
) -> None:
    """Outcome-weighted season progress update for a single extraction match.

    Per-player earn:
      raw = base_per_match * (
          participation_weight
        + extraction_weight * (1 if extracted else 0) / expected_extract_rate
        + kill_weight * kills / (expected_extract_rate * expected_kills_per_extract)
      )

    Then scaled by concavity: actual += raw * (1 - current_progress)^concavity
    """
    from mm_sim.outcomes.base import MatchResult  # noqa: F401

    if not cfg.enabled:
        return
    if result.extracted is None:
        raise ValueError("extraction season progression requires extracted array")

    expected_matches = max(mean_matches_per_day * season_days, 1.0)
    base_per_match = cfg.base_earn_per_season / expected_matches

    # Count kills per team.
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_season_progression_extraction.py -v
```

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/season_progression.py tests/test_season_progression_extraction.py
git commit -m "feat(progression): extraction-mode season earn (season-invariant, outcome-weighted, concave)"
```

---

## Task 7: Two-stage matchmaker

**Files:**
- Create: `src/mm_sim/matchmaker/two_stage.py`
- Create: `tests/test_two_stage_matchmaker.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the two-stage matchmaker."""

from __future__ import annotations

import numpy as np

from mm_sim.config import MatchmakerConfig, PopulationConfig, StageConfig
from mm_sim.matchmaker.two_stage import TwoStageMatchmaker
from mm_sim.population import Population


def _pop_with_parties(
    skills: list[float], party_ids: list[int]
) -> Population:
    cfg = PopulationConfig(initial_size=len(skills))
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    pop.true_skill[:] = np.array(skills, dtype=np.float32)
    pop.observed_skill[:] = np.array(skills, dtype=np.float32)
    pop.party_id[:] = np.array(party_ids, dtype=np.int32)
    return pop


def test_twelve_solos_form_four_teams_of_three():
    pop = _pop_with_parties(
        [float(i) / 12 for i in range(12)], list(range(12))
    )
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    assert len(lobbies[0].teams) == 4
    for team in lobbies[0].teams:
        assert len(team) == 3


def test_trio_stays_together_one_team():
    # 3 solos + 1 trio = 12 players total would need more players... use
    # 3 solos + 3 trios = 12 players = one lobby.
    skills = [0.0] * 12
    party_ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5]  # trios: 0,1,2; solos: 3,4,5
    pop = _pop_with_parties(skills, party_ids)
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    # At least one team contains party 0's three members.
    trio_players = {0, 1, 2}
    assert any(trio_players.issubset(set(team)) for team in lobbies[0].teams)


def test_duo_plus_solo_teams_together():
    skills = [0.0] * 12
    party_ids = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8]  # duos: 0, 2, 4; solos: 1,3,5,6,7,8
    pop = _pop_with_parties(skills, party_ids)
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    # Each duo's 2 members end up on the same team.
    for duo in [{0, 1}, {3, 4}, {6, 7}]:
        assert any(duo.issubset(set(team)) for team in lobbies[0].teams)


def test_partial_lobby_dropped():
    # Only 11 searching players -> no full lobby.
    pop = _pop_with_parties([0.0] * 11, list(range(11)))
    cfg = MatchmakerConfig(
        kind="two_stage", lobby_size=12, teams_per_lobby=4
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(11, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert lobbies == []


def test_stage1_groups_by_rating_proximity():
    """With skill-weighted stage1, low-skill solos team together."""
    # Low solos: skills 0.0, 0.1, 0.2; high solos: 1.8, 1.9, 2.0.
    # Mid-solos: 0.9, 1.0, 1.1, 1.2, 1.3, 1.4. 12 total -> 4 teams.
    skills = [0.0, 0.1, 0.2, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.8, 1.9, 2.0]
    pop = _pop_with_parties(skills, list(range(12)))
    cfg = MatchmakerConfig(
        kind="two_stage",
        lobby_size=12,
        teams_per_lobby=4,
        team_formation=StageConfig(
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
        ),
        lobby_assembly=StageConfig(
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
        ),
    )
    mm = TwoStageMatchmaker(cfg)
    lobbies = mm.form_lobbies(
        np.arange(12, dtype=np.int32), pop, np.random.default_rng(0)
    )
    assert len(lobbies) == 1
    # The three lowest solos (players 0, 1, 2) end up on the same team.
    assert any({0, 1, 2}.issubset(set(team)) for team in lobbies[0].teams)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_two_stage_matchmaker.py -v
```

Expected: import error.

- [ ] **Step 3: Implement TwoStageMatchmaker**

Create `src/mm_sim/matchmaker/two_stage.py`:

```python
"""Two-stage matchmaker: form teams of 3, then assemble lobbies of 4 teams."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mm_sim.config import MatchmakerConfig, StageConfig
from mm_sim.matchmaker.base import Lobby, group_by_party
from mm_sim.matchmaker.composite_mm import compute_composite_rating
from mm_sim.population import Population


class TwoStageMatchmaker:
    """Team formation + lobby assembly with independent policies per stage."""

    def __init__(self, cfg: MatchmakerConfig) -> None:
        self.cfg = cfg
        self.team_size = cfg.lobby_size // cfg.teams_per_lobby
        if cfg.lobby_size % cfg.teams_per_lobby != 0:
            raise ValueError(
                "lobby_size must be divisible by teams_per_lobby"
            )

    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]:
        parties = group_by_party(searching_player_ids, pop)

        # Stage 1: form teams of exactly self.team_size.
        teams = self._form_teams(parties, pop, self.cfg.team_formation)

        # Stage 2: assemble teams into lobbies of self.cfg.teams_per_lobby teams.
        return self._assemble_lobbies(teams, pop, self.cfg.lobby_assembly)

    def _form_teams(
        self,
        parties: list[list[int]],
        pop: Population,
        stage_cfg: StageConfig,
    ) -> list[list[int]]:
        """Combine solos/duos into teams of self.team_size; pre-sized parties
        pass through."""
        rating = compute_composite_rating(pop, stage_cfg.composite_weights)

        # Split by party size.
        full_teams: list[list[int]] = []  # already team_size
        partial: list[list[int]] = []  # need filling

        for party in parties:
            if len(party) == self.team_size:
                full_teams.append(list(party))
            elif len(party) < self.team_size:
                partial.append(list(party))
            # Parties larger than team_size are invalid per design — skip.

        # Sort partial parties by aggregate rating so similar-rating parties
        # combine together.
        partial_sorted = sorted(
            partial, key=lambda p: float(np.mean(rating[p]))
        )

        # Greedily pack partial parties into teams.
        i = 0
        while i < len(partial_sorted):
            team = list(partial_sorted[i])
            i += 1
            while len(team) < self.team_size and i < len(partial_sorted):
                candidate = partial_sorted[i]
                if len(team) + len(candidate) <= self.team_size:
                    team.extend(candidate)
                    i += 1
                else:
                    # Skip parties that would overflow; try next.
                    i += 1
            if len(team) == self.team_size:
                full_teams.append(team)

        return full_teams

    def _assemble_lobbies(
        self,
        teams: list[list[int]],
        pop: Population,
        stage_cfg: StageConfig,
    ) -> list[Lobby]:
        if len(teams) < self.cfg.teams_per_lobby:
            return []

        rating = compute_composite_rating(pop, stage_cfg.composite_weights)
        team_ratings = [float(rating[t].mean()) for t in teams]
        order = np.argsort(team_ratings)
        teams_sorted = [teams[int(i)] for i in order]

        lobbies: list[Lobby] = []
        for i in range(0, len(teams_sorted) - self.cfg.teams_per_lobby + 1,
                       self.cfg.teams_per_lobby):
            lobby_teams = teams_sorted[i : i + self.cfg.teams_per_lobby]
            lobbies.append(Lobby(teams=lobby_teams))
        return lobbies
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_two_stage_matchmaker.py -v
```

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/matchmaker/two_stage.py tests/test_two_stage_matchmaker.py
git commit -m "feat(matchmaker): two-stage matchmaker (team formation + lobby assembly)"
```

---

## Task 8: Wire extraction mode through the engine

**Files:**
- Modify: `src/mm_sim/engine.py`

- [ ] **Step 1: Write an integration test**

Create `tests/test_engine_extraction_integration.py`:

```python
"""End-to-end integration test: extraction mode runs without error."""

from __future__ import annotations

from mm_sim.config import (
    GearConfig,
    MatchmakerConfig,
    OutcomeConfig,
    PartyConfig,
    PopulationConfig,
    RatingUpdaterConfig,
    SeasonProgressionConfig,
    SimulationConfig,
    StageConfig,
)
from mm_sim.engine import SimulationEngine


def test_extraction_end_to_end_smoke():
    cfg = SimulationConfig(
        seed=42,
        season_days=5,
        population=PopulationConfig(initial_size=240),
        parties=PartyConfig(size_distribution={1: 0.5, 2: 0.2, 3: 0.3}),
        matchmaker=MatchmakerConfig(
            kind="two_stage",
            lobby_size=12,
            teams_per_lobby=4,
            team_formation=StageConfig(
                composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
            ),
            lobby_assembly=StageConfig(
                composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0}
            ),
        ),
        outcomes=OutcomeConfig(kind="extraction", gear_weight=0.5),
        gear=GearConfig(transfer_enabled=True),
        rating_updater=RatingUpdaterConfig(kind="elo_extract"),
        season_progression=SeasonProgressionConfig(enabled=True),
    )

    engine = SimulationEngine(cfg)
    df = engine.run()
    # Basic sanity — some players survive, some matches happened, some players
    # progressed on the season pass.
    assert df.height >= 5  # at least 5 day rows
    pop = engine.population
    assert pop.season_progress.max() > 0.0
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_engine_extraction_integration.py -v
```

Expected: FAIL — engine doesn't know about the new kinds.

- [ ] **Step 3: Update engine factories and tick loop**

In `src/mm_sim/engine.py`:

Replace imports and factory functions:

```python
from mm_sim.gear import (
    apply_gear_update,
    apply_gear_transfer_for_match,
    apply_extraction_gear_update,
)
from mm_sim.matchmaker.base import Matchmaker
from mm_sim.matchmaker.composite_mm import CompositeRatingMatchmaker
from mm_sim.matchmaker.random_mm import RandomMatchmaker
from mm_sim.matchmaker.two_stage import TwoStageMatchmaker
from mm_sim.outcomes.base import OutcomeGenerator
from mm_sim.outcomes.default import DefaultOutcomeGenerator
from mm_sim.outcomes.extraction import ExtractionOutcomeGenerator
from mm_sim.rating_updaters.base import RatingUpdater
from mm_sim.rating_updaters.elo import EloUpdater
from mm_sim.rating_updaters.elo_extract import ExtractEloUpdater
from mm_sim.rating_updaters.kpm import KPMUpdater
from mm_sim.season_progression import (
    apply_season_progression_update,
    apply_extraction_season_progression,
)


def _make_matchmaker(cfg: SimulationConfig) -> Matchmaker:
    kind = cfg.matchmaker.kind
    if kind == "random":
        return RandomMatchmaker(cfg.matchmaker)
    if kind == "composite":
        return CompositeRatingMatchmaker(cfg.matchmaker)
    if kind == "two_stage":
        return TwoStageMatchmaker(cfg.matchmaker)
    raise ValueError(f"unknown matchmaker kind: {kind}")


def _make_outcome_generator(cfg: SimulationConfig) -> OutcomeGenerator:
    if cfg.outcomes.kind == "default":
        return DefaultOutcomeGenerator(cfg.outcomes)
    if cfg.outcomes.kind == "extraction":
        return ExtractionOutcomeGenerator(cfg.outcomes)
    raise ValueError(f"unknown outcome kind: {cfg.outcomes.kind}")


def _make_rating_updater(cfg: SimulationConfig) -> RatingUpdater:
    kind = cfg.rating_updater.kind
    if kind == "elo":
        return EloUpdater(cfg.rating_updater)
    if kind == "kpm":
        return KPMUpdater(cfg.rating_updater)
    if kind == "elo_extract":
        return ExtractEloUpdater(cfg.rating_updater)
    raise ValueError(f"unknown rating updater kind: {kind}")
```

Then refactor the per-lobby block inside `_tick` to branch on outcome kind. Replace the block from `for lobby_idx, lobby in enumerate(lobbies):` through the end of that for-loop with:

```python
            for lobby_idx, lobby in enumerate(lobbies):
                result = self.outcome_generator.generate(
                    lobby,
                    self.population,
                    spawn_child(round_rng, f"lobby_{lobby_idx}"),
                )
                self.rating_updater.update(result, self.population)

                if self.cfg.outcomes.kind == "extraction":
                    apply_extraction_gear_update(
                        self.population, result, self.cfg.gear
                    )
                    apply_extraction_season_progression(
                        self.population,
                        result,
                        self.cfg.season_progression,
                        mean_matches_per_day=self.cfg.frequency.mean_matches_per_day,
                        season_days=self.cfg.season_days,
                    )
                    matches_today += 1
                    flat_ids = result.flat_player_ids()
                    total_matches[flat_ids] += 1
                    # Per-team win/loss accounting: extractors count as a
                    # "win" for streak purposes; dead teams count as a loss
                    # (blowout loss if killed by a significantly stronger
                    # team — threshold 1.0 for now).
                    for team_idx, team in enumerate(lobby.teams):
                        team_ids = np.array(team, dtype=np.int32)
                        if bool(result.extracted[team_idx]):
                            total_wins[team_ids] += 1
                            self.population.loss_streak[team_ids] = 0
                        else:
                            total_losses[team_ids] += 1
                            self.population.loss_streak[team_ids] += 1
                            # Blowout if killed by a team >1.0 stronger.
                            killer = next(
                                (k for (k, v) in result.kill_credits if v == team_idx),
                                None,
                            )
                            if killer is not None:
                                delta = (
                                    result.team_strength[killer]
                                    - result.team_strength[team_idx]
                                )
                                if delta > 1.0:
                                    total_blowout_losses[team_ids] += 1
                                    blowouts_today += 1

                    # Per-match snapshot writer — reuse existing record_match
                    # using extractor team index 0 as "winning_team" for legacy
                    # display purposes. Pick strongest extractor if any, else -1.
                    if result.extracted.any():
                        winning_team = int(
                            max(
                                np.flatnonzero(result.extracted).tolist(),
                                key=lambda i: float(result.team_strength[i]),
                            )
                        )
                    else:
                        winning_team = -1
                    lobby_true = self.population.true_skill[flat_ids]
                    team_trues = [
                        self.population.true_skill[np.array(t, dtype=np.int32)]
                        for t in lobby.teams
                    ]
                    self.snapshot_writer.record_match(
                        day=day,
                        match_idx=day_match_idx,
                        lobby_true_skills=lobby_true,
                        team_true_skills=team_trues,
                        is_blowout=False,  # blowout tracked per-team above
                        winning_team=winning_team,
                    )
                    day_match_idx += 1
                else:
                    # Legacy 2-team path — unchanged.
                    winners_arr = np.array(
                        lobby.teams[result.winning_team], dtype=np.int32
                    )
                    losers_arr = np.concatenate([
                        np.array(team, dtype=np.int32)
                        for team_idx, team in enumerate(lobby.teams)
                        if team_idx != result.winning_team
                    ]) if len(lobby.teams) > 1 else np.array([], dtype=np.int32)
                    apply_gear_transfer_for_match(
                        self.population,
                        winners=winners_arr,
                        losers=losers_arr,
                        is_blowout=bool(result.is_blowout),
                        cfg=self.cfg.gear,
                    )

                    matches_today += 1
                    if result.is_blowout:
                        blowouts_today += 1

                    flat_ids = result.flat_player_ids()
                    total_matches[flat_ids] += 1

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
                    self.population.loss_streak[winning_team_ids] = 0
                    for team_idx, team in enumerate(lobby.teams):
                        if team_idx == result.winning_team:
                            continue
                        losing_team_ids = np.array(team, dtype=np.int32)
                        total_losses[losing_team_ids] += 1
                        self.population.loss_streak[losing_team_ids] += 1
                        if result.is_blowout:
                            total_blowout_losses[losing_team_ids] += 1
```

Also, below the for-loop where `apply_season_progression_update(...)` and `apply_gear_update(...)` are called, we need to skip those in extraction mode (already applied per-match). Replace that block:

```python
        apply_experience_update(
            self.population,
            total_matches,
            normalization_max_matches=max(self.cfg.season_days * 5, 1),
        )
        if self.cfg.outcomes.kind == "default":
            apply_gear_update(
                self.population,
                total_matches,
                total_blowout_losses,
                self.cfg.gear,
            )
            apply_season_progression_update(
                self.population, total_matches, self.cfg.season_progression
            )
        apply_skill_progression_update(
            self.population,
            total_matches,
            self.cfg.skill_progression,
            spawn_child(day_rng, "skill_progression"),
        )
```

- [ ] **Step 4: Run integration test**

```bash
uv run pytest tests/test_engine_extraction_integration.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite to check no legacy regressions**

```bash
uv run pytest -q
```

Expected: all pass (including the legacy 2-team tests which still use `kind="default"`).

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/engine.py tests/test_engine_extraction_integration.py
git commit -m "feat(engine): wire extraction outcome, two-stage MM, extract-Elo into tick loop"
```

---

## Task 9: Default scenario opts in to extraction mode

**Files:**
- Modify: `scenarios/defaults.toml`

- [ ] **Step 1: Update defaults.toml**

Replace `scenarios/defaults.toml` with:

```toml
# Shared defaults for every scenario in this directory. Individual
# scenario files can override any of these by setting the same key.
# This file is not itself a scenario (no `name` field).
#
# Season name is computed as
#   {season_days}d-{initial_size}p-{true_skill_distribution}-{season_suffix}
# so it auto-updates when you change the underlying config.
season_suffix = "multi-team-extraction"

[config]
seed = 1999
season_days = 90

[config.population]
initial_size = 15000
true_skill_distribution = "right_skewed"
daily_new_player_fraction = 0.0067

[config.parties]
size_distribution = {1 = 0.5, 2 = 0.2, 3 = 0.3}

[config.matchmaker]
kind = "two_stage"
lobby_size = 12
teams_per_lobby = 4

[config.matchmaker.team_formation]
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.matchmaker.lobby_assembly]
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.outcomes]
kind = "extraction"
gear_weight = 0.5
baseline_extract_prob = 0.4
strength_sensitivity = 1.0

[config.rating_updater]
kind = "elo_extract"
k_factor = 32.0

[config.frequency]
mean_matches_per_day = 5.0

[config.churn]
baseline_daily_quit_prob = 0.005
loss_weight = 0.04
blowout_loss_weight = 0.14
win_streak_weight = -0.02
new_player_bonus = 0.75
new_player_threshold = 15
loss_streak_exp = 0.3
max_loss_streak_multiplier = 4.0

[config.skill_progression]
enabled = true

[config.gear]
transfer_enabled = true
extract_growth = 0.003
transfer_rate = 0.05
strength_bonus = 1.0
punching_down_floor = 0.2
transfer_efficiency = 0.9

[config.season_progression]
enabled = true
base_earn_per_season = 0.8
concavity = 1.0
participation_weight = 0.3
extraction_weight = 0.5
kill_weight = 0.2
```

- [ ] **Step 2: Smoke test: run one existing scenario**

```bash
uv run python -m mm_sim.cli scenario skill_only
```

Expected: scenario runs end-to-end, producing an experiment directory under `experiments/90d-15000p-right_skewed-multi-team-extraction/skill_only/vN/`.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add scenarios/defaults.toml
git commit -m "change(scenarios): defaults.toml opts in to multi-team extraction mode"
```

---

## Task 10: Update existing scenarios to use two_stage matchmaker

**Files:**
- Modify: `scenarios/skill_only.toml`, `scenarios/experience_only.toml`, `scenarios/skill_weighted.toml`, `scenarios/skill_gear_composite.toml`, `scenarios/random_mm.toml`, `scenarios/season_progression_off.toml`

- [ ] **Step 1: Rewrite each scenario to configure two-stage weights**

For each matchmaker-variant scenario, set both `team_formation` and `lobby_assembly` with the appropriate composite weights. Update:

`scenarios/skill_only.toml`:
```toml
name = "skill_only"
category = "matchmaker"

[config.matchmaker]
kind = "two_stage"

[config.matchmaker.team_formation]
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.matchmaker.lobby_assembly]
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}
```

`scenarios/experience_only.toml`:
```toml
name = "experience_only"
category = "matchmaker"

[config.matchmaker]
kind = "two_stage"

[config.matchmaker.team_formation]
composite_weights = {skill = 0.0, experience = 1.0, gear = 0.0}

[config.matchmaker.lobby_assembly]
composite_weights = {skill = 0.0, experience = 1.0, gear = 0.0}
```

`scenarios/skill_weighted.toml`:
```toml
name = "skill_weighted"
category = "matchmaker"

[config.matchmaker]
kind = "two_stage"

[config.matchmaker.team_formation]
composite_weights = {skill = 0.2, experience = 0.8, gear = 0.0}

[config.matchmaker.lobby_assembly]
composite_weights = {skill = 0.2, experience = 0.8, gear = 0.0}
```

`scenarios/skill_gear_composite.toml`:
```toml
name = "skill_gear_composite"
category = "matchmaker"

# Composite matchmaker balancing skill and gear in both stages.

[config.matchmaker]
kind = "two_stage"

[config.matchmaker.team_formation]
composite_weights = {skill = 0.8, experience = 0.0, gear = 0.2}

[config.matchmaker.lobby_assembly]
composite_weights = {skill = 0.8, experience = 0.0, gear = 0.2}
```

`scenarios/random_mm.toml`:
```toml
name = "random_mm"
category = "matchmaker"

# No matchmaking at either stage — lower-bound reference.

[config.matchmaker]
kind = "random"
lobby_size = 12
teams_per_lobby = 4
```

`scenarios/season_progression_off.toml`:
```toml
name = "season_progression_off"
category = "ablation"

[config.matchmaker]
kind = "two_stage"

[config.matchmaker.team_formation]
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.matchmaker.lobby_assembly]
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.season_progression]
enabled = false
```

- [ ] **Step 2: Verify random_mm still works with random matchmaker under 4-team lobby shape**

The `RandomMatchmaker` uses `pack_parties_into_lobbies`, which already supports `teams_per_lobby`. Confirm by running:

```bash
uv run python -m mm_sim.cli scenario random_mm
```

Expected: runs successfully.

- [ ] **Step 3: Run all scenarios**

```bash
uv run python -m mm_sim.cli scenarios
```

Expected: all 6 scenarios produce experiment output without error.

- [ ] **Step 4: Commit**

```bash
git add scenarios/
git commit -m "change(scenarios): scenarios use two-stage matchmaker under extraction defaults"
```

---

## Task 11: Update sweep scenarios

**Files:**
- Modify: `scenarios/sweep_mm_skill_weight.toml`, `scenarios/sweep_skill_weight.toml`, `scenarios/sweep_skill_gear_grid.toml`

- [ ] **Step 1: Update sweep parameter paths**

The sweep files use `config.matchmaker.composite_weights.*`. Under two-stage, we want to sweep both stages in lockstep. Replace:

`scenarios/sweep_mm_skill_weight.toml`:
```toml
name = "sweep_mm_skill_weight"

# Sweep both team_formation and lobby_assembly skill weight from 0 to 1
# with experience carrying the remainder. Gear weight fixed at 0.

[[sweep.zip]]
parameter = "config.matchmaker.team_formation.composite_weights.skill"
values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

[[sweep.zip]]
parameter = "config.matchmaker.team_formation.composite_weights.experience"
values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

[[sweep.zip]]
parameter = "config.matchmaker.team_formation.composite_weights.gear"
values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

[[sweep.zip]]
parameter = "config.matchmaker.lobby_assembly.composite_weights.skill"
values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

[[sweep.zip]]
parameter = "config.matchmaker.lobby_assembly.composite_weights.experience"
values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

[[sweep.zip]]
parameter = "config.matchmaker.lobby_assembly.composite_weights.gear"
values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

`scenarios/sweep_skill_weight.toml`:
```toml
name = "sweep_skill_weight"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.team_formation.composite_weights.skill"
values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.lobby_assembly.composite_weights.skill"
values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

`scenarios/sweep_skill_gear_grid.toml`:
```toml
name = "sweep_skill_gear_grid"
base_scenario = "skill_gear_composite"

[[sweep.grid]]
parameter = "config.matchmaker.team_formation.composite_weights.skill"
values = [0.0, 0.25, 0.5, 0.75, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.team_formation.composite_weights.gear"
values = [0.0, 0.25, 0.5, 0.75, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.lobby_assembly.composite_weights.skill"
values = [0.0, 0.25, 0.5, 0.75, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.lobby_assembly.composite_weights.gear"
values = [0.0, 0.25, 0.5, 0.75, 1.0]
```

- [ ] **Step 2: Smoke test one sweep point**

Pick a sweep with a small grid (e.g. `sweep_skill_weight`) and run:

```bash
uv run python -m mm_sim.cli sweep sweep_mm_skill_weight
```

Expected: all sweep points run without error.

- [ ] **Step 3: Commit**

```bash
git add scenarios/sweep_mm_skill_weight.toml scenarios/sweep_skill_weight.toml scenarios/sweep_skill_gear_grid.toml
git commit -m "change(scenarios): sweeps target two-stage matchmaker paths"
```

---

## Task 12: Snapshot writer support for extraction match-metadata

**Files:**
- Check: `src/mm_sim/snapshot.py`

- [ ] **Step 1: Review `record_match` signature**

```bash
grep -n "def record_match" src/mm_sim/snapshot.py
```

The engine already calls `record_match(day, match_idx, lobby_true_skills, team_true_skills, is_blowout, winning_team)`. In extraction mode we pass `winning_team=-1` when no one extracts and the strongest-extractor index otherwise. That signature should already accept these values.

- [ ] **Step 2: Write test confirming no regression**

```bash
uv run pytest tests/test_engine_extraction_integration.py tests/test_snapshot.py -v
```

If `tests/test_snapshot.py` exists and passes, no change needed.

If `record_match` rejects `winning_team=-1` or blows up on 4-team input, patch it: widen the type accepting `int` including `-1`, and ensure `team_true_skills` can be length 4.

- [ ] **Step 3: Commit (only if changes were made)**

```bash
git add src/mm_sim/snapshot.py
git commit -m "fix(snapshot): support 4-team lobbies and no-extractor sentinel"
```

---

## Task 13: End-to-end sanity verification

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run all scenarios end-to-end**

```bash
uv run python -m mm_sim.cli scenarios
```

Expected: all scenarios produce experiments under `experiments/90d-15000p-right_skewed-multi-team-extraction/`.

- [ ] **Step 3: Spot-check results**

Launch the dashboard and inspect:
- Player trajectories: `season_progress` curves should show meaningful variance across players (not the lockstep line we saw pre-fix).
- Gear distribution: should show spread driven by kills/upsets, not just participation.
- Churn curves: should respond to outcome asymmetry.

```bash
uv run streamlit run src/mm_sim/dashboard/app.py
```

Expected: the new season under `experiments/90d-15000p-right_skewed-multi-team-extraction/` appears in the season dropdown; scenarios load without errors.

- [ ] **Step 4: No commit** (verification only)

---

## Self-Review

**Spec coverage:**
- Match outcome model → Task 3 (extraction generator), 2 (MatchResult extension)
- Two-stage matchmaker → Task 7
- Gear transfer rules → Task 5
- Season progression → Task 6
- Rating update → Task 4
- Config schema → Task 1
- Engine wiring → Task 8
- Defaults + scenarios → Tasks 9, 10, 11

**Placeholder scan:** no TBDs, every step has code/commands.

**Type consistency:** `MatchResult` fields (`extracted`, `kill_credits`, `expected_extract`, `team_strength`) consistent across Tasks 2–8. `StageConfig` shape consistent between Tasks 1 and 7. Config field names consistent with design spec.
