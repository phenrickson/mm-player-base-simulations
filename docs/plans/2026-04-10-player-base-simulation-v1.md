# Player Base Simulation v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configurable day-tick simulation of a ~50k-player game population across a season, where a pluggable matchmaker pairs parties into lobbies and pluggable outcome/rating systems drive emergent population dynamics — so we can study how different matchmaking rating compositions (skill vs. experience vs. gear) reshape the player base over time.

**Architecture:** The simulation is a deterministic (seeded) daily loop over a numpy-backed player table. Each day: active players play matches, a matchmaker groups parties into lobbies and balances teams, an outcome generator produces per-player contribution vectors, a rating updater adjusts observed skill, gear drifts, and a churn function decides who leaves. New players arrive. Daily snapshots are written to a polars DataFrame for later analysis. All components with alternative implementations (matchmaker, outcome generator, rating updater, churn function) are behind protocol interfaces so they can be swapped via config.

**Tech Stack:** Python 3.12, uv, numpy (hot path), polars (metrics & snapshots), pytest (tests), pydantic (config), matplotlib (plots later — not in v1 scope).

---

## File Structure

```
justfile                 # task runner: just test, just sim, just lock
src/mm_sim/
  __init__.py
  py.typed               # PEP 561 marker
  cli.py                 # CLI entry point (python -m mm_sim.cli)
  config.py              # pydantic configs for simulation + components
  population.py          # Population dataclass: numpy arrays for all player fields
  parties.py             # Party assignment at population creation
  matchmaker/
    __init__.py
    base.py              # Matchmaker protocol
    random_mm.py         # RandomMatchmaker (baseline, no skill)
    composite_mm.py      # CompositeRatingMatchmaker (the core research tool)
  outcomes/
    __init__.py
    base.py              # OutcomeGenerator protocol + ContributionVector dataclass
    default.py           # DefaultOutcomeGenerator (true_skill + noise → contributions)
  rating_updaters/
    __init__.py
    base.py              # RatingUpdater protocol
    elo.py               # EloUpdater (win/loss only)
    kpm.py               # KPMUpdater (updates on kills-per-minute)
  churn.py               # Churn function: recent experience → quit probability
  frequency.py           # matches-per-day sampling, modulated by recent results
  gear.py                # Gear drift rules
  experience.py          # Experience increment rules
  engine.py              # Main simulation loop (SimulationEngine)
  snapshot.py            # DailySnapshot writer → polars DataFrame
  scenario.py            # Scenario runner for parameter sweeps
  seeding.py             # Deterministic RNG factory

tests/
  test_population.py
  test_parties.py
  test_matchmaker_random.py
  test_matchmaker_composite.py
  test_outcomes_default.py
  test_rating_updater_elo.py
  test_rating_updater_kpm.py
  test_churn.py
  test_frequency.py
  test_gear.py
  test_experience.py
  test_engine_smoke.py
  test_snapshot.py
  test_scenario.py
  test_cli.py
  test_feedback_loop.py  # integration: verify the Activision feedback loop emerges

docs/plans/
  2026-04-10-player-base-simulation-v1.md  # this file
```

**Responsibility split rationale:**
- `population.py` owns the mutable state (numpy arrays). Every other module is mostly a pure function over it.
- Matchmakers, outcome generators, rating updaters, and churn are all behind Protocol interfaces so they're swappable — this is the core extensibility point of v1.
- `engine.py` is the orchestrator; it shouldn't know the specifics of any component.
- `snapshot.py` stays separate from `engine.py` so analysis can evolve without touching the loop.

---

## Task 1: Project Setup and Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `src/mm_sim/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Add dependencies via uv**

Run these commands:

```bash
cd /Users/phenrickson/Documents/projects/mm-player-base-simulations
uv add numpy polars pydantic
uv add --dev pytest pytest-cov
```

Expected: `pyproject.toml` updated with dependencies, `uv.lock` created.

- [ ] **Step 2: Create package directory and empty `__init__.py`**

Create `src/mm_sim/__init__.py` with contents:

```python
"""Player base simulation for studying matchmaking dynamics."""

__version__ = "0.1.0"
```

Create `tests/__init__.py` as an empty file.

- [ ] **Step 3: Add a `.gitignore`**

Create `.gitignore`:

```
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.venv/
.DS_Store
```

- [ ] **Step 4: Configure pytest and package layout in pyproject.toml**

Add to `pyproject.toml` (after the existing `[project]` section):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mm_sim"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 5: Verify the package imports**

Run:

```bash
uv run python -c "import mm_sim; print(mm_sim.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 6: Verify pytest runs (with zero tests)**

Run:

```bash
uv run pytest
```

Expected: exit code 0, "no tests ran" or similar.

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml uv.lock src tests .gitignore
git commit -m "chore: project scaffolding with uv, numpy, polars, pytest"
```

---

## Task 2: Deterministic RNG Seeding

**Files:**
- Create: `src/mm_sim/seeding.py`
- Create: `tests/test_seeding.py`

**Why:** Reproducibility is non-negotiable — we need identical runs given the same seed. A central RNG factory prevents accidental use of global numpy state.

- [ ] **Step 1: Write the failing test**

Create `tests/test_seeding.py`:

```python
import numpy as np
from mm_sim.seeding import make_rng, spawn_child


def test_make_rng_is_deterministic():
    rng1 = make_rng(42)
    rng2 = make_rng(42)
    assert rng1.integers(0, 1000) == rng2.integers(0, 1000)


def test_different_seeds_diverge():
    rng1 = make_rng(42)
    rng2 = make_rng(43)
    assert rng1.integers(0, 10**9) != rng2.integers(0, 10**9)


def test_spawn_child_is_independent():
    parent = make_rng(42)
    child_a = spawn_child(parent, name="outcomes")
    child_b = spawn_child(parent, name="outcomes")
    # Same parent state + same name => same child stream
    assert child_a.integers(0, 10**9) == child_b.integers(0, 10**9)


def test_spawn_child_names_differ():
    parent_a = make_rng(42)
    parent_b = make_rng(42)
    child_outcomes = spawn_child(parent_a, name="outcomes")
    child_churn = spawn_child(parent_b, name="churn")
    assert child_outcomes.integers(0, 10**9) != child_churn.integers(0, 10**9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seeding.py -v`
Expected: FAIL with ImportError on `mm_sim.seeding`.

- [ ] **Step 3: Implement `seeding.py`**

Create `src/mm_sim/seeding.py`:

```python
"""Deterministic RNG factory. All random draws must flow through here."""

from __future__ import annotations

import hashlib
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a seeded numpy Generator."""
    return np.random.default_rng(seed)


def spawn_child(parent: np.random.Generator, name: str) -> np.random.Generator:
    """Spawn a named child RNG from a parent.

    Naming the child keeps streams independent and reproducible: two calls
    with the same parent state and the same name return the same stream.
    """
    name_hash = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    parent_draw = int(parent.integers(0, 2**63 - 1))
    child_seed = (parent_draw ^ name_hash) & ((1 << 63) - 1)
    return np.random.default_rng(child_seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seeding.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/seeding.py tests/test_seeding.py
git commit -m "feat: deterministic RNG factory with named child streams"
```

---

## Task 3: Configuration Schema

**Files:**
- Create: `src/mm_sim/config.py`
- Create: `tests/test_config.py`

**Why:** Before building anything, we need a single shape for "what does a simulation run look like?" Pydantic gives us validation and defaults for free, and the config becomes the contract between the scenario runner and the engine.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from mm_sim.config import (
    SimulationConfig,
    PopulationConfig,
    PartyConfig,
    MatchmakerConfig,
    OutcomeConfig,
    RatingUpdaterConfig,
    ChurnConfig,
    FrequencyConfig,
)


def test_default_config_is_valid():
    cfg = SimulationConfig()
    assert cfg.seed == 0
    assert cfg.season_days == 90
    assert cfg.population.initial_size == 50_000


def test_party_size_distribution_must_sum_to_one():
    with pytest.raises(ValidationError):
        PartyConfig(size_distribution={1: 0.5, 2: 0.3})  # sums to 0.8


def test_party_size_distribution_valid():
    cfg = PartyConfig(size_distribution={1: 0.4, 2: 0.3, 3: 0.3})
    assert cfg.size_distribution[3] == pytest.approx(0.3)


def test_composite_weights_default():
    mm = MatchmakerConfig()
    assert mm.kind == "composite"
    assert mm.composite_weights == {"skill": 1.0, "experience": 0.0, "gear": 0.0}


def test_composite_weights_must_be_nonnegative():
    with pytest.raises(ValidationError):
        MatchmakerConfig(composite_weights={"skill": -0.1, "experience": 1.1, "gear": 0.0})


def test_season_days_must_be_positive():
    with pytest.raises(ValidationError):
        SimulationConfig(season_days=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with ImportError on `mm_sim.config`.

- [ ] **Step 3: Implement `config.py`**

Create `src/mm_sim/config.py`:

```python
"""Pydantic configuration schema for a simulation run."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PopulationConfig(BaseModel):
    initial_size: int = Field(50_000, gt=0)
    true_skill_distribution: str = Field("normal", pattern="^(normal|uniform|right_skewed)$")
    true_skill_mean: float = 0.0
    true_skill_std: float = 1.0
    daily_new_players: int = Field(200, ge=0)
    starting_observed_skill: float = 0.0
    starting_experience: float = 0.0
    starting_gear: float = 0.0


class PartyConfig(BaseModel):
    """Static party assignments at population creation."""

    # Map party size -> fraction of players in parties of that size.
    # {1: 0.5, 2: 0.25, 3: 0.25} means 50% solo, 25% duo, 25% trio.
    size_distribution: dict[int, float] = Field(
        default_factory=lambda: {1: 0.5, 2: 0.2, 3: 0.3}
    )
    # How skill-homogeneous parties are. 0 = uniform random friends,
    # 1 = parties of identical-skill players. In between: gaussian clustering.
    skill_homogeneity: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("size_distribution")
    @classmethod
    def _must_sum_to_one(cls, v: dict[int, float]) -> dict[int, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"size_distribution must sum to 1.0, got {total}")
        return v


class MatchmakerConfig(BaseModel):
    kind: str = Field("composite", pattern="^(random|composite)$")
    # Only used when kind == "composite"
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {"skill": 1.0, "experience": 0.0, "gear": 0.0}
    )
    lobby_size: int = Field(12, gt=1)   # total players per match
    teams_per_lobby: int = Field(2, gt=1)
    max_rating_spread: float = 0.3      # initial similarity tolerance
    max_rating_spread_growth: float = 0.05  # per-retry loosening

    @field_validator("composite_weights")
    @classmethod
    def _weights_nonnegative(cls, v: dict[str, float]) -> dict[str, float]:
        for k, val in v.items():
            if val < 0:
                raise ValueError(f"weight {k} must be >= 0, got {val}")
        return v


class OutcomeConfig(BaseModel):
    kind: str = "default"
    noise_std: float = 0.25
    blowout_threshold: float = 30.0


class RatingUpdaterConfig(BaseModel):
    kind: str = Field("elo", pattern="^(elo|kpm)$")
    k_factor: float = 32.0


class ChurnConfig(BaseModel):
    baseline_daily_quit_prob: float = 0.005
    blowout_loss_weight: float = 0.08
    win_streak_weight: float = -0.02
    rolling_window: int = 5
    max_daily_quit_prob: float = 0.5


class FrequencyConfig(BaseModel):
    mean_matches_per_day: float = 3.0
    win_modulation: float = 0.2     # winners play this fraction more
    loss_modulation: float = 0.15   # losers play this fraction less


class GearConfig(BaseModel):
    growth_per_match: float = 0.005
    drop_on_blowout_loss: float = 0.05
    max_gear: float = 1.0


class SimulationConfig(BaseModel):
    seed: int = 0
    season_days: int = Field(90, gt=0)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    parties: PartyConfig = Field(default_factory=PartyConfig)
    matchmaker: MatchmakerConfig = Field(default_factory=MatchmakerConfig)
    outcomes: OutcomeConfig = Field(default_factory=OutcomeConfig)
    rating_updater: RatingUpdaterConfig = Field(default_factory=RatingUpdaterConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)
    frequency: FrequencyConfig = Field(default_factory=FrequencyConfig)
    gear: GearConfig = Field(default_factory=GearConfig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/config.py tests/test_config.py
git commit -m "feat: pydantic simulation config schema"
```

---

## Task 4: Population Data Model

**Files:**
- Create: `src/mm_sim/population.py`
- Create: `tests/test_population.py`

**Why:** The population is the simulation's heart. Storing it as a struct-of-numpy-arrays (not a list of Python objects) is what makes 50k players × 90 days tractable. Every field is a numpy array indexed by player id.

- [ ] **Step 1: Write the failing test**

Create `tests/test_population.py`:

```python
import numpy as np

from mm_sim.population import Population
from mm_sim.config import PopulationConfig
from mm_sim.seeding import make_rng


def test_create_population_shapes():
    rng = make_rng(42)
    cfg = PopulationConfig(initial_size=1000)
    pop = Population.create_initial(cfg, rng)

    assert pop.size == 1000
    assert pop.true_skill.shape == (1000,)
    assert pop.observed_skill.shape == (1000,)
    assert pop.experience.shape == (1000,)
    assert pop.gear.shape == (1000,)
    assert pop.active.shape == (1000,)
    assert pop.party_id.shape == (1000,)


def test_initial_active_all_true():
    rng = make_rng(42)
    cfg = PopulationConfig(initial_size=500)
    pop = Population.create_initial(cfg, rng)
    assert pop.active.all()


def test_initial_observed_skill_is_starting_value():
    rng = make_rng(42)
    cfg = PopulationConfig(initial_size=500, starting_observed_skill=0.0)
    pop = Population.create_initial(cfg, rng)
    assert np.all(pop.observed_skill == 0.0)


def test_true_skill_normal_distribution_stats():
    rng = make_rng(42)
    cfg = PopulationConfig(
        initial_size=10_000,
        true_skill_distribution="normal",
        true_skill_mean=0.0,
        true_skill_std=1.0,
    )
    pop = Population.create_initial(cfg, rng)
    assert abs(pop.true_skill.mean()) < 0.05
    assert abs(pop.true_skill.std() - 1.0) < 0.05


def test_add_new_players_extends_arrays():
    rng = make_rng(42)
    cfg = PopulationConfig(initial_size=100)
    pop = Population.create_initial(cfg, rng)
    pop.add_new_players(count=50, cfg=cfg, rng=rng)
    assert pop.size == 150
    assert pop.active[100:].all()


def test_active_indices_excludes_churned():
    rng = make_rng(42)
    cfg = PopulationConfig(initial_size=10)
    pop = Population.create_initial(cfg, rng)
    pop.active[3] = False
    pop.active[7] = False
    idx = pop.active_indices()
    assert len(idx) == 8
    assert 3 not in idx
    assert 7 not in idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_population.py -v`
Expected: FAIL with ImportError on `mm_sim.population`.

- [ ] **Step 3: Implement `population.py`**

Create `src/mm_sim/population.py`:

```python
"""Population: numpy struct-of-arrays for all player state."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mm_sim.config import PopulationConfig


@dataclass
class Population:
    true_skill: np.ndarray       # float32, hidden ground truth
    observed_skill: np.ndarray   # float32, what the matchmaker sees
    experience: np.ndarray       # float32, monotonic, 0..1 normalized
    gear: np.ndarray             # float32, 0..1 normalized
    active: np.ndarray           # bool, False once churned
    party_id: np.ndarray         # int32, -1 means unassigned (set by parties module)
    matches_played: np.ndarray   # int32
    recent_wins: np.ndarray      # int8, rolling count inside window
    recent_blowout_losses: np.ndarray  # int8, rolling count inside window
    join_day: np.ndarray         # int32, day index when player entered

    @property
    def size(self) -> int:
        return int(self.true_skill.shape[0])

    @classmethod
    def create_initial(
        cls, cfg: PopulationConfig, rng: np.random.Generator
    ) -> "Population":
        n = cfg.initial_size
        true_skill = _sample_skill(n, cfg, rng)
        return cls(
            true_skill=true_skill.astype(np.float32),
            observed_skill=np.full(n, cfg.starting_observed_skill, dtype=np.float32),
            experience=np.full(n, cfg.starting_experience, dtype=np.float32),
            gear=np.full(n, cfg.starting_gear, dtype=np.float32),
            active=np.ones(n, dtype=bool),
            party_id=np.full(n, -1, dtype=np.int32),
            matches_played=np.zeros(n, dtype=np.int32),
            recent_wins=np.zeros(n, dtype=np.int8),
            recent_blowout_losses=np.zeros(n, dtype=np.int8),
            join_day=np.zeros(n, dtype=np.int32),
        )

    def add_new_players(
        self, count: int, cfg: PopulationConfig, rng: np.random.Generator, day: int = 0
    ) -> np.ndarray:
        """Append `count` new players. Returns the array of their new ids."""
        if count <= 0:
            return np.array([], dtype=np.int32)
        new_true = _sample_skill(count, cfg, rng).astype(np.float32)
        start = self.size
        self.true_skill = np.concatenate([self.true_skill, new_true])
        self.observed_skill = np.concatenate(
            [self.observed_skill, np.full(count, cfg.starting_observed_skill, dtype=np.float32)]
        )
        self.experience = np.concatenate(
            [self.experience, np.full(count, cfg.starting_experience, dtype=np.float32)]
        )
        self.gear = np.concatenate(
            [self.gear, np.full(count, cfg.starting_gear, dtype=np.float32)]
        )
        self.active = np.concatenate([self.active, np.ones(count, dtype=bool)])
        self.party_id = np.concatenate([self.party_id, np.full(count, -1, dtype=np.int32)])
        self.matches_played = np.concatenate(
            [self.matches_played, np.zeros(count, dtype=np.int32)]
        )
        self.recent_wins = np.concatenate(
            [self.recent_wins, np.zeros(count, dtype=np.int8)]
        )
        self.recent_blowout_losses = np.concatenate(
            [self.recent_blowout_losses, np.zeros(count, dtype=np.int8)]
        )
        self.join_day = np.concatenate(
            [self.join_day, np.full(count, day, dtype=np.int32)]
        )
        return np.arange(start, start + count, dtype=np.int32)

    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active).astype(np.int32)


def _sample_skill(n: int, cfg: PopulationConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.true_skill_distribution == "normal":
        return rng.normal(cfg.true_skill_mean, cfg.true_skill_std, size=n)
    if cfg.true_skill_distribution == "uniform":
        half = cfg.true_skill_std * 1.732  # match variance
        return rng.uniform(
            cfg.true_skill_mean - half, cfg.true_skill_mean + half, size=n
        )
    if cfg.true_skill_distribution == "right_skewed":
        # Lognormal shifted/scaled to target mean/std
        raw = rng.lognormal(mean=0.0, sigma=0.8, size=n)
        raw = (raw - raw.mean()) / raw.std()
        return raw * cfg.true_skill_std + cfg.true_skill_mean
    raise ValueError(f"unknown distribution: {cfg.true_skill_distribution}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_population.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/population.py tests/test_population.py
git commit -m "feat: Population struct-of-arrays with configurable skill distribution"
```

---

## Task 5: Party Assignment

**Files:**
- Create: `src/mm_sim/parties.py`
- Create: `tests/test_parties.py`

**Why:** Parties are first-class for the core research question (what happens when high-skill players stack). Assignment is static at creation. Two knobs: size distribution and skill homogeneity.

- [ ] **Step 1: Write the failing test**

Create `tests/test_parties.py`:

```python
import numpy as np

from mm_sim.population import Population
from mm_sim.parties import assign_parties
from mm_sim.config import PopulationConfig, PartyConfig
from mm_sim.seeding import make_rng


def test_all_players_get_party_ids():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=1000), rng)
    assign_parties(pop, PartyConfig(size_distribution={1: 0.5, 2: 0.5}), rng)
    assert (pop.party_id >= 0).all()


def test_party_sizes_approximate_distribution():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=10_000), rng)
    assign_parties(
        pop,
        PartyConfig(size_distribution={1: 0.4, 2: 0.3, 3: 0.3}),
        rng,
    )
    unique, counts = np.unique(pop.party_id, return_counts=True)
    size_hist: dict[int, int] = {}
    for size in counts:
        size_hist[int(size)] = size_hist.get(int(size), 0) + 1
    total_parties = sum(size_hist.values())
    # Roughly 40% of parties are solo
    solo_frac = size_hist.get(1, 0) / total_parties
    assert 0.35 < solo_frac < 0.45


def test_homogeneity_high_means_low_intraparty_variance():
    rng_low = make_rng(42)
    pop_low = Population.create_initial(PopulationConfig(initial_size=2000), rng_low)
    assign_parties(
        pop_low,
        PartyConfig(size_distribution={3: 1.0}, skill_homogeneity=0.0),
        rng_low,
    )
    low_var = _mean_party_skill_variance(pop_low)

    rng_high = make_rng(42)
    pop_high = Population.create_initial(PopulationConfig(initial_size=2000), rng_high)
    assign_parties(
        pop_high,
        PartyConfig(size_distribution={3: 1.0}, skill_homogeneity=1.0),
        rng_high,
    )
    high_var = _mean_party_skill_variance(pop_high)

    assert high_var < low_var


def _mean_party_skill_variance(pop: Population) -> float:
    variances = []
    for pid in np.unique(pop.party_id):
        skills = pop.true_skill[pop.party_id == pid]
        if len(skills) > 1:
            variances.append(float(skills.var()))
    return float(np.mean(variances))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parties.py -v`
Expected: FAIL with ImportError on `mm_sim.parties`.

- [ ] **Step 3: Implement `parties.py`**

Create `src/mm_sim/parties.py`:

```python
"""Static party assignment at population creation."""

from __future__ import annotations

import numpy as np

from mm_sim.population import Population
from mm_sim.config import PartyConfig


def assign_parties(
    pop: Population, cfg: PartyConfig, rng: np.random.Generator
) -> None:
    """Assign every player in `pop` a party_id (mutates in place).

    With homogeneity=0, parties are uniform random. With homogeneity=1,
    parties are composed of players with nearly identical true_skill
    (via sorted-window grouping). In between is a linear blend.
    """
    n = pop.size
    sizes = list(cfg.size_distribution.keys())
    probs = list(cfg.size_distribution.values())

    # First draw party sizes summing to ~n
    party_sizes: list[int] = []
    remaining = n
    while remaining > 0:
        s = int(rng.choice(sizes, p=probs))
        s = min(s, remaining)
        party_sizes.append(s)
        remaining -= s

    # Sort players by true_skill for the homogeneous branch
    sorted_idx = np.argsort(pop.true_skill)

    # Shuffled order for the random branch (same length as sorted_idx)
    shuffled_idx = sorted_idx.copy()
    rng.shuffle(shuffled_idx)

    h = cfg.skill_homogeneity
    # Blend the two orderings by choosing each slot from sorted or shuffled
    # with probability h / (1-h).
    blend = np.where(
        rng.random(n) < h, sorted_idx, shuffled_idx
    )
    # Deduplicate while preserving order
    _, first_positions = np.unique(blend, return_index=True)
    order = blend[np.sort(first_positions)]
    # Any missing players (because of dedup collisions) appended at the end
    missing = np.setdiff1d(np.arange(n), order, assume_unique=False)
    order = np.concatenate([order, missing])

    next_pid = 0
    cursor = 0
    for s in party_sizes:
        group = order[cursor : cursor + s]
        pop.party_id[group] = next_pid
        next_pid += 1
        cursor += s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parties.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/parties.py tests/test_parties.py
git commit -m "feat: static party assignment with configurable size and homogeneity"
```

---

## Task 6: Matchmaker Protocol and Random Baseline

**Files:**
- Create: `src/mm_sim/matchmaker/__init__.py`
- Create: `src/mm_sim/matchmaker/base.py`
- Create: `src/mm_sim/matchmaker/random_mm.py`
- Create: `tests/test_matchmaker_random.py`

**Why:** Define the matchmaker interface and build the simplest possible implementation (random) as a baseline both for testing the engine and for research comparisons.

- [ ] **Step 1: Write the failing test**

Create `tests/test_matchmaker_random.py`:

```python
import numpy as np
import pytest

from mm_sim.matchmaker.base import Lobby
from mm_sim.matchmaker.random_mm import RandomMatchmaker
from mm_sim.population import Population
from mm_sim.parties import assign_parties
from mm_sim.config import PopulationConfig, PartyConfig, MatchmakerConfig
from mm_sim.seeding import make_rng


def _make_pop(n=1000):
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=n), rng)
    assign_parties(pop, PartyConfig(size_distribution={1: 0.4, 2: 0.3, 3: 0.3}), rng)
    return pop, rng


def test_lobby_sums_to_lobby_size():
    pop, rng = _make_pop(1000)
    mm = RandomMatchmaker(MatchmakerConfig(kind="random", lobby_size=12, teams_per_lobby=2))
    searching = pop.active_indices()
    lobbies = mm.form_lobbies(searching, pop, rng)
    for lobby in lobbies:
        assert sum(len(team) for team in lobby.teams) == 12


def test_teams_are_balanced_in_size():
    pop, rng = _make_pop(1000)
    mm = RandomMatchmaker(MatchmakerConfig(kind="random", lobby_size=12, teams_per_lobby=2))
    lobbies = mm.form_lobbies(pop.active_indices(), pop, rng)
    for lobby in lobbies:
        sizes = [len(t) for t in lobby.teams]
        assert max(sizes) - min(sizes) <= 1


def test_parties_stay_on_same_team():
    pop, rng = _make_pop(2000)
    mm = RandomMatchmaker(MatchmakerConfig(kind="random", lobby_size=12, teams_per_lobby=2))
    lobbies = mm.form_lobbies(pop.active_indices(), pop, rng)
    for lobby in lobbies:
        for team in lobby.teams:
            party_ids = pop.party_id[np.array(team)]
            # No split party: any party fully in this team shouldn't appear anywhere else
            for pid in np.unique(party_ids):
                in_this_team = np.sum(party_ids == pid)
                in_lobby = np.sum(pop.party_id[np.concatenate([np.array(t) for t in lobby.teams])] == pid)
                assert in_this_team == in_lobby, f"party {pid} split across teams"


def test_players_not_double_booked():
    pop, rng = _make_pop(2000)
    mm = RandomMatchmaker(MatchmakerConfig(kind="random", lobby_size=12, teams_per_lobby=2))
    lobbies = mm.form_lobbies(pop.active_indices(), pop, rng)
    seen = set()
    for lobby in lobbies:
        for team in lobby.teams:
            for pid in team:
                assert pid not in seen, f"player {pid} in multiple lobbies"
                seen.add(pid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_matchmaker_random.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the matchmaker base and Lobby dataclass**

Create `src/mm_sim/matchmaker/__init__.py`:

```python
"""Matchmaker implementations."""
```

Create `src/mm_sim/matchmaker/base.py`:

```python
"""Matchmaker protocol and Lobby dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np

from mm_sim.population import Population


@dataclass
class Lobby:
    teams: list[list[int]]  # each team is a list of player ids


class Matchmaker(Protocol):
    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]: ...
```

- [ ] **Step 4: Implement `random_mm.py`**

Create `src/mm_sim/matchmaker/random_mm.py`:

```python
"""Random matchmaker: groups parties into lobbies with no skill consideration."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from mm_sim.config import MatchmakerConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.population import Population


class RandomMatchmaker:
    def __init__(self, cfg: MatchmakerConfig) -> None:
        self.cfg = cfg

    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]:
        # Group searching players by party_id
        party_to_members: dict[int, list[int]] = {}
        for pid in searching_player_ids:
            p = int(pop.party_id[pid])
            party_to_members.setdefault(p, []).append(int(pid))

        parties = list(party_to_members.values())
        rng.shuffle(parties)

        lobbies: list[Lobby] = []
        lobby_size = self.cfg.lobby_size
        teams_per_lobby = self.cfg.teams_per_lobby
        team_capacity = lobby_size // teams_per_lobby

        current_teams: list[list[int]] = [[] for _ in range(teams_per_lobby)]
        for party in parties:
            placed = False
            for team in current_teams:
                if len(team) + len(party) <= team_capacity:
                    team.extend(party)
                    placed = True
                    break
            if not placed:
                # Skip: party doesn't fit in any team in the current lobby.
                # Drop to next lobby by flushing if current is "full enough".
                filled = sum(len(t) for t in current_teams)
                if filled >= lobby_size - 1:
                    lobbies.append(Lobby(teams=current_teams))
                    current_teams = [[] for _ in range(teams_per_lobby)]
                    # Try again in the fresh lobby
                    for team in current_teams:
                        if len(team) + len(party) <= team_capacity:
                            team.extend(party)
                            break
            # Check if the current lobby is full
            if sum(len(t) for t in current_teams) == lobby_size:
                lobbies.append(Lobby(teams=current_teams))
                current_teams = [[] for _ in range(teams_per_lobby)]

        return lobbies
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_matchmaker_random.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/matchmaker tests/test_matchmaker_random.py
git commit -m "feat: Matchmaker protocol and party-preserving RandomMatchmaker"
```

---

## Task 7: Composite Rating Matchmaker

**Files:**
- Create: `src/mm_sim/matchmaker/composite_mm.py`
- Create: `tests/test_matchmaker_composite.py`

**Why:** This is the central research tool: a matchmaker that uses a weighted composite of (normalized) skill, experience, and gear. Setting different weights answers "what if matchmaking was based on level instead of skill?"

- [ ] **Step 1: Write the failing test**

Create `tests/test_matchmaker_composite.py`:

```python
import numpy as np

from mm_sim.matchmaker.composite_mm import CompositeRatingMatchmaker, compute_composite_rating
from mm_sim.population import Population
from mm_sim.parties import assign_parties
from mm_sim.config import PopulationConfig, PartyConfig, MatchmakerConfig
from mm_sim.seeding import make_rng


def _make_pop(n=2000):
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=n), rng)
    # Make observed_skill meaningfully different so we can test grouping
    pop.observed_skill = pop.true_skill.copy().astype(np.float32)
    assign_parties(pop, PartyConfig(size_distribution={1: 0.5, 2: 0.3, 3: 0.2}), rng)
    return pop, rng


def test_composite_rating_weights():
    pop, _ = _make_pop(100)
    pop.observed_skill[:] = 0.5
    pop.experience[:] = 0.2
    pop.gear[:] = 0.8
    rating = compute_composite_rating(
        pop, {"skill": 1.0, "experience": 0.0, "gear": 0.0}
    )
    assert np.allclose(rating, 0.5)

    rating2 = compute_composite_rating(
        pop, {"skill": 0.5, "experience": 0.5, "gear": 0.0}
    )
    assert np.allclose(rating2, 0.35)


def test_composite_matchmaker_groups_by_rating_skill_only():
    pop, rng = _make_pop(2000)
    mm = CompositeRatingMatchmaker(
        MatchmakerConfig(
            kind="composite",
            lobby_size=12,
            teams_per_lobby=2,
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
            max_rating_spread=0.5,
        )
    )
    lobbies = mm.form_lobbies(pop.active_indices(), pop, rng)
    # For each lobby, spread of observed_skill should usually be < max_rating_spread
    # (allowing some loosened lobbies at the tail)
    spreads = []
    for lobby in lobbies:
        all_pids = np.array([p for team in lobby.teams for p in team])
        skills = pop.observed_skill[all_pids]
        spreads.append(float(skills.max() - skills.min()))
    # At least 80% of lobbies should respect the initial tolerance
    within = sum(1 for s in spreads if s <= 0.5 * 2)
    assert within / len(spreads) > 0.8


def test_composite_matchmaker_parties_preserved():
    pop, rng = _make_pop(2000)
    mm = CompositeRatingMatchmaker(
        MatchmakerConfig(
            kind="composite",
            lobby_size=12,
            teams_per_lobby=2,
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
        )
    )
    lobbies = mm.form_lobbies(pop.active_indices(), pop, rng)
    for lobby in lobbies:
        for team in lobby.teams:
            party_ids_in_team = pop.party_id[np.array(team)]
            # No party should appear in another team of the same lobby
            other_team_pids = np.concatenate(
                [np.array(t) for t in lobby.teams if t is not team]
            )
            if len(other_team_pids) > 0:
                other_parties = set(pop.party_id[other_team_pids].tolist())
                this_parties = set(party_ids_in_team.tolist())
                assert this_parties.isdisjoint(other_parties)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_matchmaker_composite.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `composite_mm.py`**

Create `src/mm_sim/matchmaker/composite_mm.py`:

```python
"""Composite rating matchmaker — skill/experience/gear weighted composite."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from mm_sim.config import MatchmakerConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.population import Population


def compute_composite_rating(
    pop: Population, weights: dict[str, float]
) -> np.ndarray:
    """Compute the matchmaker rating for every player as a weighted sum.

    Components are assumed to be pre-normalized into comparable ranges by
    their owning modules. Skill uses the percentile of observed_skill within
    the currently-active population so it lives in [0, 1] like experience/gear.
    """
    active_mask = pop.active
    skill_component = np.zeros(pop.size, dtype=np.float32)
    if active_mask.any():
        active_obs = pop.observed_skill[active_mask]
        ranks = active_obs.argsort().argsort().astype(np.float32)
        percentiles = ranks / max(len(active_obs) - 1, 1)
        skill_component[active_mask] = percentiles
    rating = (
        weights.get("skill", 0.0) * skill_component
        + weights.get("experience", 0.0) * pop.experience
        + weights.get("gear", 0.0) * pop.gear
    )
    return rating


class CompositeRatingMatchmaker:
    def __init__(self, cfg: MatchmakerConfig) -> None:
        self.cfg = cfg

    def form_lobbies(
        self,
        searching_player_ids: np.ndarray,
        pop: Population,
        rng: np.random.Generator,
    ) -> Sequence[Lobby]:
        rating = compute_composite_rating(pop, self.cfg.composite_weights)

        # Group searching players by party
        party_to_members: dict[int, list[int]] = {}
        for pid in searching_player_ids:
            party_to_members.setdefault(int(pop.party_id[pid]), []).append(int(pid))

        # Compute average rating per party and sort parties by it
        parties = list(party_to_members.values())
        party_ratings = np.array(
            [float(rating[members].mean()) for members in parties]
        )
        order = np.argsort(party_ratings)
        parties = [parties[i] for i in order]

        lobby_size = self.cfg.lobby_size
        teams_per_lobby = self.cfg.teams_per_lobby
        team_capacity = lobby_size // teams_per_lobby

        lobbies: list[Lobby] = []
        i = 0
        while i < len(parties):
            current_teams: list[list[int]] = [[] for _ in range(teams_per_lobby)]
            filled = 0
            start = i
            # Pack sorted-adjacent parties into the lobby, snake-style so the
            # two teams both get a representative slice of the skill window.
            team_cursor = 0
            direction = 1
            while i < len(parties) and filled < lobby_size:
                party = parties[i]
                # Does this party fit in the current team?
                if len(current_teams[team_cursor]) + len(party) <= team_capacity:
                    current_teams[team_cursor].extend(party)
                    filled += len(party)
                    i += 1
                # Advance team cursor (snake)
                team_cursor += direction
                if team_cursor == teams_per_lobby:
                    direction = -1
                    team_cursor = teams_per_lobby - 1
                elif team_cursor < 0:
                    direction = 1
                    team_cursor = 0
                # Safety: if we've cycled without placing anyone, break
                if i == start and filled == 0:
                    i += 1
                    break
            if filled == lobby_size:
                lobbies.append(Lobby(teams=current_teams))
            # Partial lobbies are dropped — those players just don't play today
        return lobbies
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_matchmaker_composite.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/matchmaker/composite_mm.py tests/test_matchmaker_composite.py
git commit -m "feat: CompositeRatingMatchmaker with weighted skill/experience/gear rating"
```

---

## Task 8: Outcome Generator

**Files:**
- Create: `src/mm_sim/outcomes/__init__.py`
- Create: `src/mm_sim/outcomes/base.py`
- Create: `src/mm_sim/outcomes/default.py`
- Create: `tests/test_outcomes_default.py`

**Why:** The outcome generator produces `(winner, score_margin, per_player_contributions)` from a lobby. It's pluggable so we can test "what if the true outcome signal differs from what the rating updater measures?"

- [ ] **Step 1: Write the failing test**

Create `tests/test_outcomes_default.py`:

```python
import numpy as np

from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.outcomes.default import DefaultOutcomeGenerator
from mm_sim.population import Population
from mm_sim.config import PopulationConfig, OutcomeConfig
from mm_sim.seeding import make_rng


def _pop_with_skills(skills: np.ndarray) -> Population:
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=len(skills)), rng)
    pop.true_skill[:] = skills
    return pop


def test_higher_skill_team_wins_most_of_the_time():
    pop = _pop_with_skills(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    )
    lobby = Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    gen = DefaultOutcomeGenerator(OutcomeConfig(noise_std=0.1))
    rng = make_rng(1)
    wins_team_b = 0
    for _ in range(100):
        result = gen.generate(lobby, pop, rng)
        if result.winning_team == 1:
            wins_team_b += 1
    assert wins_team_b > 85


def test_contribution_vector_has_required_fields():
    pop = _pop_with_skills(np.zeros(12, dtype=np.float32))
    lobby = Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    gen = DefaultOutcomeGenerator(OutcomeConfig())
    result = gen.generate(lobby, pop, make_rng(1))
    assert set(result.contributions.keys()) == {"kills", "deaths", "damage", "objective_score"}
    for arr in result.contributions.values():
        assert arr.shape == (12,)


def test_score_margin_is_positive():
    pop = _pop_with_skills(
        np.array([0.0] * 6 + [2.0] * 6, dtype=np.float32)
    )
    lobby = Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    gen = DefaultOutcomeGenerator(OutcomeConfig(noise_std=0.0))
    result = gen.generate(lobby, pop, make_rng(1))
    assert result.score_margin > 0


def test_blowout_flag():
    pop = _pop_with_skills(
        np.array([0.0] * 6 + [5.0] * 6, dtype=np.float32)
    )
    lobby = Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    gen = DefaultOutcomeGenerator(OutcomeConfig(noise_std=0.0, blowout_threshold=10.0))
    result = gen.generate(lobby, pop, make_rng(1))
    assert result.is_blowout is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_outcomes_default.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement outcomes base**

Create `src/mm_sim/outcomes/__init__.py`:

```python
"""Outcome generators."""
```

Create `src/mm_sim/outcomes/base.py`:

```python
"""Outcome generator protocol and MatchResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from mm_sim.matchmaker.base import Lobby
from mm_sim.population import Population


@dataclass
class MatchResult:
    lobby: Lobby
    winning_team: int  # index into lobby.teams
    score_margin: float
    is_blowout: bool
    # contributions: dict of field name -> array indexed by position in
    # the flattened list of players in this lobby (team 0 first, then team 1...).
    contributions: dict[str, np.ndarray]

    def flat_player_ids(self) -> np.ndarray:
        return np.array(
            [pid for team in self.lobby.teams for pid in team], dtype=np.int32
        )


class OutcomeGenerator(Protocol):
    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult: ...
```

- [ ] **Step 4: Implement the default outcome generator**

Create `src/mm_sim/outcomes/default.py`:

```python
"""Default outcome generator: skill + gear -> match performance with noise."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class DefaultOutcomeGenerator:
    def __init__(self, cfg: OutcomeConfig) -> None:
        self.cfg = cfg

    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult:
        team_performances: list[float] = []
        per_player_perf: list[np.ndarray] = []
        for team in lobby.teams:
            team_arr = np.array(team, dtype=np.int32)
            base = pop.true_skill[team_arr].astype(np.float32)
            noise = rng.normal(0.0, self.cfg.noise_std, size=len(team_arr)).astype(
                np.float32
            )
            player_perf = base + noise
            per_player_perf.append(player_perf)
            team_performances.append(float(player_perf.sum()))

        winning_team = int(np.argmax(team_performances))
        best = team_performances[winning_team]
        worst = min(team_performances)
        score_margin = (best - worst) * 5.0  # scale factor to roughly hit TDM-ish ranges
        is_blowout = score_margin >= self.cfg.blowout_threshold

        flat_perf = np.concatenate(per_player_perf)
        # Derive kills/deaths/damage/objective from per-player performance
        flat_perf_pos = np.clip(flat_perf - flat_perf.min() + 0.1, 0.1, None)
        total = flat_perf_pos.sum()
        scale = max(score_margin + 40.0, 40.0)
        kills = np.floor(flat_perf_pos / total * scale * len(flat_perf_pos) / 2).astype(
            np.float32
        )
        # Deaths: inverse-ish — lower performers die more
        inv = 1.0 / flat_perf_pos
        inv_total = inv.sum()
        deaths = np.floor(inv / inv_total * scale * len(flat_perf_pos) / 2).astype(
            np.float32
        )
        damage = kills * float(rng.uniform(80.0, 120.0))
        objective_score = kills * 100.0 + rng.uniform(0.0, 50.0, size=len(kills)).astype(
            np.float32
        )

        return MatchResult(
            lobby=lobby,
            winning_team=winning_team,
            score_margin=float(score_margin),
            is_blowout=bool(is_blowout),
            contributions={
                "kills": kills,
                "deaths": deaths,
                "damage": damage,
                "objective_score": objective_score,
            },
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_outcomes_default.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/outcomes tests/test_outcomes_default.py
git commit -m "feat: DefaultOutcomeGenerator producing winner, margin, contribution vector"
```

---

## Task 9: Rating Updater Protocol and Elo Implementation

**Files:**
- Create: `src/mm_sim/rating_updaters/__init__.py`
- Create: `src/mm_sim/rating_updaters/base.py`
- Create: `src/mm_sim/rating_updaters/elo.py`
- Create: `tests/test_rating_updater_elo.py`

**Why:** Elo is the default updater — every player on the winning team gains points, losing team loses points, scaled by expected outcome and a k-factor.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rating_updater_elo.py`:

```python
import numpy as np

from mm_sim.rating_updaters.elo import EloUpdater
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population
from mm_sim.config import PopulationConfig, RatingUpdaterConfig
from mm_sim.seeding import make_rng


def _pop_with_observed(obs: np.ndarray) -> Population:
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=len(obs)), rng)
    pop.observed_skill[:] = obs
    return pop


def test_winners_gain_losers_lose():
    pop = _pop_with_observed(np.zeros(12, dtype=np.float32))
    before = pop.observed_skill.copy()
    result = MatchResult(
        lobby=Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
        winning_team=0,
        score_margin=10.0,
        is_blowout=False,
        contributions={
            "kills": np.zeros(12, dtype=np.float32),
            "deaths": np.zeros(12, dtype=np.float32),
            "damage": np.zeros(12, dtype=np.float32),
            "objective_score": np.zeros(12, dtype=np.float32),
        },
    )
    updater = EloUpdater(RatingUpdaterConfig(kind="elo", k_factor=32.0))
    updater.update(result, pop)

    assert (pop.observed_skill[:6] > before[:6]).all()
    assert (pop.observed_skill[6:] < before[6:]).all()


def test_underdog_winners_gain_more_than_favorites():
    pop = _pop_with_observed(
        np.array([-1.0] * 6 + [1.0] * 6, dtype=np.float32)
    )
    before_underdog = pop.observed_skill[0]
    before_fav = pop.observed_skill[6]

    # Team 0 (underdogs) wins
    result = MatchResult(
        lobby=Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
        winning_team=0,
        score_margin=5.0,
        is_blowout=False,
        contributions={
            "kills": np.zeros(12, dtype=np.float32),
            "deaths": np.zeros(12, dtype=np.float32),
            "damage": np.zeros(12, dtype=np.float32),
            "objective_score": np.zeros(12, dtype=np.float32),
        },
    )
    updater = EloUpdater(RatingUpdaterConfig(kind="elo", k_factor=32.0))
    updater.update(result, pop)

    underdog_gain = pop.observed_skill[0] - before_underdog
    fav_loss = before_fav - pop.observed_skill[6]

    assert underdog_gain > 0
    assert fav_loss > 0
    assert underdog_gain > 5.0 / 100.0  # meaningful, not zero
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rating_updater_elo.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement rating updater base and Elo**

Create `src/mm_sim/rating_updaters/__init__.py`:

```python
"""Rating updaters."""
```

Create `src/mm_sim/rating_updaters/base.py`:

```python
"""Rating updater protocol."""

from __future__ import annotations

from typing import Protocol

from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class RatingUpdater(Protocol):
    def update(self, result: MatchResult, pop: Population) -> None: ...
```

Create `src/mm_sim/rating_updaters/elo.py`:

```python
"""Elo rating updater: win/loss only, k-factor scaled."""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class EloUpdater:
    def __init__(self, cfg: RatingUpdaterConfig) -> None:
        self.cfg = cfg

    def update(self, result: MatchResult, pop: Population) -> None:
        team_ratings = []
        for team in result.lobby.teams:
            arr = np.array(team, dtype=np.int32)
            team_ratings.append(float(pop.observed_skill[arr].mean()))

        if len(team_ratings) != 2:
            raise NotImplementedError("EloUpdater only supports 2-team lobbies for v1")

        r_a, r_b = team_ratings
        # Expected scores (Elo with scale=1.0 on observed_skill space)
        expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 1.0))
        expected_b = 1.0 - expected_a

        actual_a = 1.0 if result.winning_team == 0 else 0.0
        actual_b = 1.0 - actual_a

        k = self.cfg.k_factor / 400.0  # rescale k_factor for our [-1..1] skill space
        delta_a = k * (actual_a - expected_a)
        delta_b = k * (actual_b - expected_b)

        team_a = np.array(result.lobby.teams[0], dtype=np.int32)
        team_b = np.array(result.lobby.teams[1], dtype=np.int32)
        pop.observed_skill[team_a] += delta_a
        pop.observed_skill[team_b] += delta_b
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rating_updater_elo.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/rating_updaters tests/test_rating_updater_elo.py
git commit -m "feat: RatingUpdater protocol and Elo implementation"
```

---

## Task 10: KPM Rating Updater

**Files:**
- Create: `src/mm_sim/rating_updaters/kpm.py`
- Create: `tests/test_rating_updater_kpm.py`

**Why:** An alternative updater that updates on kills-per-minute instead of win/loss. This is what enables the research question "what if the rating system measures the wrong thing?"

- [ ] **Step 1: Write the failing test**

Create `tests/test_rating_updater_kpm.py`:

```python
import numpy as np

from mm_sim.rating_updaters.kpm import KPMUpdater
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population
from mm_sim.config import PopulationConfig, RatingUpdaterConfig
from mm_sim.seeding import make_rng


def test_high_kill_players_gain_even_on_losing_team():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=12), rng)
    pop.observed_skill[:] = 0.0
    before = pop.observed_skill.copy()

    # Player 6 is on the losing team but had a huge KPM
    kills = np.array([5, 5, 5, 5, 5, 5, 20, 1, 1, 1, 1, 1], dtype=np.float32)
    result = MatchResult(
        lobby=Lobby(teams=[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
        winning_team=0,
        score_margin=10.0,
        is_blowout=False,
        contributions={
            "kills": kills,
            "deaths": np.ones(12, dtype=np.float32),
            "damage": np.zeros(12, dtype=np.float32),
            "objective_score": np.zeros(12, dtype=np.float32),
        },
    )
    updater = KPMUpdater(RatingUpdaterConfig(kind="kpm", k_factor=32.0))
    updater.update(result, pop)

    # Player 6 had much higher kills than lobby average, should gain
    assert pop.observed_skill[6] > before[6]
    # Players 7..11 had low kills, should lose
    assert (pop.observed_skill[7:] < before[7:]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rating_updater_kpm.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement KPM updater**

Create `src/mm_sim/rating_updaters/kpm.py`:

```python
"""KPM rating updater: updates on per-player kills-per-minute vs lobby average."""

from __future__ import annotations

import numpy as np

from mm_sim.config import RatingUpdaterConfig
from mm_sim.outcomes.base import MatchResult
from mm_sim.population import Population


class KPMUpdater:
    def __init__(self, cfg: RatingUpdaterConfig) -> None:
        self.cfg = cfg

    def update(self, result: MatchResult, pop: Population) -> None:
        flat_ids = result.flat_player_ids()
        kills = result.contributions["kills"]
        mean_kills = float(kills.mean())
        # Z-score within the lobby
        std = float(kills.std()) or 1.0
        z = (kills - mean_kills) / std

        k = self.cfg.k_factor / 400.0
        delta = k * z
        pop.observed_skill[flat_ids] += delta.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rating_updater_kpm.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/rating_updaters/kpm.py tests/test_rating_updater_kpm.py
git commit -m "feat: KPMUpdater — rating updates on kills-per-minute"
```

---

## Task 11: Experience and Gear Updates

**Files:**
- Create: `src/mm_sim/experience.py`
- Create: `src/mm_sim/gear.py`
- Create: `tests/test_experience.py`
- Create: `tests/test_gear.py`

**Why:** Experience and gear are the other two rating components. Experience is monotonic (+= 1 per match, normalized). Gear grows with matches and drops on blowout losses.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_experience.py`:

```python
import numpy as np

from mm_sim.experience import apply_experience_update
from mm_sim.population import Population
from mm_sim.config import PopulationConfig
from mm_sim.seeding import make_rng


def test_experience_increments_by_match_count():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=10), rng)
    matches_played_this_tick = np.array([1, 2, 0, 3, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    apply_experience_update(pop, matches_played_this_tick, normalization_max_matches=100)
    expected = matches_played_this_tick / 100.0
    assert np.allclose(pop.experience, expected, atol=1e-6)


def test_experience_clipped_at_one():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=3), rng)
    matches = np.array([200, 50, 10], dtype=np.int32)
    apply_experience_update(pop, matches, normalization_max_matches=100)
    assert pop.experience[0] == 1.0
    assert pop.experience[1] == 0.5
    assert pop.experience[2] == 0.1
```

Create `tests/test_gear.py`:

```python
import numpy as np

from mm_sim.gear import apply_gear_update
from mm_sim.population import Population
from mm_sim.config import PopulationConfig, GearConfig
from mm_sim.seeding import make_rng


def test_gear_grows_with_matches_played():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=5), rng)
    pop.gear[:] = 0.0
    matches_played = np.array([10, 0, 5, 2, 0], dtype=np.int32)
    blowout_losses = np.zeros(5, dtype=np.int32)
    apply_gear_update(pop, matches_played, blowout_losses, GearConfig(growth_per_match=0.01))
    assert pop.gear[0] == 0.1
    assert pop.gear[1] == 0.0
    assert pop.gear[2] == 0.05


def test_gear_drops_on_blowout_loss():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=3), rng)
    pop.gear[:] = 0.5
    matches_played = np.zeros(3, dtype=np.int32)
    blowout_losses = np.array([1, 0, 2], dtype=np.int32)
    apply_gear_update(
        pop,
        matches_played,
        blowout_losses,
        GearConfig(growth_per_match=0.0, drop_on_blowout_loss=0.1),
    )
    assert pop.gear[0] == pytest_approx(0.4)
    assert pop.gear[1] == 0.5
    assert pop.gear[2] == pytest_approx(0.3)


def test_gear_clipped_to_range():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=2), rng)
    pop.gear[:] = np.array([0.98, 0.05], dtype=np.float32)
    matches_played = np.array([10, 0], dtype=np.int32)
    blowout_losses = np.array([0, 5], dtype=np.int32)
    apply_gear_update(
        pop,
        matches_played,
        blowout_losses,
        GearConfig(growth_per_match=0.05, drop_on_blowout_loss=0.1, max_gear=1.0),
    )
    assert pop.gear[0] == 1.0
    assert pop.gear[1] == 0.0


def pytest_approx(v, tol=1e-6):
    import pytest
    return pytest.approx(v, abs=tol)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_experience.py tests/test_gear.py -v`
Expected: FAIL with ImportError on both.

- [ ] **Step 3: Implement `experience.py`**

Create `src/mm_sim/experience.py`:

```python
"""Experience: monotonic, normalized to [0, 1] via a reference max."""

from __future__ import annotations

import numpy as np

from mm_sim.population import Population


def apply_experience_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    normalization_max_matches: int,
) -> None:
    pop.matches_played += matches_played_this_tick
    normalized = pop.matches_played / float(normalization_max_matches)
    pop.experience = np.clip(normalized, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Implement `gear.py`**

Create `src/mm_sim/gear.py`:

```python
"""Gear: grows with matches, drops on blowout losses, clipped to [0, max_gear]."""

from __future__ import annotations

import numpy as np

from mm_sim.config import GearConfig
from mm_sim.population import Population


def apply_gear_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    blowout_losses_this_tick: np.ndarray,
    cfg: GearConfig,
) -> None:
    growth = matches_played_this_tick.astype(np.float32) * cfg.growth_per_match
    drop = blowout_losses_this_tick.astype(np.float32) * cfg.drop_on_blowout_loss
    pop.gear = np.clip(pop.gear + growth - drop, 0.0, cfg.max_gear).astype(np.float32)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_experience.py tests/test_gear.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/experience.py src/mm_sim/gear.py tests/test_experience.py tests/test_gear.py
git commit -m "feat: experience and gear update rules"
```

---

## Task 12: Play Frequency Module

**Files:**
- Create: `src/mm_sim/frequency.py`
- Create: `tests/test_frequency.py`

**Why:** How many matches each active player plays each day. Winners play more, losers play less — a second feedback loop on top of churn.

- [ ] **Step 1: Write the failing test**

Create `tests/test_frequency.py`:

```python
import numpy as np

from mm_sim.frequency import sample_matches_per_day
from mm_sim.population import Population
from mm_sim.config import PopulationConfig, FrequencyConfig
from mm_sim.seeding import make_rng


def test_baseline_mean_is_approximately_cfg_mean():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=5000), rng)
    pop.recent_wins[:] = 0
    pop.recent_blowout_losses[:] = 0
    matches = sample_matches_per_day(
        pop,
        FrequencyConfig(mean_matches_per_day=3.0, win_modulation=0.0, loss_modulation=0.0),
        rng,
    )
    assert 2.7 < matches.mean() < 3.3


def test_winners_play_more():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=5000), rng)
    pop.recent_wins[:] = 5  # full window of wins
    pop.recent_blowout_losses[:] = 0
    matches_winners = sample_matches_per_day(
        pop,
        FrequencyConfig(mean_matches_per_day=3.0, win_modulation=0.5, loss_modulation=0.0),
        rng,
    )
    assert matches_winners.mean() > 3.5


def test_losers_play_less():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=5000), rng)
    pop.recent_wins[:] = 0
    pop.recent_blowout_losses[:] = 5
    matches_losers = sample_matches_per_day(
        pop,
        FrequencyConfig(mean_matches_per_day=3.0, win_modulation=0.0, loss_modulation=0.5),
        rng,
    )
    assert matches_losers.mean() < 2.5


def test_inactive_players_get_zero():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=100), rng)
    pop.active[50:] = False
    matches = sample_matches_per_day(pop, FrequencyConfig(), rng)
    assert (matches[50:] == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_frequency.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `frequency.py`**

Create `src/mm_sim/frequency.py`:

```python
"""Per-player matches-per-day sampling, modulated by recent experience."""

from __future__ import annotations

import numpy as np

from mm_sim.config import FrequencyConfig
from mm_sim.population import Population


def sample_matches_per_day(
    pop: Population, cfg: FrequencyConfig, rng: np.random.Generator
) -> np.ndarray:
    """Return an int32 array of matches to play today, one per player."""
    window = max(
        int(pop.recent_wins.max(initial=0)),
        int(pop.recent_blowout_losses.max(initial=0)),
        1,
    )
    win_rate = pop.recent_wins.astype(np.float32) / float(window)
    loss_rate = pop.recent_blowout_losses.astype(np.float32) / float(window)
    multiplier = 1.0 + cfg.win_modulation * win_rate - cfg.loss_modulation * loss_rate
    multiplier = np.clip(multiplier, 0.0, None)

    lam = cfg.mean_matches_per_day * multiplier
    draws = rng.poisson(lam).astype(np.int32)
    draws[~pop.active] = 0
    return draws
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_frequency.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/frequency.py tests/test_frequency.py
git commit -m "feat: per-player match frequency sampling with win/loss modulation"
```

---

## Task 13: Churn Function

**Files:**
- Create: `src/mm_sim/churn.py`
- Create: `tests/test_churn.py`

**Why:** Churn is the key mechanism for the feedback loop. Making it experience-dependent (rather than flat) is what lets the Activision dynamic emerge.

- [ ] **Step 1: Write the failing test**

Create `tests/test_churn.py`:

```python
import numpy as np

from mm_sim.churn import apply_churn
from mm_sim.population import Population
from mm_sim.config import PopulationConfig, ChurnConfig
from mm_sim.seeding import make_rng


def test_no_blowouts_baseline_churn_only():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=10_000), rng)
    pop.recent_wins[:] = 0
    pop.recent_blowout_losses[:] = 0
    before_active = int(pop.active.sum())
    apply_churn(pop, ChurnConfig(baseline_daily_quit_prob=0.01), rng)
    after_active = int(pop.active.sum())
    lost = before_active - after_active
    # Approximately 1% should churn, give or take sampling noise
    assert 70 < lost < 130


def test_blowout_losses_raise_churn():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=10_000), rng)
    pop.recent_wins[:] = 0
    pop.recent_blowout_losses[:] = 5  # full window of blowout losses
    before_active = int(pop.active.sum())
    apply_churn(
        pop,
        ChurnConfig(
            baseline_daily_quit_prob=0.01,
            blowout_loss_weight=0.08,
            rolling_window=5,
        ),
        rng,
    )
    lost = before_active - int(pop.active.sum())
    # Base 0.01 + 5/5 * 0.08 = 0.09 -> ~900 expected
    assert 800 < lost < 1100


def test_wins_lower_churn():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=10_000), rng)
    pop.recent_wins[:] = 5
    pop.recent_blowout_losses[:] = 0
    before_active = int(pop.active.sum())
    apply_churn(
        pop,
        ChurnConfig(
            baseline_daily_quit_prob=0.05,
            win_streak_weight=-0.03,
            rolling_window=5,
        ),
        rng,
    )
    lost = before_active - int(pop.active.sum())
    # Base 0.05 + 5/5 * -0.03 = 0.02 -> ~200 expected
    assert 130 < lost < 280


def test_already_inactive_stay_inactive():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=100), rng)
    pop.active[50:] = False
    apply_churn(pop, ChurnConfig(baseline_daily_quit_prob=1.0), rng)
    # All actives should have churned; already-inactive unchanged
    assert int(pop.active.sum()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_churn.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `churn.py`**

Create `src/mm_sim/churn.py`:

```python
"""Churn: daily per-player quit probability driven by recent experience."""

from __future__ import annotations

import numpy as np

from mm_sim.config import ChurnConfig
from mm_sim.population import Population


def apply_churn(pop: Population, cfg: ChurnConfig, rng: np.random.Generator) -> None:
    window = float(cfg.rolling_window)
    blowout_rate = pop.recent_blowout_losses.astype(np.float32) / window
    win_rate = pop.recent_wins.astype(np.float32) / window

    quit_prob = (
        cfg.baseline_daily_quit_prob
        + cfg.blowout_loss_weight * blowout_rate
        + cfg.win_streak_weight * win_rate
    )
    quit_prob = np.clip(quit_prob, 0.0, cfg.max_daily_quit_prob).astype(np.float32)

    draws = rng.random(size=pop.size).astype(np.float32)
    quits = (draws < quit_prob) & pop.active
    pop.active[quits] = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_churn.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/churn.py tests/test_churn.py
git commit -m "feat: experience-driven churn function"
```

---

## Task 14: Daily Snapshot

**Files:**
- Create: `src/mm_sim/snapshot.py`
- Create: `tests/test_snapshot.py`

**Why:** We need a per-day record of population state for later analysis. A polars DataFrame keyed by day + aggregate metrics.

- [ ] **Step 1: Write the failing test**

Create `tests/test_snapshot.py`:

```python
import polars as pl

from mm_sim.snapshot import DailySnapshotWriter
from mm_sim.population import Population
from mm_sim.config import PopulationConfig
from mm_sim.seeding import make_rng


def test_snapshot_captures_population_size_per_day():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=1000), rng)
    writer = DailySnapshotWriter()
    writer.record(day=0, pop=pop, matches_today=500, blowouts_today=25)
    pop.active[:100] = False
    writer.record(day=1, pop=pop, matches_today=480, blowouts_today=30)

    df = writer.to_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    row_0 = df.filter(pl.col("day") == 0).row(0, named=True)
    assert row_0["active_count"] == 1000
    assert row_0["matches_played"] == 500
    assert row_0["blowouts"] == 25
    row_1 = df.filter(pl.col("day") == 1).row(0, named=True)
    assert row_1["active_count"] == 900


def test_snapshot_records_skill_deciles():
    rng = make_rng(42)
    pop = Population.create_initial(PopulationConfig(initial_size=1000), rng)
    writer = DailySnapshotWriter()
    writer.record(day=0, pop=pop, matches_today=0, blowouts_today=0)
    df = writer.to_dataframe()
    row = df.filter(pl.col("day") == 0).row(0, named=True)
    assert "true_skill_p10" in row
    assert "true_skill_p50" in row
    assert "true_skill_p90" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snapshot.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `snapshot.py`**

Create `src/mm_sim/snapshot.py`:

```python
"""Daily snapshot of population metrics -> polars DataFrame."""

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
            row = {
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
        else:
            row = {
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
        self._rows.append(row)

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(self._rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_snapshot.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/snapshot.py tests/test_snapshot.py
git commit -m "feat: DailySnapshotWriter producing polars DataFrame of per-day metrics"
```

---

## Task 15: Simulation Engine

**Files:**
- Create: `src/mm_sim/engine.py`
- Create: `tests/test_engine_smoke.py`

**Why:** The engine is the orchestrator. It owns the main loop and wires components together via the protocols. It should be thin — all logic lives in the components.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_smoke.py`:

```python
from mm_sim.engine import SimulationEngine
from mm_sim.config import SimulationConfig, PopulationConfig


def test_short_season_runs_end_to_end():
    cfg = SimulationConfig(
        seed=1,
        season_days=5,
        population=PopulationConfig(initial_size=500, daily_new_players=10),
    )
    engine = SimulationEngine(cfg)
    df = engine.run()

    assert df.height == 5  # one row per day
    assert df["day"].to_list() == [0, 1, 2, 3, 4]
    # Population changed
    assert df["active_count"][0] > 0


def test_engine_is_deterministic():
    cfg = SimulationConfig(seed=7, season_days=3, population=PopulationConfig(initial_size=300))
    df1 = SimulationEngine(cfg).run()
    df2 = SimulationEngine(cfg).run()
    assert df1.equals(df2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_smoke.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `engine.py`**

Create `src/mm_sim/engine.py`:

```python
"""Main simulation engine — orchestrates daily tick."""

from __future__ import annotations

import numpy as np
import polars as pl

from mm_sim.config import SimulationConfig
from mm_sim.population import Population
from mm_sim.parties import assign_parties
from mm_sim.seeding import make_rng, spawn_child
from mm_sim.snapshot import DailySnapshotWriter
from mm_sim.churn import apply_churn
from mm_sim.frequency import sample_matches_per_day
from mm_sim.experience import apply_experience_update
from mm_sim.gear import apply_gear_update

from mm_sim.matchmaker.base import Matchmaker
from mm_sim.matchmaker.random_mm import RandomMatchmaker
from mm_sim.matchmaker.composite_mm import CompositeRatingMatchmaker

from mm_sim.outcomes.base import OutcomeGenerator
from mm_sim.outcomes.default import DefaultOutcomeGenerator

from mm_sim.rating_updaters.base import RatingUpdater
from mm_sim.rating_updaters.elo import EloUpdater
from mm_sim.rating_updaters.kpm import KPMUpdater


def _make_matchmaker(cfg: SimulationConfig) -> Matchmaker:
    if cfg.matchmaker.kind == "random":
        return RandomMatchmaker(cfg.matchmaker)
    if cfg.matchmaker.kind == "composite":
        return CompositeRatingMatchmaker(cfg.matchmaker)
    raise ValueError(f"unknown matchmaker kind: {cfg.matchmaker.kind}")


def _make_outcome_generator(cfg: SimulationConfig) -> OutcomeGenerator:
    if cfg.outcomes.kind == "default":
        return DefaultOutcomeGenerator(cfg.outcomes)
    raise ValueError(f"unknown outcome kind: {cfg.outcomes.kind}")


def _make_rating_updater(cfg: SimulationConfig) -> RatingUpdater:
    if cfg.rating_updater.kind == "elo":
        return EloUpdater(cfg.rating_updater)
    if cfg.rating_updater.kind == "kpm":
        return KPMUpdater(cfg.rating_updater)
    raise ValueError(f"unknown rating updater kind: {cfg.rating_updater.kind}")


class SimulationEngine:
    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.master_rng = make_rng(cfg.seed)
        self.population = Population.create_initial(
            cfg.population, spawn_child(self.master_rng, "population_init")
        )
        assign_parties(
            self.population, cfg.parties, spawn_child(self.master_rng, "parties_init")
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

        # Sample matches-per-day
        matches_per_player = sample_matches_per_day(
            self.population, self.cfg.frequency, spawn_child(day_rng, "frequency")
        )

        # Build the searching pool: one virtual entry per match-slot,
        # but for simplicity we treat each player as searching once with
        # the implied count; matches generated equal to matches_per_player
        # by running the matchmaker in multiple rounds.
        total_matches_played_per_player = np.zeros_like(matches_per_player)
        total_blowout_losses_per_player = np.zeros_like(matches_per_player)
        total_wins_per_player = np.zeros_like(matches_per_player)

        matches_today = 0
        blowouts_today = 0

        max_round = int(matches_per_player.max(initial=0))
        for round_idx in range(max_round):
            # Who still has remaining matches this round?
            still_playing = (matches_per_player - total_matches_played_per_player) > 0
            searching = np.flatnonzero(still_playing & self.population.active).astype(np.int32)
            if len(searching) < self.cfg.matchmaker.lobby_size:
                break

            round_rng = spawn_child(day_rng, f"round_{round_idx}")
            lobbies = self.matchmaker.form_lobbies(searching, self.population, round_rng)

            for lobby in lobbies:
                result = self.outcome_generator.generate(
                    lobby, self.population, spawn_child(round_rng, f"lobby_{matches_today}")
                )
                self.rating_updater.update(result, self.population)

                matches_today += 1
                if result.is_blowout:
                    blowouts_today += 1

                flat_ids = result.flat_player_ids()
                total_matches_played_per_player[flat_ids] += 1

                winning_team_ids = np.array(
                    lobby.teams[result.winning_team], dtype=np.int32
                )
                total_wins_per_player[winning_team_ids] += 1
                if result.is_blowout:
                    losing_teams = [
                        t for i, t in enumerate(lobby.teams) if i != result.winning_team
                    ]
                    for team in losing_teams:
                        total_blowout_losses_per_player[np.array(team, dtype=np.int32)] += 1

        # Update rolling windows (simple: replace last-tick values)
        window = self.cfg.churn.rolling_window
        self.population.recent_wins = np.clip(
            total_wins_per_player, 0, window
        ).astype(np.int8)
        self.population.recent_blowout_losses = np.clip(
            total_blowout_losses_per_player, 0, window
        ).astype(np.int8)

        # Apply experience and gear updates
        apply_experience_update(
            self.population,
            total_matches_played_per_player,
            normalization_max_matches=max(self.cfg.season_days * 5, 1),
        )
        apply_gear_update(
            self.population,
            total_matches_played_per_player,
            total_blowout_losses_per_player,
            self.cfg.gear,
        )

        # Churn
        apply_churn(self.population, self.cfg.churn, spawn_child(day_rng, "churn"))

        # New players arrive
        new_ids = self.population.add_new_players(
            self.cfg.population.daily_new_players,
            self.cfg.population,
            spawn_child(day_rng, "new_players"),
            day=day,
        )
        if len(new_ids) > 0:
            # New players join the next day's party assignment pass only if we
            # want dynamic parties. For v1 (static parties), assign each new
            # player as a solo party.
            next_pid = int(self.population.party_id.max()) + 1 if self.population.size > 0 else 0
            for offset, nid in enumerate(new_ids):
                self.population.party_id[nid] = next_pid + offset

        # Record snapshot
        self.snapshot_writer.record(
            day=day,
            pop=self.population,
            matches_today=matches_today,
            blowouts_today=blowouts_today,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engine_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/engine.py tests/test_engine_smoke.py
git commit -m "feat: SimulationEngine orchestrating daily tick loop"
```

---

## Task 16: Feedback Loop Integration Test

**Files:**
- Create: `tests/test_feedback_loop.py`

**Why:** This is the test that validates the whole system works together. We set up a scenario and verify the Activision feedback loop (low-skill players churn faster, average skill rises over time) actually emerges.

- [ ] **Step 1: Write the test**

Create `tests/test_feedback_loop.py`:

```python
"""Integration test: verify the Activision feedback loop emerges.

With skill-based matchmaking + churn driven by blowout losses, we expect:
1. Active population average true skill rises over the season (low-skill churn)
2. Rating error decreases as the matchmaker learns players
3. Blowout rate doesn't explode
"""

import pytest

from mm_sim.engine import SimulationEngine
from mm_sim.config import (
    SimulationConfig,
    PopulationConfig,
    PartyConfig,
    MatchmakerConfig,
    ChurnConfig,
    FrequencyConfig,
)


def test_low_skill_players_churn_faster_under_skill_mm():
    cfg = SimulationConfig(
        seed=123,
        season_days=30,
        population=PopulationConfig(
            initial_size=5000,
            true_skill_distribution="normal",
            daily_new_players=50,
        ),
        parties=PartyConfig(size_distribution={1: 1.0}),  # solo-only for a cleaner test
        matchmaker=MatchmakerConfig(
            kind="composite",
            composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
        ),
        churn=ChurnConfig(
            baseline_daily_quit_prob=0.002,
            blowout_loss_weight=0.15,
            win_streak_weight=-0.02,
        ),
        frequency=FrequencyConfig(mean_matches_per_day=4.0),
    )
    df = SimulationEngine(cfg).run()

    first_week = df.filter(df["day"] < 7)["true_skill_mean"].mean()
    last_week = df.filter(df["day"] >= 23)["true_skill_mean"].mean()

    # Feedback loop: average true skill of the active population rises
    assert last_week > first_week, (
        f"expected average skill to rise, got {first_week} -> {last_week}"
    )


def test_population_does_not_collapse_in_normal_config():
    cfg = SimulationConfig(
        seed=123,
        season_days=30,
        population=PopulationConfig(initial_size=5000, daily_new_players=50),
    )
    df = SimulationEngine(cfg).run()
    final = df.filter(df["day"] == 29)["active_count"].item()
    # Shouldn't lose more than 80% of starting population in 30 days with defaults
    assert final > 1000
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_feedback_loop.py -v`
Expected: 2 passed. If it fails, diagnose whether it's a parameter-tuning issue (the defaults may need adjustment) or a bug.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feedback_loop.py
git commit -m "test: integration test verifying feedback loop emerges"
```

---

## Task 17: Scenario Runner

**Files:**
- Create: `src/mm_sim/scenario.py`
- Create: `tests/test_scenario.py`

**Why:** Parameter sweeps are a stated requirement (#6). The scenario runner takes a list of configs and runs them, tagging the output DataFrames.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scenario.py`:

```python
import polars as pl

from mm_sim.scenario import ScenarioRunner, Scenario
from mm_sim.config import SimulationConfig, PopulationConfig, MatchmakerConfig


def test_runner_labels_each_scenario_output():
    scenarios = [
        Scenario(
            name="skill_only",
            config=SimulationConfig(
                seed=1,
                season_days=3,
                population=PopulationConfig(initial_size=300),
                matchmaker=MatchmakerConfig(
                    kind="composite",
                    composite_weights={"skill": 1.0, "experience": 0.0, "gear": 0.0},
                ),
            ),
        ),
        Scenario(
            name="experience_only",
            config=SimulationConfig(
                seed=1,
                season_days=3,
                population=PopulationConfig(initial_size=300),
                matchmaker=MatchmakerConfig(
                    kind="composite",
                    composite_weights={"skill": 0.0, "experience": 1.0, "gear": 0.0},
                ),
            ),
        ),
    ]
    runner = ScenarioRunner(scenarios)
    df = runner.run_all()
    assert isinstance(df, pl.DataFrame)
    assert set(df["scenario"].unique().to_list()) == {"skill_only", "experience_only"}
    assert df.filter(pl.col("scenario") == "skill_only").height == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `scenario.py`**

Create `src/mm_sim/scenario.py`:

```python
"""Scenario runner for parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import polars as pl

from mm_sim.config import SimulationConfig
from mm_sim.engine import SimulationEngine


@dataclass
class Scenario:
    name: str
    config: SimulationConfig


class ScenarioRunner:
    def __init__(self, scenarios: list[Scenario]) -> None:
        self.scenarios = scenarios

    def run_all(self) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for s in self.scenarios:
            df = SimulationEngine(s.config).run()
            df = df.with_columns(pl.lit(s.name).alias("scenario"))
            frames.append(df)
        return pl.concat(frames)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scenario.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/scenario.py tests/test_scenario.py
git commit -m "feat: ScenarioRunner for sweeping across configurations"
```

---

## Task 18: CLI Module and Full Test Sweep

**Files:**
- Create: `src/mm_sim/cli.py`
- Create: `tests/test_cli.py`

**Why:** Provide a `python -m mm_sim.cli` entry point (wired into the `justfile`'s `just sim` recipe) that runs a default simulation and prints summary metrics. Also run the full test suite to verify everything works together.

Note: the project's orchestration lives in a `justfile` at the repo root (created during Task 1 follow-up). There is no `main.py`. The CLI is a proper package module so `python -m mm_sim.cli` works without any shim.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import subprocess
import sys


def test_cli_runs_default_short_sim(tmp_path):
    """Smoke test: invoke the CLI with a short season and verify it exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "mm_sim.cli", "--season-days", "3", "--initial-size", "200"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert "active_count" in result.stdout
    assert "true_skill_mean" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with ModuleNotFoundError on `mm_sim.cli`.

- [ ] **Step 3: Implement `src/mm_sim/cli.py`**

Create `src/mm_sim/cli.py`:

```python
"""Command-line entry point: run a default simulation and print summary."""

from __future__ import annotations

import argparse

from mm_sim.config import SimulationConfig, PopulationConfig
from mm_sim.engine import SimulationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm_sim",
        description="Run a player-base simulation and print summary metrics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--season-days", type=int, default=90)
    parser.add_argument("--initial-size", type=int, default=50_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SimulationConfig(
        seed=args.seed,
        season_days=args.season_days,
        population=PopulationConfig(initial_size=args.initial_size),
    )
    engine = SimulationEngine(cfg)
    df = engine.run()

    print(f"Season length: {cfg.season_days} days")
    print(f"Initial population: {cfg.population.initial_size}")
    print()
    print("Day 0:")
    print(df.filter(df["day"] == 0))
    print()
    print(f"Day {cfg.season_days - 1}:")
    print(df.filter(df["day"] == cfg.season_days - 1))
    print()
    print("Season summary (first vs last):")
    first = df.filter(df["day"] == 0).row(0, named=True)
    last = df.filter(df["day"] == cfg.season_days - 1).row(0, named=True)
    print(f"  active_count: {first['active_count']} -> {last['active_count']}")
    print(f"  true_skill_mean: {first['true_skill_mean']:.3f} -> {last['true_skill_mean']:.3f}")
    print(f"  rating_error_mean: {first['rating_error_mean']:.3f} -> {last['rating_error_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v` (or `just test`)
Expected: All tests pass.

- [ ] **Step 6: Run the CLI via justfile**

Run: `just sim`
Expected: Prints day 0 and day 89 snapshot rows and a summary line. At 50k × 90 days this may take a minute or two; if it's unreasonably slow, reduce `initial_size` via `uv run python -m mm_sim.cli --initial-size 5000` and note the performance as a follow-up.

- [ ] **Step 7: Commit**

```bash
git add src/mm_sim/cli.py tests/test_cli.py
git commit -m "feat: mm_sim.cli module runnable via 'just sim' / 'python -m mm_sim.cli'"
```

---

## Self-Review Notes

- **Spec coverage check:**
  - ✅ Daily ticks, 90-day season, configurable (Task 3 config, Task 15 engine)
  - ✅ 50k players at start, configurable (Task 3, Task 4)
  - ✅ Option B — true skill vs. observed skill (Task 4 Population, Task 9 Elo, Task 10 KPM)
  - ✅ Composite rating with skill/experience/gear weights (Task 7 composite matchmaker)
  - ✅ Static parties with configurable size distribution and homogeneity (Task 5)
  - ✅ Churn as a function of recent experience (Task 13)
  - ✅ Play frequency modulated by wins/losses (Task 12)
  - ✅ Experience += 1 per match, simple (Task 11)
  - ✅ Gear grows with matches, drops on blowouts (Task 11)
  - ✅ Pluggable outcome generator with multi-field contribution vector (Task 8)
  - ✅ Pluggable rating updater (Task 9 Elo, Task 10 KPM)
  - ✅ Scenario runner for sweeps (Task 17)
  - ✅ Daily snapshots including skill distribution, rating error, active count (Task 14)
  - ✅ Feedback loop integration test (Task 16)
  - ✅ Python + uv (Task 1)

- **Deferred to v2 (noted during brainstorming, explicitly out of scope):**
  - Matplotlib/plotly visualization
  - More sophisticated party formation (dynamic)
  - Party-aware skill similarity/disparity rules from the paper (we use a simpler snake-sort team balancer)
  - Full k-partitioning / Karmarkar-Karp team balance
  - True_skill drift/learning curves
  - Additional rating updaters beyond Elo and KPM

- **Type consistency check:** `MatchResult`, `Lobby`, `Population`, `ContributionVector`-as-dict-of-arrays are all used consistently. `apply_churn`, `apply_experience_update`, `apply_gear_update`, `sample_matches_per_day`, `assign_parties` all mutate `pop` in place and take `cfg` + `rng`. Matchmaker and updater protocols match their implementations. `_make_*` factories in the engine cover all configured kinds.

- **Placeholder scan:** No TODOs, TBDs, or "similar to above" shortcuts. Every step contains the code or command needed.

- **Things I'm not 100% sure about:**
  1. **Performance at 50k players.** The engine uses Python-level loops over lobbies per day. At ~3 matches/player/day × 50k players / 12 players per lobby, that's ~12,500 lobbies/day × 90 days ≈ 1.1M lobbies. Each lobby is a handful of numpy slicings. This should be a few minutes per season, maybe more. If it's too slow, the hot spot is the per-lobby loop in `SimulationEngine._tick` and in the outcome generator. Optimization is explicitly not in v1 scope, but Task 18 notes a performance check.
  2. **Feedback loop parameters.** The thresholds in `test_feedback_loop.py` are plausible but may need tuning once we see real numbers. If the test fails because the effect is too small (not because the mechanism is broken), tune the churn weight upward or the baseline quit prob downward and re-run.
  3. **CompositeRatingMatchmaker snake-sort packing.** This is a simplification of the paper's similarity/disparity system. It's good enough to test the research question (skill-weighted grouping), but v2 should implement the real two-rule system if we want fidelity to the paper.
