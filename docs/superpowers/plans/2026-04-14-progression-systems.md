# Progression Systems Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three orthogonal per-player progression systems — **skill progression**, **gear progression (outcome-driven)**, and **season progression** — so scenarios can evaluate how matchmakers behave under realistic season-long player development dynamics.

**Architecture:** Each system is an independently toggleable module (`skill_progression.py`, gear rework in existing `gear.py`, new `season_progression.py`) that updates a per-player numpy array each tick. All three default to **disabled** so existing scenarios behave identically. Each has its own `*Config` section in `SimulationConfig`. Season progression adds a new churn term via a small extension to `apply_churn`; skill and gear do not touch churn.

**Tech Stack:** Python 3, numpy, pydantic (config), polars (snapshots), pytest.

---

## Design Summary (locked decisions)

### Skill progression
- New array: `talent_ceiling` (drawn once at player creation from same distribution as initial `true_skill`, fixed for life).
- Initial `true_skill = starting_true_skill_fraction * talent_ceiling` (default 0.3).
- Per-tick drift toward ceiling: `delta = (ceiling - true_skill) * (matches_this_tick / tau) + noise`, where noise scales by `sqrt(matches_this_tick)`.
- Strict asymptote: `true_skill` clipped at `ceiling`.
- Defaults: `tau=75`, `noise_std=0.02`, `enabled=False`.
- Match outcomes do NOT drive true skill — the rating updater continues to move *observed* skill based on wins/losses.

### Gear progression
- Keep baseline drift (`growth_per_match`) but retune down by default when transfer is on.
- **New: outcome-based transfer.** Each match, losing team members transfer a fraction of gear to winning team members. Blowouts transfer more.
- Transfer amount per losing player = `transfer_rate * gear_losing_player` (base) or `transfer_rate_blowout * gear_losing_player` (blowout). Winners split the total equally.
- Existing `drop_on_blowout_loss` is removed — subsumed by the transfer model (the loser's drop IS the winner's gain).
- Soft ceiling via existing `max_gear` clip.
- Defaults: `transfer_rate=0.01`, `transfer_rate_blowout=0.04`, transfer `enabled=False`.

### Season progression
- New array: `season_progress ∈ [0, 1]`.
- Earned per match played (flat `earn_per_match`), outcome-independent. Capped at 1.0.
- Global expected curve: `expected(day) = 1 - exp(-curve_steepness * day / season_days)`. Diminishing returns.
- Two-sided churn pressure added to `apply_churn`:
  - `gap = expected(day) - season_progress`.
  - If `gap > 0`: quit_prob increases by `behind_weight * gap` (falling behind pressure).
  - If `gap < 0` AND `day / season_days < boredom_cutoff`: quit_prob increases by `boredom_weight * (-gap)` (maxed-out-early pressure).
- Does NOT reset at season end (this plan simulates one season).
- Defaults: `earn_per_match=0.02`, `curve_steepness=3.0`, `behind_weight=0.02`, `boredom_weight=0.01`, `boredom_cutoff=0.7`, `enabled=False`.

### Shared infrastructure
- All three systems add their array to `Population` (created in `create_initial`, extended in `add_new_players`, persisted in `record_population` snapshots).
- All three have `enabled: bool = False` on their config.
- Each ships with an on/off scenario pair for end-to-end validation.

---

## File Structure

**Create:**
- `src/mm_sim/skill_progression.py`
- `src/mm_sim/season_progression.py`
- `tests/test_config_progression.py`
- `tests/test_skill_progression.py`
- `tests/test_gear_transfer.py`
- `tests/test_season_progression.py`
- `tests/test_engine_progression.py`
- `scenarios/skill_progression_on.toml`, `scenarios/skill_progression_off.toml`
- `scenarios/gear_transfer_on.toml`, `scenarios/gear_transfer_off.toml`
- `scenarios/season_progression_on.toml`, `scenarios/season_progression_off.toml`
- `scenarios/all_progression_on.toml` (all three enabled together)

**Modify:**
- `src/mm_sim/config.py` — add `SkillProgressionConfig`, `SeasonProgressionConfig`; extend `GearConfig`; extend `PopulationConfig`; extend `ChurnConfig`; attach all to `SimulationConfig`.
- `src/mm_sim/population.py` — add `talent_ceiling`, `season_progress` arrays; update `create_initial` and `add_new_players`.
- `src/mm_sim/gear.py` — add outcome-based transfer; remove `drop_on_blowout_loss`.
- `src/mm_sim/churn.py` — add season-progression pressure term (only active when season progression is enabled).
- `src/mm_sim/engine.py` — pass match-outcome data into gear update; call `apply_skill_progression_update` and `apply_season_progression_update` each tick; pass current day and season length into churn.
- `src/mm_sim/snapshot.py` — persist `talent_ceiling` and `season_progress`.
- `tests/test_population.py` — verify new arrays shape/extension.

---

## Phase 1: Skill progression

### Task 1: Add `SkillProgressionConfig` + `starting_true_skill_fraction`

**Files:**
- Modify: `src/mm_sim/config.py`
- Test: `tests/test_config_progression.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_progression.py`:
```python
"""Tests for progression config schemas (skill, gear transfer, season)."""

from __future__ import annotations

import pytest

from mm_sim.config import (
    GearConfig,
    PopulationConfig,
    SeasonProgressionConfig,
    SimulationConfig,
    SkillProgressionConfig,
)


def test_skill_progression_defaults_disabled():
    cfg = SimulationConfig()
    assert cfg.skill_progression.enabled is False
    assert cfg.skill_progression.tau == 75.0
    assert cfg.skill_progression.noise_std == 0.02
    assert cfg.skill_progression.starting_true_skill_fraction == 0.3


def test_skill_progression_tau_must_be_positive():
    with pytest.raises(Exception):
        SkillProgressionConfig(tau=0.0)


def test_skill_progression_fraction_in_unit_interval():
    with pytest.raises(Exception):
        SkillProgressionConfig(starting_true_skill_fraction=-0.1)
    with pytest.raises(Exception):
        SkillProgressionConfig(starting_true_skill_fraction=1.5)


def test_population_starting_true_skill_fraction_default():
    cfg = PopulationConfig()
    assert cfg.starting_true_skill_fraction == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_progression.py -v`
Expected: FAIL with `ImportError: cannot import name 'SkillProgressionConfig'`.

- [ ] **Step 3: Add `SkillProgressionConfig` and wire into `SimulationConfig`**

In `src/mm_sim/config.py`, after `GearConfig` (around line 99), add:
```python
class SkillProgressionConfig(BaseModel):
    """Per-tick true_skill drift toward a per-player talent ceiling."""

    enabled: bool = False
    tau: float = Field(75.0, gt=0.0)
    noise_std: float = Field(0.02, ge=0.0)
    starting_true_skill_fraction: float = Field(0.3, ge=0.0, le=1.0)
```

In `PopulationConfig`, after `starting_gear`, add:
```python
    starting_true_skill_fraction: float = Field(0.3, ge=0.0, le=1.0)
```

In `SimulationConfig`, after `gear: GearConfig = ...`, add:
```python
    skill_progression: SkillProgressionConfig = Field(
        default_factory=SkillProgressionConfig
    )
```

Note: we keep the default on `PopulationConfig.starting_true_skill_fraction` too, since population construction reads it directly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_progression.py -v`
Expected: PASS (the 4 tests added so far).

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -x`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/config.py tests/test_config_progression.py
git commit -m "feat(config): add SkillProgressionConfig (disabled by default)"
```

---

### Task 2: Add `talent_ceiling` to `Population`

**Files:**
- Modify: `src/mm_sim/population.py`
- Test: `tests/test_population.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_population.py`:
```python
def test_population_has_talent_ceiling():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=100)
    pop = Population.create_initial(cfg, np.random.default_rng(42))
    assert pop.talent_ceiling.shape == pop.true_skill.shape
    assert pop.talent_ceiling.dtype == np.float32
    assert pop.talent_ceiling.std() > 0


def test_initial_true_skill_is_fraction_of_ceiling():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=500, starting_true_skill_fraction=0.3)
    pop = Population.create_initial(cfg, np.random.default_rng(1))
    np.testing.assert_allclose(pop.true_skill, pop.talent_ceiling * 0.3, rtol=1e-5)


def test_add_new_players_extends_talent_ceiling():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=10)
    pop = Population.create_initial(cfg, np.random.default_rng(7))
    before = pop.talent_ceiling.shape[0]
    pop.add_new_players(5, cfg, np.random.default_rng(8), day=3)
    assert pop.talent_ceiling.shape[0] == before + 5
    assert pop.talent_ceiling.shape == pop.true_skill.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_population.py -v -k "talent_ceiling or fraction_of_ceiling"`
Expected: FAIL with `AttributeError: 'Population' object has no attribute 'talent_ceiling'`.

- [ ] **Step 3: Add `talent_ceiling` to `Population`**

In `src/mm_sim/population.py`:

Add to dataclass fields (after `true_skill`):
```python
    talent_ceiling: np.ndarray          # per-player ceiling for true_skill drift
```

Replace the body of `create_initial` to draw ceiling first, then compute starting skill:
```python
    @classmethod
    def create_initial(
        cls, cfg: PopulationConfig, rng: np.random.Generator
    ) -> "Population":
        n = cfg.initial_size
        talent_ceiling = _sample_skill(n, cfg, rng).astype(np.float32)
        fraction = cfg.starting_true_skill_fraction
        true_skill = (talent_ceiling * fraction).astype(np.float32)
        return cls(
            true_skill=true_skill,
            talent_ceiling=talent_ceiling,
            observed_skill=np.full(n, cfg.starting_observed_skill, dtype=np.float32),
            experience=np.full(n, cfg.starting_experience, dtype=np.float32),
            gear=np.full(n, cfg.starting_gear, dtype=np.float32),
            active=np.ones(n, dtype=bool),
            party_id=np.full(n, -1, dtype=np.int32),
            matches_played=np.zeros(n, dtype=np.int32),
            recent_wins=np.zeros(n, dtype=np.int8),
            recent_losses=np.zeros(n, dtype=np.int8),
            loss_streak=np.zeros(n, dtype=np.int8),
            recent_blowout_losses=np.zeros(n, dtype=np.int8),
            join_day=np.zeros(n, dtype=np.int32),
        )
```

In `add_new_players`, change the first two lines to draw ceiling first:
```python
        new_ceiling = _sample_skill(count, cfg, rng).astype(np.float32)
        new_true = (new_ceiling * cfg.starting_true_skill_fraction).astype(np.float32)
        start = self.size
        self.true_skill = np.concatenate([self.true_skill, new_true])
        self.talent_ceiling = np.concatenate([self.talent_ceiling, new_ceiling])
```
Leave remaining concatenates unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_population.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -x`
Expected: PASS. If any scenario regression test asserts exact numeric skill outputs, note that initial `true_skill` is now `0.3 * ceiling` rather than a direct draw. If such a test fails, set `starting_true_skill_fraction=1.0` in that scenario's config to preserve old behavior.

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/population.py tests/test_population.py
git commit -m "feat(population): add talent_ceiling; initial true_skill = fraction * ceiling"
```

---

### Task 3: Implement `skill_progression.py`

**Files:**
- Create: `src/mm_sim/skill_progression.py`
- Test: `tests/test_skill_progression.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_skill_progression.py`:
```python
"""Tests for skill progression (per-tick true_skill drift toward talent ceiling)."""

from __future__ import annotations

import numpy as np

from mm_sim.config import PopulationConfig, SkillProgressionConfig
from mm_sim.population import Population
from mm_sim.skill_progression import apply_skill_progression_update


def _make_pop(n: int = 1000, seed: int = 0) -> Population:
    cfg = PopulationConfig(initial_size=n)
    return Population.create_initial(cfg, np.random.default_rng(seed))


def test_disabled_is_noop():
    pop = _make_pop()
    before = pop.true_skill.copy()
    cfg = SkillProgressionConfig(enabled=False)
    matches = np.ones(pop.size, dtype=np.int32) * 5
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(0))
    np.testing.assert_array_equal(pop.true_skill, before)


def test_enabled_moves_true_skill_toward_ceiling_zero_noise():
    pop = _make_pop(n=5000, seed=1)
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.0)
    matches = np.full(pop.size, 10, dtype=np.int32)
    gap_before = pop.talent_ceiling - pop.true_skill
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(2))
    gap_after = pop.talent_ceiling - pop.true_skill
    assert np.all(gap_after < gap_before)
    expected_delta = gap_before * 10.0 / 75.0
    actual_delta = (pop.true_skill - (pop.talent_ceiling - gap_before)).astype(np.float32)
    np.testing.assert_allclose(actual_delta, expected_delta, rtol=1e-4)


def test_players_with_zero_matches_do_not_change():
    pop = _make_pop()
    before = pop.true_skill.copy()
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.1)
    matches = np.zeros(pop.size, dtype=np.int32)
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(3))
    np.testing.assert_array_equal(pop.true_skill, before)


def test_true_skill_clipped_at_ceiling():
    pop = _make_pop(n=100, seed=4)
    pop.true_skill = pop.talent_ceiling.copy()
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.5)
    matches = np.full(pop.size, 100, dtype=np.int32)
    apply_skill_progression_update(pop, matches, cfg, np.random.default_rng(5))
    assert np.all(pop.true_skill <= pop.talent_ceiling + 1e-6)


def test_noise_produces_variance_across_runs():
    pop_a = _make_pop(n=2000, seed=9)
    pop_b = _make_pop(n=2000, seed=9)
    cfg = SkillProgressionConfig(enabled=True, tau=75.0, noise_std=0.05)
    matches = np.full(pop_a.size, 5, dtype=np.int32)
    apply_skill_progression_update(pop_a, matches, cfg, np.random.default_rng(100))
    apply_skill_progression_update(pop_b, matches, cfg, np.random.default_rng(200))
    assert not np.allclose(pop_a.true_skill, pop_b.true_skill)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_skill_progression.py -v`
Expected: FAIL with `ImportError: No module named 'mm_sim.skill_progression'`.

- [ ] **Step 3: Implement `skill_progression.py`**

Create `src/mm_sim/skill_progression.py`:
```python
"""True-skill progression: per-match drift toward talent_ceiling.

Discrete analogue of dx/dn = (ceiling - x) / tau, which integrates to
x(n) = ceiling - (ceiling - x0) * exp(-n / tau). Applied per tick so noise
accumulates naturally. No-op when disabled.
"""

from __future__ import annotations

import numpy as np

from mm_sim.config import SkillProgressionConfig
from mm_sim.population import Population


def apply_skill_progression_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    cfg: SkillProgressionConfig,
    rng: np.random.Generator,
) -> None:
    if not cfg.enabled:
        return

    matches = matches_played_this_tick.astype(np.float32)
    gap = pop.talent_ceiling - pop.true_skill
    deterministic = gap * (matches / cfg.tau)

    if cfg.noise_std > 0.0:
        noise = rng.normal(
            loc=0.0,
            scale=cfg.noise_std * np.sqrt(np.maximum(matches, 0.0)),
            size=pop.size,
        ).astype(np.float32)
    else:
        noise = np.zeros(pop.size, dtype=np.float32)

    played_mask = matches > 0
    delta = np.where(played_mask, deterministic + noise, 0.0).astype(np.float32)
    new_skill = pop.true_skill + delta
    pop.true_skill = np.minimum(new_skill, pop.talent_ceiling).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_skill_progression.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/skill_progression.py tests/test_skill_progression.py
git commit -m "feat(skill): add per-tick true_skill progression toward talent ceiling"
```

---

### Task 4: Wire skill progression into engine + snapshots + scenarios

**Files:**
- Modify: `src/mm_sim/engine.py`, `src/mm_sim/snapshot.py`
- Create: `tests/test_engine_progression.py`, `scenarios/skill_progression_on.toml`, `scenarios/skill_progression_off.toml`

- [ ] **Step 1: Write the failing engine test**

Create `tests/test_engine_progression.py`:
```python
"""End-to-end progression tests (enabled vs disabled affects mean metrics)."""

from __future__ import annotations

from mm_sim.config import (
    PopulationConfig,
    SimulationConfig,
    SkillProgressionConfig,
)
from mm_sim.engine import SimulationEngine


def _cfg(skill_enabled: bool) -> SimulationConfig:
    return SimulationConfig(
        seed=123,
        season_days=5,
        population=PopulationConfig(
            initial_size=500, daily_new_player_fraction=0.0
        ),
        skill_progression=SkillProgressionConfig(
            enabled=skill_enabled, tau=50.0, noise_std=0.0
        ),
    )


def test_skill_progression_enabled_raises_mean_true_skill():
    engine = SimulationEngine(_cfg(skill_enabled=True))
    initial = float(engine.population.true_skill.mean())
    engine.run()
    final = float(engine.population.true_skill.mean())
    assert final > initial


def test_skill_progression_disabled_leaves_mean_true_skill_static():
    engine = SimulationEngine(_cfg(skill_enabled=False))
    initial = float(engine.population.true_skill.mean())
    engine.run()
    final = float(engine.population.true_skill.mean())
    assert abs(final - initial) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_progression.py -v`
Expected: FAIL on enabled case.

- [ ] **Step 3: Wire skill progression and snapshot into engine**

Edit `src/mm_sim/engine.py`:

Add import near the other imports:
```python
from mm_sim.skill_progression import apply_skill_progression_update
```

In `_tick`, directly after the existing `apply_gear_update(...)` block, add:
```python
        apply_skill_progression_update(
            self.population,
            total_matches,
            self.cfg.skill_progression,
            spawn_child(day_rng, "skill_progression"),
        )
```

In `src/mm_sim/snapshot.py`, inside `record_population`, add to the dict (next to `true_skill`):
```python
                "talent_ceiling": pop.talent_ceiling.copy(),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engine_progression.py tests/test_skill_progression.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -x`
Expected: PASS.

- [ ] **Step 6: Create validation scenarios**

Create `scenarios/skill_progression_off.toml`:
```toml
name = "skill_progression_off"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.skill_progression]
enabled = false
```

Create `scenarios/skill_progression_on.toml`:
```toml
name = "skill_progression_on"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.skill_progression]
enabled = true
tau = 75.0
noise_std = 0.02
```

- [ ] **Step 7: Run both scenarios**

Run:
```bash
just scenario skill_progression_off
just scenario skill_progression_on
```
Expected: Both complete without error.

- [ ] **Step 8: Commit**

```bash
git add src/mm_sim/engine.py src/mm_sim/snapshot.py tests/test_engine_progression.py scenarios/skill_progression_on.toml scenarios/skill_progression_off.toml
git commit -m "feat(engine): invoke skill_progression + persist talent_ceiling; add on/off scenarios"
```

---

## Phase 2: Gear progression rework (outcome-based transfer)

### Task 5: Extend `GearConfig` for transfer model

**Files:**
- Modify: `src/mm_sim/config.py`
- Test: `tests/test_config_progression.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_progression.py`:
```python
def test_gear_transfer_defaults_disabled():
    cfg = SimulationConfig()
    assert cfg.gear.transfer_enabled is False
    assert cfg.gear.transfer_rate == 0.01
    assert cfg.gear.transfer_rate_blowout == 0.04


def test_gear_transfer_rates_nonnegative():
    with pytest.raises(Exception):
        GearConfig(transfer_rate=-0.1)
    with pytest.raises(Exception):
        GearConfig(transfer_rate_blowout=-0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_progression.py -v -k transfer`
Expected: FAIL — attributes don't exist.

- [ ] **Step 3: Extend `GearConfig`**

In `src/mm_sim/config.py`, replace `GearConfig` with:
```python
class GearConfig(BaseModel):
    # Baseline drift: small gear gain per match played, regardless of outcome.
    growth_per_match: float = Field(0.005, ge=0.0)
    max_gear: float = Field(1.0, gt=0.0)
    # Outcome-based transfer: when enabled, losing-team members transfer a
    # fraction of their gear to winning-team members each match.
    transfer_enabled: bool = False
    transfer_rate: float = Field(0.01, ge=0.0)
    transfer_rate_blowout: float = Field(0.04, ge=0.0)
    # Legacy: direct drop on blowout loss. Kept for backwards-compat but only
    # applies when transfer_enabled is False.
    drop_on_blowout_loss: float = Field(0.05, ge=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_progression.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -x`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/config.py tests/test_config_progression.py
git commit -m "feat(config): add gear transfer model fields (disabled by default)"
```

---

### Task 6: Implement gear transfer in engine tick

Gear transfer must run *inside the match loop* (it needs winner/loser team IDs and blowout status), unlike the current `apply_gear_update` which runs once per tick with aggregate counts. We'll add a helper `apply_gear_transfer_for_match` that runs per-match when transfer is enabled, and keep `apply_gear_update` for baseline drift and legacy drop.

**Files:**
- Modify: `src/mm_sim/gear.py`, `src/mm_sim/engine.py`
- Test: `tests/test_gear_transfer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gear_transfer.py`:
```python
"""Tests for per-match gear transfer from losers to winners."""

from __future__ import annotations

import numpy as np

from mm_sim.config import GearConfig, PopulationConfig
from mm_sim.gear import apply_gear_transfer_for_match
from mm_sim.population import Population


def _make_pop(n: int, gear_values: np.ndarray) -> Population:
    pop = Population.create_initial(
        PopulationConfig(initial_size=n, starting_gear=0.5),
        np.random.default_rng(0),
    )
    pop.gear = gear_values.astype(np.float32)
    return pop


def test_transfer_disabled_is_noop():
    pop = _make_pop(6, np.full(6, 0.5, dtype=np.float32))
    before = pop.gear.copy()
    cfg = GearConfig(transfer_enabled=False)
    apply_gear_transfer_for_match(
        pop, winners=np.array([0, 1, 2]), losers=np.array([3, 4, 5]),
        is_blowout=False, cfg=cfg,
    )
    np.testing.assert_array_equal(pop.gear, before)


def test_transfer_moves_gear_from_losers_to_winners():
    pop = _make_pop(6, np.array([0.3, 0.3, 0.3, 0.6, 0.6, 0.6], dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.1)
    total_before = pop.gear.sum()
    apply_gear_transfer_for_match(
        pop, winners=np.array([0, 1, 2]), losers=np.array([3, 4, 5]),
        is_blowout=False, cfg=cfg,
    )
    total_after = pop.gear.sum()
    # Transfer is internal; total (up to clipping) is preserved.
    np.testing.assert_allclose(total_after, total_before, atol=1e-5)
    # Losers went down, winners went up.
    assert np.all(pop.gear[:3] > 0.3)
    assert np.all(pop.gear[3:] < 0.6)


def test_blowout_transfers_more_than_regular():
    pop_a = _make_pop(6, np.full(6, 0.5, dtype=np.float32))
    pop_b = _make_pop(6, np.full(6, 0.5, dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.01, transfer_rate_blowout=0.04)
    apply_gear_transfer_for_match(
        pop_a, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=False, cfg=cfg
    )
    apply_gear_transfer_for_match(
        pop_b, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=True, cfg=cfg
    )
    assert pop_b.gear[0] > pop_a.gear[0]
    assert pop_b.gear[3] < pop_a.gear[3]


def test_transfer_clipped_to_max_gear():
    pop = _make_pop(6, np.array([0.98, 0.98, 0.98, 0.5, 0.5, 0.5], dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.5, max_gear=1.0)
    apply_gear_transfer_for_match(
        pop, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=False, cfg=cfg
    )
    assert np.all(pop.gear <= 1.0 + 1e-6)


def test_transfer_clipped_at_zero():
    pop = _make_pop(6, np.array([0.5, 0.5, 0.5, 0.01, 0.01, 0.01], dtype=np.float32))
    cfg = GearConfig(transfer_enabled=True, transfer_rate=0.5)
    apply_gear_transfer_for_match(
        pop, np.array([0, 1, 2]), np.array([3, 4, 5]), is_blowout=False, cfg=cfg
    )
    assert np.all(pop.gear >= 0.0 - 1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gear_transfer.py -v`
Expected: FAIL — `apply_gear_transfer_for_match` doesn't exist.

- [ ] **Step 3: Implement `apply_gear_transfer_for_match`**

In `src/mm_sim/gear.py`, add (after `apply_gear_update`):
```python
def apply_gear_transfer_for_match(
    pop: Population,
    winners: np.ndarray,
    losers: np.ndarray,
    is_blowout: bool,
    cfg: GearConfig,
) -> None:
    """Transfer a fraction of each loser's gear to the winners (split equally).

    No-op when transfer_enabled is False. Blowouts use the higher rate.
    """
    if not cfg.transfer_enabled:
        return
    if len(winners) == 0 or len(losers) == 0:
        return

    rate = cfg.transfer_rate_blowout if is_blowout else cfg.transfer_rate
    if rate <= 0.0:
        return

    loser_gear = pop.gear[losers]
    loss = (loser_gear * rate).astype(np.float32)
    total_transferred = float(loss.sum())

    pop.gear[losers] = np.clip(loser_gear - loss, 0.0, cfg.max_gear).astype(np.float32)
    per_winner = total_transferred / float(len(winners))
    pop.gear[winners] = np.clip(
        pop.gear[winners] + per_winner, 0.0, cfg.max_gear
    ).astype(np.float32)
```

- [ ] **Step 4: Update `apply_gear_update` to skip legacy drop when transfer is enabled**

In `src/mm_sim/gear.py`, replace the existing `apply_gear_update` with:
```python
def apply_gear_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    blowout_losses_this_tick: np.ndarray,
    cfg: GearConfig,
) -> None:
    growth = matches_played_this_tick.astype(np.float32) * cfg.growth_per_match
    # Legacy drop only applies when transfer is disabled. When transfer is on,
    # blowout gear effects are handled per-match inside the transfer function.
    if cfg.transfer_enabled:
        drop = np.zeros_like(growth)
    else:
        drop = blowout_losses_this_tick.astype(np.float32) * cfg.drop_on_blowout_loss
    pop.gear = np.clip(
        pop.gear + growth - drop, 0.0, cfg.max_gear
    ).astype(np.float32)
```

- [ ] **Step 5: Wire the per-match transfer call into the engine**

In `src/mm_sim/engine.py`, add import:
```python
from mm_sim.gear import apply_gear_update, apply_gear_transfer_for_match
```

Inside the inner match loop in `_tick`, directly after the rating updater call (`self.rating_updater.update(result, self.population)`), add:
```python
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
```

(Note: an existing `winning_team_ids` variable is computed a few lines later in the current code — that's fine, the new `winners_arr` is a local variable scoped to this block. Do not remove the existing `winning_team_ids` assignment; it's used for `total_wins` and `loss_streak` resets.)

- [ ] **Step 6: Run all gear tests**

Run: `uv run pytest tests/test_gear_transfer.py tests/test_engine_smoke.py -v`
Expected: PASS.

- [ ] **Step 7: Run full suite**

Run: `uv run pytest -x`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mm_sim/gear.py src/mm_sim/engine.py tests/test_gear_transfer.py
git commit -m "feat(gear): add outcome-based per-match gear transfer (disabled by default)"
```

---

### Task 7: Gear transfer validation scenarios

**Files:**
- Create: `scenarios/gear_transfer_on.toml`, `scenarios/gear_transfer_off.toml`

- [ ] **Step 1: Create `scenarios/gear_transfer_off.toml`**

```toml
name = "gear_transfer_off"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.gear]
transfer_enabled = false
```

- [ ] **Step 2: Create `scenarios/gear_transfer_on.toml`**

```toml
name = "gear_transfer_on"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.gear]
transfer_enabled = true
transfer_rate = 0.01
transfer_rate_blowout = 0.04
growth_per_match = 0.002
```

- [ ] **Step 3: Run both**

Run:
```bash
just scenario gear_transfer_off
just scenario gear_transfer_on
```
Expected: Both complete without error.

- [ ] **Step 4: Verify gear distribution is wider when transfer is on**

Run: `just compare gear_transfer_off gear_transfer_on`
Expected: Completes. In the "on" scenario, gear distribution spread (std) among survivors should be larger by end of season than in the "off" scenario.

- [ ] **Step 5: Commit**

```bash
git add scenarios/gear_transfer_on.toml scenarios/gear_transfer_off.toml
git commit -m "feat(scenarios): add gear_transfer on/off validation pair"
```

---

## Phase 3: Season progression

### Task 8: Add `SeasonProgressionConfig` + churn fields

**Files:**
- Modify: `src/mm_sim/config.py`
- Test: `tests/test_config_progression.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config_progression.py`:
```python
def test_season_progression_defaults_disabled():
    cfg = SimulationConfig()
    assert cfg.season_progression.enabled is False
    assert cfg.season_progression.earn_per_match == 0.02
    assert cfg.season_progression.curve_steepness == 3.0
    assert cfg.season_progression.behind_weight == 0.02
    assert cfg.season_progression.boredom_weight == 0.01
    assert cfg.season_progression.boredom_cutoff == 0.7


def test_season_progression_earn_rate_nonnegative():
    with pytest.raises(Exception):
        SeasonProgressionConfig(earn_per_match=-0.01)


def test_season_progression_boredom_cutoff_in_unit():
    with pytest.raises(Exception):
        SeasonProgressionConfig(boredom_cutoff=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_progression.py -v -k season`
Expected: FAIL — `SeasonProgressionConfig` doesn't exist.

- [ ] **Step 3: Add `SeasonProgressionConfig`**

In `src/mm_sim/config.py`, after `SkillProgressionConfig`, add:
```python
class SeasonProgressionConfig(BaseModel):
    """Per-player season pass progress and its churn pressure."""

    enabled: bool = False
    earn_per_match: float = Field(0.02, ge=0.0)
    # Expected curve: expected(d) = 1 - exp(-curve_steepness * d/season_days)
    curve_steepness: float = Field(3.0, gt=0.0)
    # Churn additions when player is behind expected progression.
    behind_weight: float = Field(0.02, ge=0.0)
    # Churn additions when player is ahead (maxed out early) AND day/season < cutoff.
    boredom_weight: float = Field(0.01, ge=0.0)
    boredom_cutoff: float = Field(0.7, ge=0.0, le=1.0)
```

In `SimulationConfig`, after `skill_progression: ...`, add:
```python
    season_progression: SeasonProgressionConfig = Field(
        default_factory=SeasonProgressionConfig
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_progression.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/config.py tests/test_config_progression.py
git commit -m "feat(config): add SeasonProgressionConfig (disabled by default)"
```

---

### Task 9: Add `season_progress` array to `Population`

**Files:**
- Modify: `src/mm_sim/population.py`
- Test: `tests/test_population.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_population.py`:
```python
def test_population_has_season_progress():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=50)
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    assert pop.season_progress.shape == (50,)
    assert pop.season_progress.dtype == np.float32
    assert np.all(pop.season_progress == 0.0)


def test_add_new_players_extends_season_progress():
    import numpy as np
    from mm_sim.config import PopulationConfig
    from mm_sim.population import Population

    cfg = PopulationConfig(initial_size=10)
    pop = Population.create_initial(cfg, np.random.default_rng(0))
    pop.add_new_players(5, cfg, np.random.default_rng(1), day=2)
    assert pop.season_progress.shape == (15,)
    assert np.all(pop.season_progress[-5:] == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_population.py -v -k season_progress`
Expected: FAIL — attribute missing.

- [ ] **Step 3: Add `season_progress` to `Population`**

In `src/mm_sim/population.py`:

Add to dataclass fields (after `join_day`):
```python
    season_progress: np.ndarray         # per-player [0, 1] season-pass progress
```

In `create_initial`'s return, add:
```python
            season_progress=np.zeros(n, dtype=np.float32),
```

In `add_new_players`, add before the final return:
```python
        self.season_progress = np.concatenate(
            [self.season_progress, np.zeros(count, dtype=np.float32)]
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_population.py -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -x`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mm_sim/population.py tests/test_population.py
git commit -m "feat(population): add season_progress array"
```

---

### Task 10: Implement `season_progression.py`

**Files:**
- Create: `src/mm_sim/season_progression.py`
- Test: `tests/test_season_progression.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_season_progression.py`:
```python
"""Tests for season progression: per-match earning and churn-pressure term."""

from __future__ import annotations

import numpy as np

from mm_sim.config import PopulationConfig, SeasonProgressionConfig
from mm_sim.population import Population
from mm_sim.season_progression import (
    apply_season_progression_update,
    expected_progress,
    season_churn_pressure,
)


def _make_pop(n: int = 100) -> Population:
    return Population.create_initial(
        PopulationConfig(initial_size=n), np.random.default_rng(0)
    )


def test_disabled_is_noop():
    pop = _make_pop()
    before = pop.season_progress.copy()
    cfg = SeasonProgressionConfig(enabled=False)
    matches = np.ones(pop.size, dtype=np.int32) * 5
    apply_season_progression_update(pop, matches, cfg)
    np.testing.assert_array_equal(pop.season_progress, before)


def test_earn_per_match_accumulates():
    pop = _make_pop(n=10)
    cfg = SeasonProgressionConfig(enabled=True, earn_per_match=0.05)
    matches = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    apply_season_progression_update(pop, matches, cfg)
    np.testing.assert_allclose(pop.season_progress, matches * 0.05, rtol=1e-5)


def test_earn_clipped_at_one():
    pop = _make_pop(n=3)
    cfg = SeasonProgressionConfig(enabled=True, earn_per_match=0.5)
    matches = np.full(3, 10, dtype=np.int32)
    apply_season_progression_update(pop, matches, cfg)
    assert np.all(pop.season_progress <= 1.0 + 1e-6)


def test_expected_progress_monotone_saturating():
    cfg = SeasonProgressionConfig(curve_steepness=3.0)
    values = [expected_progress(d, season_days=90, cfg=cfg) for d in range(0, 91, 10)]
    assert values[0] == 0.0
    for a, b in zip(values, values[1:]):
        assert b > a
    assert values[-1] < 1.0  # not fully saturated at d=90 unless steepness is huge
    assert values[-1] > 0.9


def test_churn_pressure_zero_when_disabled():
    cfg = SeasonProgressionConfig(enabled=False)
    progress = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    pressure = season_churn_pressure(progress, day=30, season_days=90, cfg=cfg)
    np.testing.assert_array_equal(pressure, np.zeros(3, dtype=np.float32))


def test_churn_pressure_behind_positive():
    cfg = SeasonProgressionConfig(
        enabled=True, behind_weight=0.02, boredom_weight=0.0, curve_steepness=3.0
    )
    expected = expected_progress(60, season_days=90, cfg=cfg)
    progress = np.array([expected - 0.2, expected, expected + 0.2], dtype=np.float32)
    pressure = season_churn_pressure(progress, day=60, season_days=90, cfg=cfg)
    assert pressure[0] > 0  # behind
    assert pressure[1] == 0  # on curve
    assert pressure[2] == 0  # ahead (but boredom_weight=0)


def test_churn_pressure_boredom_only_before_cutoff():
    cfg = SeasonProgressionConfig(
        enabled=True, behind_weight=0.0, boredom_weight=0.05,
        boredom_cutoff=0.7, curve_steepness=3.0,
    )
    # Day 10/90 = 0.11 (before cutoff) and day 80/90 = 0.89 (after cutoff).
    progress = np.array([1.0], dtype=np.float32)
    early = season_churn_pressure(progress, day=10, season_days=90, cfg=cfg)
    late = season_churn_pressure(progress, day=80, season_days=90, cfg=cfg)
    assert early[0] > 0
    assert late[0] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_season_progression.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `season_progression.py`**

Create `src/mm_sim/season_progression.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_season_progression.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/season_progression.py tests/test_season_progression.py
git commit -m "feat(season): add season_progression module with two-sided churn pressure"
```

---

### Task 11: Integrate season pressure into churn

**Files:**
- Modify: `src/mm_sim/churn.py`, `src/mm_sim/engine.py`
- Test: extend `tests/test_season_progression.py`

`apply_churn` currently takes `(pop, cfg, rng)`. We need it to also receive the current day, season length, and season-progression config. We extend the signature with keyword args defaulting to None so older tests that call it directly still work.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_season_progression.py`:
```python
def test_churn_uses_season_pressure_when_enabled():
    import numpy as np
    from mm_sim.churn import apply_churn
    from mm_sim.config import ChurnConfig, PopulationConfig, SeasonProgressionConfig
    from mm_sim.population import Population

    # Two identical populations; one with season pressure, one without.
    cfg_pop = PopulationConfig(initial_size=2000, starting_true_skill_fraction=1.0)
    pop_no_pressure = Population.create_initial(cfg_pop, np.random.default_rng(0))
    pop_with_pressure = Population.create_initial(cfg_pop, np.random.default_rng(0))

    # Make everyone have 0 progress while expected is ~0.6 — huge gap.
    pop_no_pressure.season_progress[:] = 0.0
    pop_with_pressure.season_progress[:] = 0.0

    churn_cfg = ChurnConfig(baseline_daily_quit_prob=0.0)
    season_cfg_on = SeasonProgressionConfig(
        enabled=True, behind_weight=0.5, curve_steepness=3.0
    )
    season_cfg_off = SeasonProgressionConfig(enabled=False)

    apply_churn(
        pop_no_pressure, churn_cfg, np.random.default_rng(1),
        day=30, season_days=90, season_cfg=season_cfg_off,
    )
    apply_churn(
        pop_with_pressure, churn_cfg, np.random.default_rng(1),
        day=30, season_days=90, season_cfg=season_cfg_on,
    )

    alive_no_pressure = int(pop_no_pressure.active.sum())
    alive_with_pressure = int(pop_with_pressure.active.sum())
    assert alive_with_pressure < alive_no_pressure
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_season_progression.py::test_churn_uses_season_pressure_when_enabled -v`
Expected: FAIL — `apply_churn` doesn't accept these kwargs.

- [ ] **Step 3: Extend `apply_churn` signature**

In `src/mm_sim/churn.py`, replace the function signature and body:
```python
from mm_sim.config import ChurnConfig, SeasonProgressionConfig
from mm_sim.population import Population
from mm_sim.season_progression import season_churn_pressure


def apply_churn(
    pop: Population,
    cfg: ChurnConfig,
    rng: np.random.Generator,
    *,
    day: int = 0,
    season_days: int = 1,
    season_cfg: SeasonProgressionConfig | None = None,
) -> None:
    window = float(cfg.rolling_window)
    loss_rate = pop.recent_losses.astype(np.float32) / window
    loss_streak = pop.loss_streak.astype(np.float32)
    blowout_rate = pop.recent_blowout_losses.astype(np.float32) / window
    win_rate = pop.recent_wins.astype(np.float32) / window

    threshold = float(cfg.new_player_threshold)
    newness = np.clip(
        1.0 - pop.matches_played.astype(np.float32) / threshold, 0.0, 1.0
    )
    sensitivity = (1.0 + cfg.new_player_bonus * newness).astype(np.float32)

    loss_streak_multiplier = np.exp(cfg.loss_streak_exp * loss_streak) - 1.0
    loss_streak_multiplier = np.clip(
        loss_streak_multiplier, 0.0, cfg.max_loss_streak_multiplier
    ).astype(np.float32)
    loss_streak_factor = 1.0 + loss_streak_multiplier

    loss_term = cfg.loss_weight * (loss_rate ** 2) * loss_streak_factor
    blowout_term = cfg.blowout_loss_weight * blowout_rate

    if season_cfg is not None:
        season_term = season_churn_pressure(
            pop.season_progress, day=day, season_days=season_days, cfg=season_cfg
        )
    else:
        season_term = np.zeros(pop.size, dtype=np.float32)

    quit_prob = np.clip(
        cfg.baseline_daily_quit_prob
        + sensitivity * (loss_term + blowout_term)
        + cfg.win_streak_weight * win_rate
        + season_term,
        0.0,
        cfg.max_daily_quit_prob,
    ).astype(np.float32)

    draws = rng.random(size=pop.size).astype(np.float32)
    quits = (draws < quit_prob) & pop.active
    pop.active[quits] = False
```

(Remove the existing top-of-file import of `ChurnConfig` if it's now duplicated — only import each symbol once.)

- [ ] **Step 4: Update engine to pass new args**

In `src/mm_sim/engine.py`, replace the existing `apply_churn(...)` call in `_tick` with:
```python
        apply_churn(
            self.population,
            self.cfg.churn,
            spawn_child(day_rng, "churn"),
            day=day,
            season_days=self.cfg.season_days,
            season_cfg=self.cfg.season_progression,
        )
```

Also add the season progression update call right before churn (after skill progression):
```python
        from mm_sim.season_progression import apply_season_progression_update
        apply_season_progression_update(
            self.population, total_matches, self.cfg.season_progression
        )
```
(Move the import to the top of the file with the others — `from mm_sim.season_progression import apply_season_progression_update` — don't leave it inline.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_season_progression.py tests/test_engine_smoke.py tests/test_engine_progression.py -v`
Expected: PASS.

- [ ] **Step 6: Run full suite**

Run: `uv run pytest -x`
Expected: PASS. If any test calls `apply_churn` without the new kwargs and assumes season pressure is zero, that still works because `season_cfg` defaults to `None`.

- [ ] **Step 7: Commit**

```bash
git add src/mm_sim/churn.py src/mm_sim/engine.py src/mm_sim/season_progression.py tests/test_season_progression.py
git commit -m "feat(churn): add season-progress pressure term; wire season_progression into engine"
```

---

### Task 12: Persist `season_progress` in snapshots

**Files:**
- Modify: `src/mm_sim/snapshot.py`
- Test: add assertion to `tests/test_engine_progression.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engine_progression.py`:
```python
def test_snapshot_contains_season_progress_and_talent_ceiling():
    from mm_sim.config import SeasonProgressionConfig

    cfg = SimulationConfig(
        seed=7, season_days=3,
        population=PopulationConfig(initial_size=100),
        skill_progression=SkillProgressionConfig(enabled=True),
        season_progression=SeasonProgressionConfig(enabled=True),
    )
    engine = SimulationEngine(cfg)
    engine.run()
    frames = engine.snapshot_writer.population_frames
    for frame in frames:
        assert "talent_ceiling" in frame.columns
        assert "season_progress" in frame.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engine_progression.py::test_snapshot_contains_season_progress_and_talent_ceiling -v`
Expected: FAIL — `season_progress` not in columns.

- [ ] **Step 3: Add `season_progress` to snapshot**

In `src/mm_sim/snapshot.py`, inside `record_population`, add next to the `talent_ceiling` entry:
```python
                "season_progress": pop.season_progress.copy(),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engine_progression.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/snapshot.py tests/test_engine_progression.py
git commit -m "feat(snapshot): persist season_progress in population frames"
```

---

### Task 13: Season progression validation scenarios

**Files:**
- Create: `scenarios/season_progression_on.toml`, `scenarios/season_progression_off.toml`

- [ ] **Step 1: Create `scenarios/season_progression_off.toml`**

```toml
name = "season_progression_off"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.season_progression]
enabled = false
```

- [ ] **Step 2: Create `scenarios/season_progression_on.toml`**

```toml
name = "season_progression_on"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.season_progression]
enabled = true
earn_per_match = 0.02
curve_steepness = 3.0
behind_weight = 0.02
boredom_weight = 0.01
boredom_cutoff = 0.7
```

- [ ] **Step 3: Run both and compare**

Run:
```bash
just scenario season_progression_off
just scenario season_progression_on
just compare season_progression_off season_progression_on
```
Expected: Both complete. With season pressure on, late-season churn should visibly increase for players who fell behind the curve.

- [ ] **Step 4: Commit**

```bash
git add scenarios/season_progression_on.toml scenarios/season_progression_off.toml
git commit -m "feat(scenarios): add season_progression on/off validation pair"
```

---

## Phase 4: Combined scenario + final validation

### Task 14: All-progression-on scenario

**Files:**
- Create: `scenarios/all_progression_on.toml`

- [ ] **Step 1: Create `scenarios/all_progression_on.toml`**

```toml
name = "all_progression_on"

[config.parties]
size_distribution = {1 = 1.0}

[config.matchmaker]
kind = "composite"
composite_weights = {skill = 1.0, experience = 0.0, gear = 0.0}

[config.skill_progression]
enabled = true
tau = 75.0
noise_std = 0.02

[config.gear]
transfer_enabled = true
transfer_rate = 0.01
transfer_rate_blowout = 0.04
growth_per_match = 0.002

[config.season_progression]
enabled = true
earn_per_match = 0.02
curve_steepness = 3.0
behind_weight = 0.02
boredom_weight = 0.01
boredom_cutoff = 0.7
```

- [ ] **Step 2: Run it**

Run: `just scenario all_progression_on`
Expected: Completes without error.

- [ ] **Step 3: Commit**

```bash
git add scenarios/all_progression_on.toml
git commit -m "feat(scenarios): add all_progression_on combined scenario"
```

---

### Task 15: Final full-suite verification

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest`
Expected: PASS across all modules.

- [ ] **Step 2: List scenarios**

Run: `just scenarios-list`
Expected: Output includes all 7 new scenarios (3 on/off pairs + `all_progression_on`).

- [ ] **Step 3: Run the full scenario batch**

Run: `just scenarios`
Expected: All scenarios complete without error.

- [ ] **Step 4: No commit — verification only**

If anything failed, go back and fix it. Do not claim the plan complete until all three steps pass cleanly.

---

## Out of scope (explicitly deferred)

- **Multi-season simulation** (season resets, gear decay, carry-over rules).
- **Scenario sweep / analysis notebooks** comparing matchmakers under full progression.
- **Observed-skill tracking tuning** — the existing Elo/KPM updaters continue to track the (now-moving) true skill; retuning them if they lag is a future concern.
- **Party-level progression effects** — parties currently share no progression state; if we want "dragging friends into matches they'll lose" dynamics, that's a separate plan.
- **Engagement-optimized matchmaking** — a new matchmaker kind that uses progression state to feed winnable matches to select players. Can be built on top of this infrastructure later.
