# Softmax Extraction Outcome Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-team independent-threshold extraction model with a softmax-over-strength model that samples a variable number of winners per match, while preserving the per-team `expected_extract` semantics the engine and calibration chart rely on.

**Architecture:**
- New pure helper `plackett_luce_marginals(strengths, k, beta)` computes each team's probability of being among the top-k winners under sampling-without-replacement from `softmax(beta * strengths)`.
- The outcome generator samples `k` (number of extractors per match) from a three-component mixture that decouples the target mean from the P(k=0) and P(k=n_teams) tails.
- `strength_sensitivity` is repurposed as the softmax temperature β; default raised to 2.0 (empirically closest to the prior normal-CDF behavior across lobby widths).
- `baseline_extract_prob` is removed; `mean_extractors_per_match`, `p_zero_extract`, `p_all_extract` replace it.
- The MM calibration chart design (Elo-view vs outcome-view) is unchanged — the `floor` line continues to capture the intentional updater-vs-outcome formula mismatch. Only the stored `expected_extract` values change (from normal-CDF to Plackett-Luce marginals).

**Tech Stack:** Python 3.14, numpy, polars, pydantic, pytest, uv. Project uses `uv run` for every command.

---

## File Structure

**New files:**
- `src/mm_sim/outcomes/softmax_winners.py` — pure helper containing `sample_extractor_count()` and `plackett_luce_marginals()`. Kept separate from `extraction.py` so both are independently testable and the helpers can be reused by the dashboard.
- `tests/test_softmax_winners.py` — unit tests for the two helpers.

**Modified files:**
- `src/mm_sim/config.py` — replace `baseline_extract_prob` with `mean_extractors_per_match`, `p_zero_extract`, `p_all_extract`; bump `strength_sensitivity` default to 2.0.
- `src/mm_sim/outcomes/extraction.py` — rewrite `ExtractionOutcomeGenerator.generate` to use the new helpers.
- `scenarios/defaults.toml` — drop `baseline_extract_prob`, set new knobs.
- `src/debug/debug_rating.py` — update the one `OutcomeConfig(...)` call site.
- `tests/test_extraction_outcomes.py` — rewrite the tests against the new model.
- `tests/test_config_progression.py` — update the one assertion about `OutcomeConfig` defaults.

---

## Task 1: Add `plackett_luce_marginals` helper with tests

**Files:**
- Create: `src/mm_sim/outcomes/softmax_winners.py`
- Test: `tests/test_softmax_winners.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_softmax_winners.py`:

```python
"""Tests for softmax-based winner sampling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.outcomes.softmax_winners import (
    plackett_luce_marginals,
    sample_extractor_count,
)


def test_marginals_sum_to_k():
    strengths = np.array([0.2, 0.5, 0.8, 1.1], dtype=np.float32)
    for k in range(5):
        m = plackett_luce_marginals(strengths, k=k, beta=2.0)
        assert m.shape == (4,)
        assert pytest.approx(m.sum(), abs=1e-5) == float(k)


def test_marginals_stronger_team_higher():
    """With beta>0, a stronger team should have a higher marginal."""
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=2, beta=2.0)
    # Strictly increasing in strength
    assert m[0] < m[1] < m[2] < m[3]


def test_marginals_beta_zero_is_uniform():
    """beta=0 means strengths are ignored; every team has marginal k/n."""
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=2, beta=0.0)
    np.testing.assert_allclose(m, [0.5, 0.5, 0.5, 0.5], atol=1e-6)


def test_marginals_k_zero_all_zero():
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=0, beta=2.0)
    np.testing.assert_allclose(m, [0.0, 0.0, 0.0, 0.0])


def test_marginals_k_equals_n_all_one():
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=4, beta=2.0)
    np.testing.assert_allclose(m, [1.0, 1.0, 1.0, 1.0])


def test_marginals_hand_computed_values():
    """Reference values computed out of band:
    strengths = [0.2, 0.5, 0.8, 1.1], beta=1.0, k=2.
    Expected marginals (exact enumeration): [0.335, 0.437, 0.556, 0.672]."""
    strengths = np.array([0.2, 0.5, 0.8, 1.1], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=2, beta=1.0)
    np.testing.assert_allclose(
        m, [0.335, 0.437, 0.556, 0.672], atol=1e-3
    )


def test_sample_extractor_count_mean():
    """Over many draws, the sample mean should match the target."""
    rng = np.random.default_rng(42)
    draws = [
        sample_extractor_count(
            n_teams=4,
            mean_extractors=1.8,
            p_zero=0.01,
            p_all=0.03,
            rng=rng,
        )
        for _ in range(20_000)
    ]
    arr = np.array(draws)
    # Mean within 0.03 of target (large N, tight tolerance)
    assert abs(arr.mean() - 1.8) < 0.03


def test_sample_extractor_count_tail_probs():
    """P(k=0) and P(k=n_teams) should match the knobs."""
    rng = np.random.default_rng(42)
    draws = np.array([
        sample_extractor_count(
            n_teams=4,
            mean_extractors=1.8,
            p_zero=0.01,
            p_all=0.03,
            rng=rng,
        )
        for _ in range(50_000)
    ])
    p0 = float((draws == 0).mean())
    pn = float((draws == 4).mean())
    assert abs(p0 - 0.01) < 0.005
    assert abs(pn - 0.03) < 0.008


def test_sample_extractor_count_support_in_range():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        k = sample_extractor_count(
            n_teams=4, mean_extractors=1.8, p_zero=0.01, p_all=0.03, rng=rng,
        )
        assert 0 <= k <= 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_softmax_winners.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mm_sim.outcomes.softmax_winners'`.

- [ ] **Step 3: Implement `softmax_winners.py`**

Create `src/mm_sim/outcomes/softmax_winners.py`:

```python
"""Softmax-based winner-sampling helpers for extraction outcomes.

Two pure functions, no state:

- ``sample_extractor_count`` draws k (teams that extract this match) from a
  three-component mixture: a small P(k=0) spike, a small P(k=n_teams) spike,
  and a shifted Binomial in between, parameterized to hit a target mean.

- ``plackett_luce_marginals`` computes, for each team, the probability that
  it is among the top-k winners when k teams are sampled without replacement
  from ``softmax(beta * strengths)``. Exact via ordering enumeration; cheap
  for ``n_teams <= 8``.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def sample_extractor_count(
    n_teams: int,
    mean_extractors: float,
    p_zero: float,
    p_all: float,
    rng: np.random.Generator,
) -> int:
    """Draw k, the number of extracting teams this match.

    Three-component mixture:
      - with probability ``p_zero``: k = 0
      - with probability ``p_all``:  k = n_teams
      - with the remaining mass:     k ~ Binomial(n_teams - 1, p) + 1

    The Binomial ``p`` is solved so that the overall expected k equals
    ``mean_extractors``.
    """
    r = rng.random()
    if r < p_zero:
        return 0
    if r < p_zero + p_all:
        return n_teams
    p_mid_mass = 1.0 - p_zero - p_all
    if p_mid_mass <= 0.0:
        # Degenerate config; fall back to 1 to keep the sim running.
        return 1
    mid_mean = (mean_extractors - p_all * n_teams) / p_mid_mass
    # Binomial(n-1, p) + 1 has mean 1 + (n-1)*p.
    p = (mid_mean - 1.0) / (n_teams - 1)
    p = max(0.0, min(1.0, p))
    return int(rng.binomial(n_teams - 1, p)) + 1


def plackett_luce_marginals(
    strengths: np.ndarray,
    k: int,
    beta: float,
) -> np.ndarray:
    """Per-team probability of being among the top-k winners.

    Winners are sampled without replacement from ``softmax(beta * strengths)``.
    This function exactly enumerates every ordered length-k pick.

    Args:
        strengths: 1-D array of per-team strengths (any real values).
        k: number of winners (0 <= k <= n_teams).
        beta: softmax temperature; 0 = uniform, larger = stronger teams win more.

    Returns:
        float64 array with shape ``(n_teams,)``; entries in [0, 1] summing to k.
    """
    n = int(strengths.shape[0])
    if k <= 0:
        return np.zeros(n, dtype=np.float64)
    if k >= n:
        return np.ones(n, dtype=np.float64)

    # Exponentiated weights, with a max-shift for numerical stability.
    shifted = beta * strengths.astype(np.float64)
    shifted = shifted - shifted.max()
    w = np.exp(shifted)

    marginals = np.zeros(n, dtype=np.float64)
    for ordering in permutations(range(n), k):
        remaining_mask = np.ones(n, dtype=bool)
        prob = 1.0
        for pick in ordering:
            total = float(w[remaining_mask].sum())
            prob *= float(w[pick]) / total
            remaining_mask[pick] = False
        for t in ordering:
            marginals[t] += prob
    return marginals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_softmax_winners.py -v`
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/outcomes/softmax_winners.py tests/test_softmax_winners.py
git commit -m "feat(outcomes): add softmax winner-sampling helpers

sample_extractor_count draws k from a three-component mixture
(p_zero, p_all, shifted binomial), solving binomial p to hit a
target mean. plackett_luce_marginals returns per-team probability
of being among the top-k via exact ordering enumeration."
```

---

## Task 2: Update `OutcomeConfig` with new fields; drop `baseline_extract_prob`

**Files:**
- Modify: `src/mm_sim/config.py:88-97`
- Modify: `tests/test_config_progression.py:77-81`

- [ ] **Step 1: Update the failing test to reflect the new shape**

Replace the body of the relevant test in `tests/test_config_progression.py` — find the block that currently reads:

```python
    cfg = OutcomeConfig(kind="extraction")
    assert cfg.kind == "extraction"
    assert cfg.baseline_extract_prob == 0.4
    assert cfg.strength_sensitivity == 1.0
```

with:

```python
    cfg = OutcomeConfig(kind="extraction")
    assert cfg.kind == "extraction"
    assert cfg.mean_extractors_per_match == 1.8
    assert cfg.p_zero_extract == 0.01
    assert cfg.p_all_extract == 0.03
    assert cfg.strength_sensitivity == 2.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config_progression.py -v`
Expected: FAIL — existing `baseline_extract_prob` default is 0.4 and the new fields don't exist yet.

- [ ] **Step 3: Update `OutcomeConfig`**

In `src/mm_sim/config.py`, replace the `OutcomeConfig` class body:

```python
class OutcomeConfig(BaseModel):
    kind: str = Field("default", pattern="^(default|extraction)$")
    noise_std: float = 0.25
    blowout_threshold: float = 30.0
    # How much gear contributes to match performance alongside true_skill.
    # 0.0 = gear is cosmetic (default, preserves pre-existing scenarios).
    # Nonzero = effective_rating = true_skill + gear_weight * gear + noise.
    gear_weight: float = Field(0.0, ge=0.0)
    # Softmax-extract parameters (see outcomes.softmax_winners).
    # mean_extractors_per_match: target mean number of extractors per match.
    # p_zero_extract / p_all_extract: mass at the k=0 / k=n_teams tails.
    mean_extractors_per_match: float = Field(1.8, gt=0.0)
    p_zero_extract: float = Field(0.01, ge=0.0, le=1.0)
    p_all_extract: float = Field(0.03, ge=0.0, le=1.0)
    # Softmax temperature beta. Higher = stronger team dominates more.
    strength_sensitivity: float = Field(2.0, gt=0.0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_config_progression.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/config.py tests/test_config_progression.py
git commit -m "change(config): replace baseline_extract_prob with mixture knobs

OutcomeConfig now exposes mean_extractors_per_match, p_zero_extract,
and p_all_extract for the softmax-extract model. strength_sensitivity
default bumped to 2.0 (softmax temperature tuned to approximate the
old normal-CDF model across lobby widths). baseline_extract_prob
removed."
```

---

## Task 3: Rewrite `ExtractionOutcomeGenerator` to use softmax winners

**Files:**
- Modify: `src/mm_sim/outcomes/extraction.py` (full rewrite of `__init__` and `generate`)
- Modify: `tests/test_extraction_outcomes.py` (rewrite tests against the new model)

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_extraction_outcomes.py` with:

```python
"""Tests for the softmax-based extraction outcome generator."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig, PopulationConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.extraction import ExtractionOutcomeGenerator
from mm_sim.population import Population


def _pop_with_skills(
    skills: list[float], gears: list[float] | None = None
) -> Population:
    pop_cfg = PopulationConfig(initial_size=len(skills))
    pop = Population.create_initial(pop_cfg, np.random.default_rng(0))
    pop.true_skill[:] = np.array(skills, dtype=np.float32)
    if gears is not None:
        pop.gear[:] = np.array(gears, dtype=np.float32)
    return pop


def test_team_strength_includes_gear_weight():
    pop = _pop_with_skills(
        [1.0, 0.0, -1.0, 2.0], gears=[0.5, 0.5, 0.5, 0.5]
    )
    lobby = Lobby(teams=[[0, 1], [2, 3]])
    cfg = OutcomeConfig(kind="extraction", gear_weight=0.5)
    gen = ExtractionOutcomeGenerator(cfg)
    result = gen.generate(lobby, pop, np.random.default_rng(0))

    # Team 0: mean(1.0, 0.0) + 0.5*mean(0.5,0.5) = 0.75
    # Team 1: mean(-1.0, 2.0) + 0.5*0.5 = 0.75
    assert result.team_strength is not None
    np.testing.assert_allclose(result.team_strength, [0.75, 0.75], atol=1e-5)


def test_mean_extractor_count_matches_target():
    """Sample mean extractor count over many matches hits the config target."""
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)
    totals = []
    for seed in range(5000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        totals.append(int(r.extracted.sum()))
    totals_arr = np.array(totals)
    # Mean within 0.05 of target.
    assert abs(totals_arr.mean() - 1.8) < 0.05


def test_p_zero_extract_tail_rate():
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)
    zeros = 0
    trials = 10_000
    for seed in range(trials):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        if not r.extracted.any():
            zeros += 1
    p0 = zeros / trials
    assert abs(p0 - 0.01) < 0.005


def test_stronger_team_extracts_more_often():
    skills = [
        2.0, 2.0, 2.0,
        -2.0, -2.0, -2.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
        strength_sensitivity=2.0,
    )
    gen = ExtractionOutcomeGenerator(cfg)

    counts = np.zeros(4, dtype=int)
    for seed in range(1000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        counts += r.extracted.astype(int)

    # Strong team extracts most; weak team least.
    assert counts[0] > counts[2]
    assert counts[0] > counts[3]
    assert counts[1] < counts[2]
    assert counts[1] < counts[3]


def test_expected_extract_sums_to_mean_target():
    """expected_extract is a Plackett-Luce marginal; it sums to the k drawn
    for that match. Averaged across many matches it approaches the target
    mean number of extractors."""
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)

    sums = []
    for seed in range(2000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        sums.append(float(r.expected_extract.sum()))
    sums_arr = np.array(sums)
    assert abs(sums_arr.mean() - 1.8) < 0.05


def test_kill_attribution_only_when_extractors_exist():
    """If nobody extracts, kill_credits is empty; if some extract, credits
    only reference actual extractors as killers."""
    skills = [0.0] * 12
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    cfg = OutcomeConfig(
        kind="extraction",
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
    )
    gen = ExtractionOutcomeGenerator(cfg)

    saw_empty = False
    saw_populated = False
    for seed in range(2000):
        r = gen.generate(lobby, pop, np.random.default_rng(seed))
        if not r.extracted.any():
            assert r.kill_credits == []
            saw_empty = True
        else:
            for killer, dead in r.kill_credits:
                assert bool(r.extracted[killer]) is True
                assert bool(r.extracted[dead]) is False
            if r.kill_credits:
                saw_populated = True
    # With P(k=0)=1% and 2000 trials we should have seen both cases.
    assert saw_empty is True
    assert saw_populated is True


def test_higher_beta_concentrates_wins_on_stronger_team():
    """Raising strength_sensitivity (beta) should push the strongest team's
    extract rate higher."""
    skills = [
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        -0.5, -0.5, -0.5,
        -1.0, -1.0, -1.0,
    ]
    pop = _pop_with_skills(skills)
    lobby = Lobby(teams=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])

    def run_and_count(beta: float) -> np.ndarray:
        cfg = OutcomeConfig(
            kind="extraction",
            mean_extractors_per_match=1.8,
            p_zero_extract=0.01,
            p_all_extract=0.03,
            strength_sensitivity=beta,
        )
        gen = ExtractionOutcomeGenerator(cfg)
        counts = np.zeros(4, dtype=int)
        for seed in range(500):
            r = gen.generate(lobby, pop, np.random.default_rng(seed))
            counts += r.extracted.astype(int)
        return counts

    counts_low = run_and_count(0.5)
    counts_high = run_and_count(5.0)
    # Strongest team (idx 0) should extract more often at higher beta.
    assert counts_high[0] > counts_low[0]
    # Weakest team (idx 3) should extract less often at higher beta.
    assert counts_high[3] < counts_low[3]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_extraction_outcomes.py -v`
Expected: FAIL — many tests will error because `ExtractionOutcomeGenerator` still references `cfg.baseline_extract_prob`.

- [ ] **Step 3: Rewrite `extraction.py`**

Replace the contents of `src/mm_sim/outcomes/extraction.py`:

```python
"""Extraction outcome generator: softmax-over-strength winner sampling."""

from __future__ import annotations

import numpy as np

from mm_sim.config import OutcomeConfig
from mm_sim.matchmaker.base import Lobby
from mm_sim.outcomes.base import MatchResult
from mm_sim.outcomes.softmax_winners import (
    plackett_luce_marginals,
    sample_extractor_count,
)
from mm_sim.population import Population


class ExtractionOutcomeGenerator:
    """Each match draws k (number of extractors) from a mixture, then samples
    k winners without replacement from ``softmax(beta * strengths)``.

    ``expected_extract[i]`` is the Plackett-Luce marginal probability that
    team i is among the top-k under that same softmax — the quantity the
    calibration chart compares against the rating-updater's Elo view.
    """

    def __init__(self, cfg: OutcomeConfig) -> None:
        self.cfg = cfg

    def generate(
        self, lobby: Lobby, pop: Population, rng: np.random.Generator
    ) -> MatchResult:
        n_teams = len(lobby.teams)
        strengths = np.zeros(n_teams, dtype=np.float32)
        # Per-player effective performance (without per-team noise). Used for
        # within-team contribution shares.
        player_perfs: list[np.ndarray] = []
        for i, team in enumerate(lobby.teams):
            arr = np.array(team, dtype=np.int32)
            s = pop.true_skill[arr].astype(np.float32)
            if self.cfg.gear_weight > 0:
                s = s + self.cfg.gear_weight * pop.gear[arr].astype(np.float32)
            strengths[i] = s.mean()
            perf_noise = rng.normal(
                0.0, self.cfg.noise_std, size=len(arr)
            ).astype(np.float32)
            player_perfs.append(s + perf_noise)

        # Draw k winners for this match.
        k = sample_extractor_count(
            n_teams=n_teams,
            mean_extractors=self.cfg.mean_extractors_per_match,
            p_zero=self.cfg.p_zero_extract,
            p_all=self.cfg.p_all_extract,
            rng=rng,
        )
        beta = self.cfg.strength_sensitivity
        extracted = np.zeros(n_teams, dtype=bool)
        if k >= n_teams:
            extracted[:] = True
        elif k > 0:
            # Sample winners via Plackett-Luce: softmax pick, remove, renorm.
            shifted = beta * strengths.astype(np.float64)
            shifted = shifted - shifted.max()
            w = np.exp(shifted)
            remaining = np.ones(n_teams, dtype=bool)
            for _ in range(k):
                pool = np.flatnonzero(remaining)
                probs = w[pool] / w[pool].sum()
                pick = int(rng.choice(pool, p=probs))
                extracted[pick] = True
                remaining[pick] = False

        # Per-team marginal probability (truth view) stored for calibration.
        expected_extract = plackett_luce_marginals(
            strengths=strengths, k=k, beta=beta
        ).astype(np.float32)

        # Kill attribution: weakest extractor above the dead team in strength,
        # or the strongest extractor if no extractor is above them.
        kill_credits: list[tuple[int, int]] = []
        extractor_idxs = np.flatnonzero(extracted)
        if extractor_idxs.size > 0:
            for dead in np.flatnonzero(~extracted):
                dead_strength = strengths[dead]
                above = [
                    i for i in extractor_idxs if strengths[i] > dead_strength
                ]
                if above:
                    killer = int(min(above, key=lambda i: strengths[i]))
                else:
                    killer = int(max(extractor_idxs, key=lambda i: strengths[i]))
                kill_credits.append((killer, int(dead)))

        # Per-player contributions (ordering matches result.flat_player_ids()).
        flat_perf = np.concatenate(player_perfs).astype(np.float32)
        shares = np.zeros_like(flat_perf)
        cursor = 0
        for i, team in enumerate(lobby.teams):
            n = len(team)
            team_perf = flat_perf[cursor : cursor + n]
            shifted = team_perf - team_perf.min() + 0.1
            shares[cursor : cursor + n] = shifted / shifted.mean()
            cursor += n

        return MatchResult(
            lobby=lobby,
            extracted=extracted,
            kill_credits=kill_credits,
            expected_extract=expected_extract,
            team_strength=strengths,
            winning_team=-1,
            contributions={
                "player_perf": flat_perf,
                "share": shares,
            },
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_extraction_outcomes.py -v`
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/outcomes/extraction.py tests/test_extraction_outcomes.py
git commit -m "change(outcomes): switch extraction to softmax winner sampling

ExtractionOutcomeGenerator now draws k from the mixture helper,
samples k winners via Plackett-Luce on softmax(beta*strength), and
reports per-team marginals as expected_extract. Kill attribution
and contribution-share logic unchanged."
```

---

## Task 4: Update scenarios and debug entry points

**Files:**
- Modify: `scenarios/defaults.toml:36-37`
- Modify: `src/debug/debug_rating.py:52-57`

- [ ] **Step 1: Update `scenarios/defaults.toml`**

Replace the `[config.outcomes]` block:

```toml
[config.outcomes]
kind = "extraction"
gear_weight = 0.5
mean_extractors_per_match = 1.8
p_zero_extract = 0.01
p_all_extract = 0.03
strength_sensitivity = 2.0
```

- [ ] **Step 2: Update `src/debug/debug_rating.py`**

Replace the `OutcomeConfig(...)` call block (currently lines 52-57):

```python
    oc_cfg = OutcomeConfig(
        kind="extraction",
        gear_weight=0.0,  # ignore gear for now — focus on skill signal
        mean_extractors_per_match=1.8,
        p_zero_extract=0.01,
        p_all_extract=0.03,
        strength_sensitivity=2.0,
    )
```

- [ ] **Step 3: Verify scenarios still load**

Run: `uv run python -c "from mm_sim.scenarios import load_all_scenarios; print(len(load_all_scenarios('scenarios')))"`
Expected: prints an integer (number of scenarios loaded), no validation error.

If the loader symbol is different, fall back to running a scenario loader directly. Whatever path the dashboard uses to read `scenarios/defaults.toml` must not raise a pydantic validation error. If no single-command loader is exposed, run one scenario end to end as the smoke test in Step 5.

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: all tests pass, no reference to `baseline_extract_prob` anywhere.

- [ ] **Step 5: Commit**

```bash
git add scenarios/defaults.toml src/debug/debug_rating.py
git commit -m "change(scenarios): swap baseline_extract_prob for mixture knobs

scenarios/defaults.toml and debug_rating now use mean_extractors_per_match,
p_zero_extract, p_all_extract, and strength_sensitivity=2.0."
```

---

## Task 5: End-to-end smoke test

**Files:** none to modify; run a scenario and inspect output.

- [ ] **Step 1: Run one scenario**

Pick the smallest scenario that exercises extraction. Run:

```bash
uv run python -m mm_sim run scenarios/skill_only.toml
```

Or whichever entry point the project uses (check `pyproject.toml` `[project.scripts]` or `src/mm_sim/__main__.py`). Expected: scenario completes without raising; outputs go to `experiments/<season>/skill_only/...`.

- [ ] **Step 2: Inspect match_teams.parquet for sanity**

```bash
uv run python - <<'EOF'
import polars as pl
from pathlib import Path

roots = sorted(Path("experiments").glob("*/skill_only/*"))
assert roots, "no skill_only runs found"
latest = roots[-1]
mt = pl.read_parquet(latest / "match_teams.parquet")
# Group by match and count extractors per match
per_match = (
    mt.group_by(["day", "match_idx"])
      .agg(pl.col("extracted").cast(pl.Int32).sum().alias("k"))
)
counts = per_match.group_by("k").agg(pl.len().alias("n")).sort("k")
total = int(counts["n"].sum())
print(f"Total matches: {total}")
for row in counts.iter_rows(named=True):
    print(f"  k={row['k']}: {row['n']} ({row['n']/total:.3f})")
print(f"Mean k: {per_match['k'].mean():.3f}")
EOF
```

Expected: `Mean k` ≈ 1.8; `k=0` share ≈ 0.01; `k=4` share ≈ 0.03.

- [ ] **Step 3: Inspect the dashboard**

Run: `uv run streamlit run src/mm_sim/dashboard/0_Home.py`

Open the Compare Scenarios page, select the newly-run scenario, and confirm:
- The "extracts per match" stacked-area chart shows a thin k=0 band at ~1%, a thin k=4 band at ~3%, with k=1 and k=2 dominating.
- "MM rating calibration" still renders — gap and floor lines present.

No commit here; this is verification only.

---

## Notes on scope excluded from this plan

- **Re-tuning β.** Default is 2.0 based on the comparison we did against the old normal-CDF model. Actual tuning should happen after running a couple of scenarios and inspecting the dashboard.
- **Re-running existing scenarios.** All prior parquet runs are under the old model; they are not compatible with the new model's `expected_extract` values. The user will re-run scenarios manually after this plan lands.
- **`baseline_extract_prob` removal from old plan/spec docs.** `docs/superpowers/plans/2026-04-15-multi-team-extraction.md` and the corresponding spec are history; they are not live config and do not need editing.

---

## Self-Review

**Spec coverage:**
- New softmax sampling model ✓ (Tasks 1, 3)
- Mixture-based k ✓ (Task 1, covered by `sample_extractor_count`)
- Config changes (new knobs + remove baseline_extract_prob) ✓ (Task 2)
- Default β=2.0 ✓ (Task 2)
- Scenarios and debug entry points updated ✓ (Task 4)
- Tests rewritten against new model ✓ (Tasks 1, 2, 3)
- Smoke test exercising the full pipeline ✓ (Task 5)
- MM calibration chart: **no change** — the design choice is to leave the Elo-vs-outcome mismatch visible via the `floor` line. Confirmed with the user.

**Placeholder scan:** All code blocks contain concrete code; all commands concrete; no "TBD" / "handle edge cases" phrasing.

**Type consistency:**
- `sample_extractor_count(n_teams, mean_extractors, p_zero, p_all, rng)` — same signature Tasks 1 and 3.
- `plackett_luce_marginals(strengths, k, beta)` — same signature Tasks 1 and 3.
- Config fields `mean_extractors_per_match`, `p_zero_extract`, `p_all_extract`, `strength_sensitivity` — same names in Tasks 2, 3, 4.
