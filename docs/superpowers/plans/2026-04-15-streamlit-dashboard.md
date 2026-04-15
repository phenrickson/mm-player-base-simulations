# Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit dashboard (`uv run streamlit run src/mm_sim/dashboard/app.py`) with two pages — Single Run and Compare Scenarios — for interactive exploration of experiment parquet artifacts.

**Architecture:** Thin loader module wrapping the existing `load_experiment` / `list_experiments` helpers from `mm_sim.experiments`, cached with `@st.cache_data`. Shared Plotly chart builders taking labeled polars DataFrames. Streamlit multipage app with pages auto-discovered from `src/mm_sim/dashboard/pages/`.

**Tech Stack:** Python 3.12, Streamlit, Plotly, polars (existing), uv (existing).

**Spec:** `docs/superpowers/specs/2026-04-15-streamlit-dashboard-design.md`

---

## Background the engineer needs

- The project is `mm-player-base-simulations`, a simulation of matchmaking
  over a player base. Experiments are runs of the simulator.
- Data layout: `experiments/<season>/<scenario>/<version>/` with files
  `metadata.json`, `config.json`, `aggregate.parquet`,
  `population.parquet`, `matches.parquet`, `plots/`.
- `aggregate.parquet` has one row per day. Relevant columns:
  - `day` (int) — 0-indexed day
  - `active_count` (int) — active players that day
  - `matches_played` (int)
  - `blowouts` (int)
  - `true_skill_mean`, `true_skill_p10`, `true_skill_p50`, `true_skill_p90`
  - `observed_skill_mean`
  - `rating_error_mean` — |observed − true|
  - `lobby_range_mean`, `lobby_range_p50`, `lobby_range_p90`
  - `lobby_std_mean`
  - `team_gap_mean`, `team_gap_p50`, `team_gap_p90`
  - `win_prob_dev_mean`, `win_prob_dev_p50`, `win_prob_dev_p90` — deviation
    from 50/50 win prob; **low is good match quality**
- `population.parquet` has `day, player_id, true_skill, observed_skill,
  experience, gear, active, ...`. Big — lazy-load.
- Existing helpers in `src/mm_sim/experiments.py`:
  - `list_experiments(experiments_dir)` → polars DF, one row per
    (season, name, version)
  - `load_experiment(name, season, version, experiments_dir)` → `Experiment`
    dataclass with `metadata`, `config`, `aggregate`, `population`, `matches`
  - `latest_version_dir(base, name)` — returns Path

**Use polars, not pandas.** Convert to pandas only at the Plotly
boundary if needed (Plotly accepts polars DataFrames directly via
`.to_pandas()` when necessary; `px` functions often work with
polars too, but safer to call `.to_pandas()` explicitly).

**Metric naming convention** (used throughout pages + charts):
- "active population" → `active_count`
- "retention" → `active_count / aggregate.row(0)["active_count"]`
- "match quality" → `win_prob_dev_mean` (labeled "win-prob deviation;
  lower is better")
- "rating error" → `rating_error_mean`
- "blowout share" → `blowouts / matches_played` (guard div-by-zero)

All commands below must be prefixed with `uv run` per project convention.

---

## File Structure

**Create:**
- `src/mm_sim/dashboard/__init__.py`
- `src/mm_sim/dashboard/app.py` — entry / landing page
- `src/mm_sim/dashboard/loader.py` — cached experiment discovery + loading
- `src/mm_sim/dashboard/charts.py` — Plotly chart builders
- `src/mm_sim/dashboard/pages/1_Single_Run.py`
- `src/mm_sim/dashboard/pages/2_Compare_Scenarios.py`
- `tests/dashboard/__init__.py`
- `tests/dashboard/test_loader.py`
- `tests/dashboard/test_charts.py`

**Modify:**
- `pyproject.toml` — add `streamlit`, `plotly` dependencies
- `justfile` — add `dashboard` recipe

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add streamlit and plotly to dependencies**

Open `pyproject.toml`, find the `dependencies` list, and add:

```toml
    "streamlit>=1.37",
    "plotly>=5.22",
```

- [ ] **Step 2: Sync the venv**

Run: `uv sync`
Expected: `uv` installs streamlit, plotly, and their deps; no errors.

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "import streamlit; import plotly; print(streamlit.__version__, plotly.__version__)"`
Expected: two version numbers printed, no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add streamlit and plotly for dashboard"
```

---

## Task 2: Create dashboard package skeleton

**Files:**
- Create: `src/mm_sim/dashboard/__init__.py`
- Create: `tests/dashboard/__init__.py`

- [ ] **Step 1: Create dashboard package**

Create `src/mm_sim/dashboard/__init__.py` with contents:

```python
"""Streamlit dashboard for interactive experiment exploration."""
```

- [ ] **Step 2: Create dashboard tests package**

Create `tests/dashboard/__init__.py` as an empty file.

- [ ] **Step 3: Commit**

```bash
git add src/mm_sim/dashboard/__init__.py tests/dashboard/__init__.py
git commit -m "scaffold: create dashboard package"
```

---

## Task 3: Loader — list_seasons

**Files:**
- Create: `src/mm_sim/dashboard/loader.py`
- Create: `tests/dashboard/test_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dashboard/test_loader.py`:

```python
"""Tests for the dashboard loader module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mm_sim.dashboard import loader


@pytest.fixture
def experiments_tree(tmp_path: Path) -> Path:
    """Build a minimal experiments/ tree with two seasons and two scenarios."""
    root = tmp_path / "experiments"
    for season in ("season-a", "season-b"):
        for scen in ("skill_only", "random_mm"):
            for ver in ("v1", "v2"):
                d = root / season / scen / ver
                d.mkdir(parents=True)
                (d / "metadata.json").write_text("{}")
    return root


def test_list_seasons_returns_sorted_season_names(experiments_tree: Path):
    assert loader.list_seasons(experiments_tree) == ["season-a", "season-b"]


def test_list_seasons_returns_empty_when_missing(tmp_path: Path):
    assert loader.list_seasons(tmp_path / "missing") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mm_sim.dashboard.loader'` or `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/mm_sim/dashboard/loader.py`:

```python
"""Experiment discovery and cached loading for the Streamlit dashboard.

All public functions accept an explicit `experiments_dir` so they are
testable without Streamlit. The page modules call the module-level
`cached_*` wrappers that add `@st.cache_data`.
"""
from __future__ import annotations

from pathlib import Path


def list_seasons(experiments_dir: Path) -> list[str]:
    """Return season directory names, sorted alphabetically."""
    if not experiments_dir.exists():
        return []
    return sorted(
        p.name for p in experiments_dir.iterdir() if p.is_dir()
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/dashboard/test_loader.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/dashboard/loader.py tests/dashboard/test_loader.py
git commit -m "feat(dashboard): loader.list_seasons"
```

---

## Task 4: Loader — list_scenarios and list_versions

**Files:**
- Modify: `src/mm_sim/dashboard/loader.py`
- Modify: `tests/dashboard/test_loader.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/dashboard/test_loader.py`:

```python
def test_list_scenarios_returns_sorted_scenario_names(experiments_tree: Path):
    assert loader.list_scenarios(experiments_tree, "season-a") == [
        "random_mm",
        "skill_only",
    ]


def test_list_scenarios_returns_empty_for_unknown_season(experiments_tree: Path):
    assert loader.list_scenarios(experiments_tree, "nope") == []


def test_list_versions_returns_versions_oldest_first(experiments_tree: Path):
    assert loader.list_versions(experiments_tree, "season-a", "skill_only") == [
        "v1",
        "v2",
    ]


def test_latest_version_returns_highest(experiments_tree: Path):
    assert loader.latest_version(experiments_tree, "season-a", "skill_only") == "v2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_loader.py -v`
Expected: 4 new tests FAIL with AttributeError.

- [ ] **Step 3: Implement**

Append to `src/mm_sim/dashboard/loader.py`:

```python
def list_scenarios(experiments_dir: Path, season: str) -> list[str]:
    """Return scenario directory names under a season, sorted."""
    season_dir = experiments_dir / season
    if not season_dir.exists():
        return []
    return sorted(
        p.name
        for p in season_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def list_versions(
    experiments_dir: Path, season: str, scenario: str
) -> list[str]:
    """Return version directory names (e.g. ['v1', 'v2']) sorted by
    integer suffix ascending. Latest is the last element."""
    scen_dir = experiments_dir / season / scenario
    if not scen_dir.exists():
        return []
    versions = [
        p.name
        for p in scen_dir.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    ]
    return sorted(versions, key=lambda v: int(v[1:]))


def latest_version(
    experiments_dir: Path, season: str, scenario: str
) -> str:
    """Return the highest-numbered version under (season, scenario).

    Raises FileNotFoundError if none exist.
    """
    versions = list_versions(experiments_dir, season, scenario)
    if not versions:
        raise FileNotFoundError(
            f"no versions under {experiments_dir / season / scenario}"
        )
    return versions[-1]
```

Note: `list_scenarios` filters out `_comparisons` (names starting with `_`)
because the comparison output directory sits alongside scenarios.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/dashboard/test_loader.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/dashboard/loader.py tests/dashboard/test_loader.py
git commit -m "feat(dashboard): list_scenarios, list_versions, latest_version"
```

---

## Task 5: Loader — load_run wrapping load_experiment

**Files:**
- Modify: `src/mm_sim/dashboard/loader.py`
- Modify: `tests/dashboard/test_loader.py`

This task wraps the existing `mm_sim.experiments.load_experiment`.
It returns the existing `Experiment` dataclass unchanged — no new
wrapper type needed.

- [ ] **Step 1: Write failing test**

Append to `tests/dashboard/test_loader.py`:

```python
def test_load_run_uses_load_experiment(monkeypatch, experiments_tree: Path):
    """load_run delegates to mm_sim.experiments.load_experiment."""
    called = {}

    def fake_load(name, season, version, experiments_dir):
        called["args"] = (name, season, version, Path(experiments_dir))
        return "sentinel"

    monkeypatch.setattr(
        "mm_sim.dashboard.loader.load_experiment", fake_load
    )
    result = loader.load_run(
        experiments_tree, "season-a", "skill_only", "v2"
    )
    assert result == "sentinel"
    assert called["args"] == (
        "skill_only",
        "season-a",
        "v2",
        experiments_tree,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_loader.py::test_load_run_uses_load_experiment -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'load_run'`.

- [ ] **Step 3: Implement**

Add near the top of `src/mm_sim/dashboard/loader.py`:

```python
from mm_sim.experiments import Experiment, load_experiment
```

Append:

```python
def load_run(
    experiments_dir: Path,
    season: str,
    scenario: str,
    version: str,
) -> Experiment:
    """Load a single experiment run.

    Thin wrapper over `mm_sim.experiments.load_experiment`, with the
    argument order reshuffled to match the dashboard's (season, scenario,
    version) triple.
    """
    return load_experiment(
        scenario,
        season=season,
        version=version,
        experiments_dir=experiments_dir,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/dashboard/test_loader.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/dashboard/loader.py tests/dashboard/test_loader.py
git commit -m "feat(dashboard): load_run wrapper"
```

---

## Task 6: Loader — Streamlit-cached wrappers

**Files:**
- Modify: `src/mm_sim/dashboard/loader.py`

Pages will import the `cached_*` functions. Unit tests stay on the
uncached functions (Streamlit's cache requires a script context).

- [ ] **Step 1: Add cached wrappers**

Append to `src/mm_sim/dashboard/loader.py`:

```python
import streamlit as st


@st.cache_data(show_spinner=False)
def cached_list_seasons(experiments_dir_str: str) -> list[str]:
    return list_seasons(Path(experiments_dir_str))


@st.cache_data(show_spinner=False)
def cached_list_scenarios(experiments_dir_str: str, season: str) -> list[str]:
    return list_scenarios(Path(experiments_dir_str), season)


@st.cache_data(show_spinner=False)
def cached_list_versions(
    experiments_dir_str: str, season: str, scenario: str
) -> list[str]:
    return list_versions(Path(experiments_dir_str), season, scenario)


@st.cache_data(show_spinner="Loading experiment…")
def cached_load_run(
    experiments_dir_str: str,
    season: str,
    scenario: str,
    version: str,
) -> Experiment:
    return load_run(Path(experiments_dir_str), season, scenario, version)
```

Cache keys are `str` so the `Path` object doesn't mess with hashing.

- [ ] **Step 2: Verify import still works**

Run: `uv run python -c "from mm_sim.dashboard import loader; print(loader.cached_list_seasons)"`
Expected: prints a `<streamlit.runtime.caching...>` object, no error.

- [ ] **Step 3: Run all loader tests**

Run: `uv run pytest tests/dashboard/test_loader.py -v`
Expected: all tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add src/mm_sim/dashboard/loader.py
git commit -m "feat(dashboard): streamlit cache_data wrappers"
```

---

## Task 7: Charts — one builder + test

Establish the chart function shape with the simplest one. Remaining
builders follow the same pattern in Task 8.

**Files:**
- Create: `src/mm_sim/dashboard/charts.py`
- Create: `tests/dashboard/test_charts.py`

- [ ] **Step 1: Write failing test**

Create `tests/dashboard/test_charts.py`:

```python
"""Tests for dashboard chart builders."""
from __future__ import annotations

import polars as pl
import plotly.graph_objects as go

from mm_sim.dashboard import charts


def _agg(days: int = 10, start_pop: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "day": list(range(days)),
            "active_count": [start_pop - i for i in range(days)],
            "matches_played": [50 for _ in range(days)],
            "blowouts": [5 for _ in range(days)],
            "rating_error_mean": [0.1 * i for i in range(days)],
            "win_prob_dev_mean": [0.2 for _ in range(days)],
            "true_skill_p10": [0.1 for _ in range(days)],
            "true_skill_p50": [0.5 for _ in range(days)],
            "true_skill_p90": [0.9 for _ in range(days)],
        }
    )


def test_population_over_time_single_run_returns_one_trace():
    fig = charts.population_over_time([("run-a", _agg())])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "run-a"


def test_population_over_time_multi_run_returns_one_trace_per_run():
    fig = charts.population_over_time(
        [("a", _agg()), ("b", _agg()), ("c", _agg())]
    )
    assert len(fig.data) == 3
    assert [t.name for t in fig.data] == ["a", "b", "c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dashboard/test_charts.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/mm_sim/dashboard/charts.py`:

```python
"""Plotly chart builders shared by Single Run and Compare pages.

Each function accepts `runs: list[tuple[str, pl.DataFrame]]` — a list
of (label, aggregate) pairs — and returns a `plotly.graph_objects.Figure`
with one trace per run. Single-run pages pass a 1-element list; compare
pages pass N.
"""
from __future__ import annotations

import plotly.graph_objects as go
import polars as pl


RunList = list[tuple[str, pl.DataFrame]]


def _line_chart(
    runs: RunList,
    y_col: str,
    title: str,
    y_label: str,
) -> go.Figure:
    fig = go.Figure()
    for label, df in runs:
        fig.add_trace(
            go.Scatter(
                x=df["day"].to_list(),
                y=df[y_col].to_list(),
                mode="lines",
                name=label,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="day",
        yaxis_title=y_label,
        hovermode="x unified",
    )
    return fig


def population_over_time(runs: RunList) -> go.Figure:
    return _line_chart(
        runs, "active_count", "Active population over time", "active players"
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/dashboard/test_charts.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/dashboard/charts.py tests/dashboard/test_charts.py
git commit -m "feat(dashboard): population_over_time chart"
```

---

## Task 8: Charts — remaining builders

**Files:**
- Modify: `src/mm_sim/dashboard/charts.py`
- Modify: `tests/dashboard/test_charts.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/dashboard/test_charts.py`:

```python
def test_retention_over_time_normalizes_to_day_zero():
    df = _agg(days=5, start_pop=100)  # 100, 99, 98, 97, 96
    fig = charts.retention_over_time([("r", df)])
    ys = list(fig.data[0].y)
    assert ys[0] == 1.0
    assert abs(ys[-1] - 0.96) < 1e-9


def test_match_quality_over_time_uses_win_prob_dev():
    fig = charts.match_quality_over_time([("r", _agg())])
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [0.2] * 10


def test_rating_error_over_time_has_trace():
    fig = charts.rating_error_over_time([("r", _agg())])
    assert len(fig.data) == 1


def test_blowout_share_over_time_handles_zero_matches():
    df = _agg(days=3).with_columns(pl.Series("matches_played", [0, 50, 50]))
    fig = charts.blowout_share_over_time([("r", df)])
    assert list(fig.data[0].y)[0] == 0.0  # div-by-zero guarded


def test_skill_distribution_returns_histogram():
    pop = pl.DataFrame(
        {"day": [9] * 5, "true_skill": [0.1, 0.3, 0.5, 0.7, 0.9]}
    )
    fig = charts.skill_distribution(pop, day=9)
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_charts.py -v`
Expected: 5 new tests FAIL with AttributeError.

- [ ] **Step 3: Implement remaining builders**

Append to `src/mm_sim/dashboard/charts.py`:

```python
def retention_over_time(runs: RunList) -> go.Figure:
    """Active count normalized by each run's day-0 value."""
    normalized: RunList = []
    for label, df in runs:
        day0 = df["active_count"].item(0)
        factor = 1.0 / day0 if day0 else 0.0
        normalized.append(
            (label, df.with_columns((pl.col("active_count") * factor).alias("retention")))
        )
    return _line_chart(
        normalized, "retention", "Retention over time", "fraction of day-0 population"
    )


def match_quality_over_time(runs: RunList) -> go.Figure:
    return _line_chart(
        runs,
        "win_prob_dev_mean",
        "Match quality (win-prob deviation, lower is better)",
        "mean |win prob − 0.5|",
    )


def rating_error_over_time(runs: RunList) -> go.Figure:
    return _line_chart(
        runs,
        "rating_error_mean",
        "Rating error over time",
        "mean |observed − true|",
    )


def blowout_share_over_time(runs: RunList) -> go.Figure:
    """blowouts / matches_played per day, with div-by-zero guarded."""
    shared: RunList = []
    for label, df in runs:
        shared.append(
            (
                label,
                df.with_columns(
                    pl.when(pl.col("matches_played") > 0)
                    .then(pl.col("blowouts") / pl.col("matches_played"))
                    .otherwise(0.0)
                    .alias("blowout_share")
                ),
            )
        )
    return _line_chart(
        shared, "blowout_share", "Blowout share over time", "blowouts / matches"
    )


def skill_distribution(population: pl.DataFrame, day: int) -> go.Figure:
    """Histogram of true_skill for a single run's population on one day."""
    snap = population.filter(pl.col("day") == day)
    fig = go.Figure(
        go.Histogram(x=snap["true_skill"].to_list(), nbinsx=40, name=f"day {day}")
    )
    fig.update_layout(
        title=f"True-skill distribution on day {day}",
        xaxis_title="true skill",
        yaxis_title="player count",
    )
    return fig
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/dashboard/test_charts.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mm_sim/dashboard/charts.py tests/dashboard/test_charts.py
git commit -m "feat(dashboard): retention, match quality, rating error, blowout share, skill distribution charts"
```

---

## Task 9: Landing page (app.py)

**Files:**
- Create: `src/mm_sim/dashboard/app.py`

- [ ] **Step 1: Write app.py**

Create `src/mm_sim/dashboard/app.py`:

```python
"""Streamlit dashboard entry point.

Run: `uv run streamlit run src/mm_sim/dashboard/app.py`

Streamlit auto-discovers pages under `pages/` and shows them in the
sidebar. This file is the landing page.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from mm_sim.dashboard import loader

DEFAULT_EXPERIMENTS_DIR = Path("experiments")

st.set_page_config(
    page_title="mm-sim dashboard",
    layout="wide",
)

st.title("mm-sim experiment dashboard")

st.markdown(
    """
    Explore experiment artifacts interactively. Use the sidebar to pick
    a page:

    - **Single Run** — drill into one scenario/version
    - **Compare Scenarios** — overlay metrics across scenarios in a season
    """
)

exp_dir_str = st.sidebar.text_input(
    "experiments directory",
    value=str(DEFAULT_EXPERIMENTS_DIR),
    help="Path to the top-level experiments/ directory.",
)
st.session_state["experiments_dir"] = exp_dir_str

seasons = loader.cached_list_seasons(exp_dir_str)
if not seasons:
    st.warning(
        f"No experiments found under `{exp_dir_str}`. "
        "Run `just scenarios` to generate some."
    )
else:
    st.subheader("Available seasons")
    for season in seasons:
        scenarios = loader.cached_list_scenarios(exp_dir_str, season)
        st.markdown(f"**{season}** — {len(scenarios)} scenarios")
```

The `experiments_dir` path is stored in `st.session_state` so page
modules share it.

- [ ] **Step 2: Smoke-test it launches**

Run: `uv run streamlit run src/mm_sim/dashboard/app.py --server.headless true --server.port 8599 &`
Then: `sleep 3 && curl -s http://localhost:8599/ | head -c 200 && kill %1`
Expected: HTML starts with `<!DOCTYPE html>`, no crash.

If the background job control doesn't work cleanly, just run without `&`,
Ctrl-C after 3 seconds, and verify no Python traceback printed.

- [ ] **Step 3: Commit**

```bash
git add src/mm_sim/dashboard/app.py
git commit -m "feat(dashboard): landing page"
```

---

## Task 10: Single Run page

**Files:**
- Create: `src/mm_sim/dashboard/pages/1_Single_Run.py`

- [ ] **Step 1: Write the page**

Create `src/mm_sim/dashboard/pages/1_Single_Run.py`:

```python
"""Single Run page — deep-dive into one experiment."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from mm_sim.dashboard import charts, loader

st.set_page_config(page_title="Single Run", layout="wide")
st.title("Single Run")

exp_dir_str = st.session_state.get("experiments_dir", "experiments")

seasons = loader.cached_list_seasons(exp_dir_str)
if not seasons:
    st.warning(f"No experiments found under `{exp_dir_str}`.")
    st.stop()

season = st.sidebar.selectbox("season", seasons, index=len(seasons) - 1)
scenarios = loader.cached_list_scenarios(exp_dir_str, season)
if not scenarios:
    st.warning(f"No scenarios in `{season}`.")
    st.stop()

scenario = st.sidebar.selectbox("scenario", scenarios)
versions = loader.cached_list_versions(exp_dir_str, season, scenario)
if not versions:
    st.warning(f"No versions under `{season}/{scenario}`.")
    st.stop()

version = st.sidebar.selectbox(
    "version", versions, index=len(versions) - 1
)

exp = loader.cached_load_run(exp_dir_str, season, scenario, version)
agg = exp.aggregate
label = f"{scenario}/{version}"

# ---- KPI row ----------------------------------------------------------
last_row = agg.sort("day").tail(1).row(0, named=True)
day0 = agg.row(0, named=True)["active_count"]
retention = last_row["active_count"] / day0 if day0 else 0.0
col1, col2, col3, col4 = st.columns(4)
col1.metric("final active", f"{last_row['active_count']:,}")
col2.metric("overall retention", f"{retention:.1%}")
col3.metric("mean match quality (↓better)", f"{last_row['win_prob_dev_mean']:.3f}")
col4.metric("mean rating error", f"{last_row['rating_error_mean']:.3f}")

# ---- Charts -----------------------------------------------------------
runs = [(label, agg)]
st.plotly_chart(charts.population_over_time(runs), use_container_width=True)
st.plotly_chart(charts.retention_over_time(runs), use_container_width=True)
st.plotly_chart(charts.match_quality_over_time(runs), use_container_width=True)
st.plotly_chart(charts.rating_error_over_time(runs), use_container_width=True)
st.plotly_chart(charts.blowout_share_over_time(runs), use_container_width=True)

if exp.population is not None:
    max_day = int(exp.population["day"].max())
    day = st.slider("skill-distribution day", 0, max_day, max_day)
    st.plotly_chart(
        charts.skill_distribution(exp.population, day=day),
        use_container_width=True,
    )

# ---- Config expander --------------------------------------------------
with st.expander("config.json", expanded=False):
    st.json(exp.config.model_dump(mode="json"))

# ---- Population preview -----------------------------------------------
if exp.population is not None:
    with st.expander("population preview (first 1000 rows of selected day)"):
        preview_day = st.number_input(
            "day", min_value=0, max_value=max_day, value=max_day, step=1
        )
        st.dataframe(
            exp.population.filter(__import__("polars").col("day") == preview_day)
            .head(1000)
            .to_pandas()
        )
```

Note: the `__import__("polars").col` inline is ugly — Task 12 cleans up.

- [ ] **Step 2: Smoke-test**

Run: `uv run streamlit run src/mm_sim/dashboard/app.py --server.headless true --server.port 8599 &`
Then: `sleep 4 && curl -s http://localhost:8599/Single_Run | head -c 200 && kill %1`
Expected: HTML returned, no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/mm_sim/dashboard/pages/1_Single_Run.py
git commit -m "feat(dashboard): single run page"
```

---

## Task 11: Compare Scenarios page

**Files:**
- Create: `src/mm_sim/dashboard/pages/2_Compare_Scenarios.py`

- [ ] **Step 1: Write the page**

Create `src/mm_sim/dashboard/pages/2_Compare_Scenarios.py`:

```python
"""Compare Scenarios page — overlay metrics across scenarios."""
from __future__ import annotations

import streamlit as st

from mm_sim.dashboard import charts, loader

st.set_page_config(page_title="Compare Scenarios", layout="wide")
st.title("Compare Scenarios")

exp_dir_str = st.session_state.get("experiments_dir", "experiments")

seasons = loader.cached_list_seasons(exp_dir_str)
if not seasons:
    st.warning(f"No experiments found under `{exp_dir_str}`.")
    st.stop()

season = st.sidebar.selectbox("season", seasons, index=len(seasons) - 1)
all_scenarios = loader.cached_list_scenarios(exp_dir_str, season)
if not all_scenarios:
    st.warning(f"No scenarios in `{season}`.")
    st.stop()

selected = st.sidebar.multiselect(
    "scenarios", all_scenarios, default=all_scenarios
)
if not selected:
    st.info("Pick at least one scenario.")
    st.stop()

metric_choice = st.sidebar.selectbox(
    "focus metric",
    [
        "active population",
        "retention",
        "match quality",
        "rating error",
        "blowout share",
    ],
)

st.caption("Version policy: the latest version of each scenario is used.")

# Load each scenario's latest version.
runs = []
meta_rows = []
for scen in selected:
    versions = loader.cached_list_versions(exp_dir_str, season, scen)
    if not versions:
        continue
    ver = versions[-1]
    exp = loader.cached_load_run(exp_dir_str, season, scen, ver)
    runs.append((scen, exp.aggregate))
    m = exp.metadata
    meta_rows.append(
        {
            "scenario": scen,
            "version": ver,
            "seed": m.seed,
            "git_sha": (m.git_sha or "")[:8],
            "elapsed_s": m.elapsed_seconds,
        }
    )

if not runs:
    st.warning("Selected scenarios have no versions.")
    st.stop()

metric_fn = {
    "active population": charts.population_over_time,
    "retention": charts.retention_over_time,
    "match quality": charts.match_quality_over_time,
    "rating error": charts.rating_error_over_time,
    "blowout share": charts.blowout_share_over_time,
}[metric_choice]

st.plotly_chart(metric_fn(runs), use_container_width=True)

st.subheader("Small multiples")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(charts.population_over_time(runs), use_container_width=True)
    st.plotly_chart(charts.match_quality_over_time(runs), use_container_width=True)
with c2:
    st.plotly_chart(charts.retention_over_time(runs), use_container_width=True)
    st.plotly_chart(charts.rating_error_over_time(runs), use_container_width=True)

st.subheader("Run metadata")
st.dataframe(meta_rows)
```

- [ ] **Step 2: Smoke-test**

Run: `uv run streamlit run src/mm_sim/dashboard/app.py --server.headless true --server.port 8599 &`
Then: `sleep 4 && curl -s http://localhost:8599/Compare_Scenarios | head -c 200 && kill %1`
Expected: HTML returned, no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/mm_sim/dashboard/pages/2_Compare_Scenarios.py
git commit -m "feat(dashboard): compare scenarios page"
```

---

## Task 12: Tidy Single Run polars import

**Files:**
- Modify: `src/mm_sim/dashboard/pages/1_Single_Run.py`

- [ ] **Step 1: Replace the inline `__import__` with a real import**

At the top of `src/mm_sim/dashboard/pages/1_Single_Run.py`, change:

```python
from mm_sim.dashboard import charts, loader
```

to:

```python
import polars as pl

from mm_sim.dashboard import charts, loader
```

And replace the ugly line:

```python
            exp.population.filter(__import__("polars").col("day") == preview_day)
```

with:

```python
            exp.population.filter(pl.col("day") == preview_day)
```

- [ ] **Step 2: Smoke-test**

Run: `uv run streamlit run src/mm_sim/dashboard/app.py --server.headless true --server.port 8599 &`
Then: `sleep 4 && curl -s http://localhost:8599/Single_Run | head -c 200 && kill %1`
Expected: HTML returned, no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/mm_sim/dashboard/pages/1_Single_Run.py
git commit -m "chore(dashboard): use proper polars import"
```

---

## Task 13: justfile recipe

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Add dashboard recipe**

Append to `justfile`:

```make
# Launch the Streamlit dashboard
dashboard:
    uv run streamlit run src/mm_sim/dashboard/app.py
```

- [ ] **Step 2: Verify just sees it**

Run: `just --list | grep dashboard`
Expected: `dashboard` appears in the list.

- [ ] **Step 3: Commit**

```bash
git add justfile
git commit -m "chore: just dashboard recipe"
```

---

## Task 14: End-to-end manual smoke test

**No file changes.**

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/dashboard/ -v`
Expected: all tests PASS.

- [ ] **Step 2: Launch the dashboard**

Run: `just dashboard`
Expected: Streamlit opens in browser at http://localhost:8501.

- [ ] **Step 3: Manually verify all three pages**

- Landing page shows the seasons present in `experiments/`
- Single Run page: pick `180-days-asymmetric` / `skill_only` / latest version
  - KPI row populated
  - All five line charts render
  - Skill distribution histogram renders and updates on slider
  - Config expander shows JSON
- Compare Scenarios page: pick `180-days-asymmetric`, leave all scenarios
  selected
  - Focus metric chart renders
  - Four small-multiples render
  - Metadata table shows one row per scenario

- [ ] **Step 4: No commit needed — this is verification only.**

---

## Done

Dashboard is usable locally. Future work (deferred per spec):

- Sweep Explorer page
- Version comparison within a single scenario
- Per-player drill-down
