# Streamlit Dashboard for Experiment Exploration — Design

Date: 2026-04-15
Status: Draft (v1)

## Goal

A local Streamlit app for interactively exploring mm-sim experiment
outputs. Currently each run produces a directory of parquet files and
~20 static PNGs; this app replaces the "open the PNG folder" workflow
with interactive Plotly charts and filters. Research tooling — not
production.

## Scope (v1)

Two pages:

1. **Single Run** — pick a season/scenario/version, drill into one run.
2. **Compare Scenarios** — pick a season + multiple scenarios, overlay
   metrics.

Out of scope for v1: sweep explorer, auth, deployment, writing or
mutating experiment artifacts, cross-panel hover sync.

## Directory Layout

```
src/mm_sim/dashboard/
  __init__.py
  app.py              # entry point
  loader.py           # experiment discovery + cached loaders
  charts.py           # plotly chart builders (shared between pages)
  pages/
    1_Single_Run.py
    2_Compare_Scenarios.py
```

Run via `uv run streamlit run src/mm_sim/dashboard/app.py` (wrapped in a
`just dashboard` recipe).

## Components

### `loader.py`

Experiment discovery + cached data access. Wraps the existing
`ExperimentRunner.load()` where sensible; otherwise reads parquet
directly.

Public API:

- `list_seasons() -> list[str]` — directories under `experiments/`
- `list_scenarios(season: str) -> list[str]`
- `list_versions(season: str, scenario: str) -> list[str]` — sorted so
  the latest version is last
- `latest_version(season: str, scenario: str) -> str`
- `load_run(season, scenario, version) -> ExperimentArtifacts`

`ExperimentArtifacts` is a dataclass holding:

- `metadata: dict` (from `metadata.json`)
- `config: dict` (from `config.json`)
- `aggregate: pd.DataFrame` (one row per day)
- `population: pd.DataFrame` (one row per day × player) — loaded lazily
- `matches: pd.DataFrame` (one row per match) — loaded lazily

Metadata/config/aggregate load eagerly. Population and matches are the
big ones; expose them as methods (`artifacts.population()`) that cache
on first call. Each cached method keyed on `(path, mtime)` via
`@st.cache_data`.

### `charts.py`

Pure Plotly builder functions. Each accepts one or more labeled
dataframes and returns a `go.Figure`. Single-run pages pass one
`(label, df)`; compare pages pass several.

Functions for v1 (one per metric family):

- `population_over_time(runs)` — active population line
- `retention_over_time(runs)` — retention line
- `match_quality_over_time(runs)` — match quality line
- `favorite_win_prob_over_time(runs)` — favorite win probability line
- `rating_error_over_time(runs)` — mean rating error line
- `skill_distribution(run)` — histogram / KDE of final-day true skill
  (single-run only)

`runs` is `list[tuple[str, pd.DataFrame]]` — label + aggregate
dataframe. Color per label; consistent color map when called across
multiple charts in the same session.

### `app.py`

Minimal landing page. Explains the two sub-pages and shows a summary of
available seasons/scenarios. Streamlit auto-discovers `pages/`.

### `pages/1_Single_Run.py`

Sidebar:

- Season selectbox
- Scenario selectbox
- Version selectbox (default: latest)

Main area:

- **KPI row:** final active population, overall retention, mean match
  quality, mean rating error (computed from aggregate)
- **Charts:** the six chart functions above, stacked vertically
- **Config expander:** collapsible `st.json(config)`
- **Population preview:** day slider + filtered dataframe view (first
  1000 rows)

### `pages/2_Compare_Scenarios.py`

Sidebar:

- Season selectbox
- Multi-select scenarios (default: all in season)
- Metric selectbox for the focused chart

Main area:

- **Focused chart:** selected metric overlaid across scenarios
- **Small-multiples grid:** population, retention, match quality,
  favorite win prob — each overlaid across the selected scenarios
- **Metadata table:** one row per scenario with version, git_sha,
  elapsed_seconds, seed

Version policy: always use the latest version per scenario. Documented
in a caption on the page; revisit in v2 if needed.

## Data Flow

1. Page loads → loader lists seasons/scenarios/versions from disk
2. User picks selections → loader returns `ExperimentArtifacts` (cached)
3. Page calls chart functions with the relevant dataframes
4. `st.plotly_chart` renders them

All disk I/O goes through loader functions cached by
`@st.cache_data`. Cache key includes the artifact directory's `mtime`
so re-running a scenario invalidates automatically.

## Error Handling

Minimal — this is local research tooling:

- Empty `experiments/` → landing page shows a friendly "no experiments
  found" message with the expected path
- Missing parquet in a run directory → skip that run with
  `st.warning`, don't crash the page
- Schema drift (missing column) → let the exception surface; the user
  can regenerate

No retries, no input validation beyond what Streamlit widgets already
do.

## Testing

Light. Covered by:

- A unit test for `loader.list_seasons/scenarios/versions` against a
  tmp experiments directory fixture
- A unit test for one chart builder (e.g. `population_over_time`)
  asserting it returns a `go.Figure` with the expected number of
  traces given N input runs

No Streamlit end-to-end test. Manual smoke test: `just dashboard`,
click through both pages.

## Dependencies

Add to `pyproject.toml`:

- `streamlit`
- `plotly`

Add `just dashboard` recipe to `justfile`:

```make
dashboard:
    uv run streamlit run src/mm_sim/dashboard/app.py
```

## Open Questions (Deferred)

- Sweep Explorer page (v2)
- Version comparison within a single scenario (v2)
- Per-player drill-down from population parquet (v2)
