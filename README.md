# mm-player-base-simulations

A simulation for exploring how matchmaking system design shapes the
player base of a competitive game over a season. Players have hidden
true skill, an observed skill estimate that updates after matches,
experience, and gear. A matchmaker groups them into lobbies; an outcome
model decides who extracts and who dies; progression systems move skill,
gear, and season rank; a churn model drops players whose recent
experience was bad. The question is: **given different matchmaking and
progression choices, how does the shape and size of the active
population evolve?**

This is exploratory research code, not a product. Expect tuning knobs to
change, plots to get rewritten, and defaults to drift as the model gets
refined.

## What it simulates

A season is a sequence of daily ticks. On each tick:

1. Every active player is assigned a number of matches they want to play.
2. The **two-stage matchmaker** first forms teams (3 players) from
   un-teamed solos/duos, then assembles 4 teams into a 12-player lobby.
   Each stage has independent composite-rating weights over
   skill/experience/gear.
3. The **extraction outcome model** resolves each lobby as 4 independent
   teams that either extract (survive) or die. A softmax over team
   strengths samples the set of extracting teams; dead teams are
   attributed kills by the strongest extracting team that outranks them.
4. The **extract-Elo** rating updater moves `observed_skill` pairwise
   using extract-vs-wipe outcomes across teams.
5. Progression systems update state:
   - **Skill progression** drifts `true_skill` toward each player's
     talent ceiling.
   - **Gear** grows on extraction and transfers from dead teams to their
     killers, scaled by strength delta (punching-up earns more).
   - **Season progression** earns outcome-weighted, concave-in-activity
     season rank.
6. The **churn model** decides who quits today based on recent losses,
   blowouts, win streaks, and season-progression pressure — with a
   new-player sensitivity bonus so fragile players are hit harder.
7. New players arrive as a frozen fraction of the initial cohort.

Legacy 2-team win/loss mode and the single-stage matchmaker are still
supported for backwards compatibility with older experiments, but the
defaults now use the extraction model.

Everything is seeded, so runs are reproducible.

## What's in the repo

```text
src/mm_sim/
  engine.py                 # daily tick loop
  population.py             # struct-of-numpy-arrays player state
  matchmaker/
    two_stage.py            # team formation + lobby assembly
    ...                     # random + single-stage composite
  outcomes/
    extraction.py           # multi-team extract/wipe outcome
    softmax_winners.py      # softmax sampling of extracting teams
    default.py              # legacy 2-team win/loss
  rating_updaters/
    elo_extract.py          # pairwise Elo over extract/wipe
    ...                     # legacy Elo, KPM
  skill_progression.py      # true_skill drift toward talent_ceiling
  gear.py                   # extract-growth + directed transfer
  season_progression.py     # outcome-weighted concave earn + pressure
  churn.py                  # daily quit-probability model
  experience.py             # experience progression
  frequency.py              # matches-per-day sampling
  parties.py                # static party assignments
  snapshot.py               # per-day + per-match + per-player logs
  experiments.py            # ExperimentRunner: run, persist, load
  scenarios.py              # TOML-based scenario system
  sweeps.py                 # parameter-sweep runner (grid / zip)
  sweep_plots.py            # 1D line + 2D heatmap sweep plots
  plots.py                  # per-experiment visualizations
  compare.py                # cross-scenario comparison plots
  dashboard/                # Streamlit dashboard (multi-page)
  cli.py                    # python -m mm_sim.cli <subcommand>

scenarios/                  # one TOML file per named scenario + defaults
experiments/                # output: experiments/<season>/<scenario>/<version>/
tests/                      # pytest
docs/superpowers/           # design specs + implementation plans
```

## Setup

**Prerequisites:**

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) — install with
  `curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`
  on macOS, or see the uv docs for other platforms
- [just](https://github.com/casey/just) (optional, for the recipe
  shortcuts) — `brew install just` or see its install docs

Clone and set up the local venv:

```bash
git clone https://github.com/phenrickson/mm-player-base-simulations.git
cd mm-player-base-simulations
just setup
```

That runs `uv sync`, which creates `.venv/` and installs everything
(including Streamlit and Plotly for the dashboard). If you're not using
just, run `uv sync` directly.

## Running simulations

Scenarios live in `scenarios/*.toml`. `scenarios/defaults.toml` holds
shared config (season name, initial population, matchmaker/outcome kind,
progression knobs, churn weights, etc.) that every scenario inherits and
may override.

```bash
# list available scenarios
just scenarios-list

# run one scenario
just scenario skill_only

# run every scenario in scenarios/
just scenarios

# show a saved experiment (latest version)
just experiment skill_only

# list all saved experiments (one row per version)
just experiments

# regenerate plots for an already-saved experiment
just plots skill_only
```

Each run is saved under `experiments/<season>/<scenario>/<version>/`:

```text
experiments/my-season/skill_only/v1/
  metadata.json           # run info
  config.json             # full resolved config
  aggregate.parquet       # one row per day
  population.parquet      # one row per (day, player_id)
  matches.parquet         # one row per match
  match_teams.parquet     # one row per (match, team) — extraction mode
  plots/                  # PNGs — individual panels + grouped figures
```

The season name is computed from `defaults.toml` as
`{days}d-{initial_size}p-{skill_distribution}-{suffix}` so it
auto-updates when you change the underlying config. Bump
`season_suffix` when you start a new research session with different
base settings so results don't mix with earlier runs.

## Comparing scenarios

Once multiple scenarios have run in the same season:

```bash
# compare all scenarios in the current season
just compare

# compare specific scenarios
just compare skill_only random_mm

# compare scenarios in an older season (first arg is the season name
# if it matches a directory under experiments/)
just compare 90d-15000p-right_skewed-multi-team-extraction
```

Comparison plots land in `experiments/<season>/_comparisons/`.

## Parameter sweeps

Sweep TOMLs declare a `base_scenario` and one or more parameters to
vary, either as a full grid or as a zipped set of points. The runner
materializes a `SimulationConfig` per point, runs each sequentially,
and writes a `sweep.json` manifest alongside per-point experiments.

```bash
# list available sweeps
just sweeps-list

# run a sweep
just sweep sweep_skill_weight

# regenerate sweep plots from saved runs
just sweep-compare sweep_skill_weight

# overlay a sweep with named reference scenarios
just sweep-overlay sweep_mm_skill_weight --reference random_mm
```

1D sweeps produce line plots; 2D grids produce heatmaps. Overlays drop
reference-scenario points onto the same axes for context.

See `scenarios/sweep_*.toml` for examples and
`docs/superpowers/plans/2026-04-14-parameter-sweeps.md` for the design.

## Dashboard

An interactive Streamlit dashboard for exploring saved experiments:

```bash
just dashboard
```

It has three pages:

- **Landing** — pick a season and experiment.
- **Single Run** — aggregate, cohorts, player trajectories, match
  detail, and per-team extraction views for one experiment.
- **Compare Scenarios** — side-by-side metrics across scenarios, with
  consistent colors for sweep points and reference categories.

The dashboard reads directly from `experiments/` and caches loads via
`streamlit.cache_data`.

## Adding a scenario

1. Copy an existing file in `scenarios/` to a new name
2. Edit the `name` field to match the filename (without `.toml`)
3. Override the config fields you care about
4. Run `just scenario <name>`

Only fields you set are overridden; the rest come from
`scenarios/defaults.toml` and the pydantic defaults in
`src/mm_sim/config.py`.

## Cleaning up

```bash
# delete ALL saved experiments (prompts for confirmation)
just clean-experiments
```

## Tests

```bash
just test             # full suite
just test-fast        # stop on first failure
just test-one tests/test_cli.py
```

## Design docs

Design specs and implementation plans for major features live under
`docs/superpowers/`:

- `specs/2026-04-15-multi-team-extraction-design.md`
- `specs/2026-04-15-streamlit-dashboard-design.md`
- `plans/2026-04-14-progression-systems.md`
- `plans/2026-04-14-parameter-sweeps.md`
- `plans/2026-04-15-multi-team-extraction.md`
- `plans/2026-04-15-streamlit-dashboard.md`
- `plans/2026-04-16-softmax-extraction-outcome.md`
