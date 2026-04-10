# mm-player-base-simulations

A simulation for exploring how matchmaking system design shapes the player
base of a competitive game over a season. Players have hidden true skill,
an observed skill estimate that updates after matches, experience, and
gear. A matchmaker groups them into lobbies; an outcome model decides who
wins; a churn model drops players whose recent experience was bad. The
question is: **given different matchmaking strategies, how does the
shape and size of the active population evolve?**

This is exploratory research code, not a product. Expect tuning knobs to
change, plots to get rewritten, and defaults to drift as the model gets
refined.

## What it simulates

A season is a sequence of daily ticks. On each tick:

1. Every active player is assigned a number of matches they want to play.
2. The matchmaker groups searching players into lobbies (random, or
   composite rating over skill/experience/gear).
3. The outcome model picks a winner for each match based on team true
   skill plus noise, flagging blowouts.
4. The rating updater (Elo) moves `observed_skill` based on outcomes.
5. Experience and gear update from match activity.
6. The churn model decides who quits today based on recent losses,
   blowouts, and wins — with a new-player sensitivity multiplier so
   fragile players are hit harder by losses.
7. New players arrive as a frozen fraction of the initial cohort.

Everything is seeded, so runs are reproducible.

## What's in the repo

```text
src/mm_sim/
  engine.py          # daily tick loop
  population.py      # struct-of-numpy-arrays player state
  matchmaker/        # random + composite-rating matchmakers
  outcomes/          # match outcome model
  rating_updaters/   # Elo, KPM
  churn.py           # daily quit-probability model
  experience.py      # experience progression
  gear.py            # gear progression
  frequency.py       # matches-per-day sampling
  parties.py         # static party assignments
  snapshot.py        # per-day aggregate + per-match + per-player logs
  experiments.py     # ExperimentRunner: run, persist, load
  scenarios.py       # TOML-based scenario system
  plots.py           # per-experiment visualizations
  compare.py         # cross-scenario comparison plots
  cli.py             # python -m mm_sim.cli <subcommand>

scenarios/           # one TOML file per named scenario + defaults.toml
experiments/         # output: experiments/<season>/<scenario>/<version>/
tests/               # pytest
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

That runs `uv sync`, which creates `.venv/` and installs everything.
If you're not using just, run `uv sync` directly.

## Running simulations

Scenarios live in `scenarios/*.toml`. `scenarios/defaults.toml` holds
shared config (season name, initial population, churn weights, etc.)
that every scenario inherits and may override.

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
  metadata.json         # run info
  config.json           # full resolved config
  aggregate.parquet     # one row per day
  population.parquet    # one row per (day, player_id)
  matches.parquet       # one row per match with quality metrics
  plots/                # PNGs — individual panels + grouped figures
```

The season name comes from `scenarios/defaults.toml`; change it there
when you start a new research session with different base settings so
results don't mix with earlier runs.

## Comparing scenarios

Once multiple scenarios have run in the same season:

```bash
# compare all scenarios in the current season
just compare

# compare specific scenarios
just compare skill_only random_mm

# compare scenarios in an older season (first arg is the season name
# if it matches a directory under experiments/)
just compare baseline-90day-asymmetric
```

Comparison plots land in `experiments/<season>/_comparisons/`.

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
