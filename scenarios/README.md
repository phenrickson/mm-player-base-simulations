# Scenarios

Named configurations for the matchmaking simulation. Each `.toml` file
is one scenario or parameter sweep. Running a scenario produces an
experiment in `../experiments/` via the experiment tracker.

## Running scenarios

```bash
# run a single scenario
just scenario skill_only

# run every scenario in this directory (excludes sweep files)
just scenarios

# list what scenarios are available
just scenarios-list
```

Re-running a scenario does not overwrite previous experiments; the
experiment tracker auto-versions them (`v1`, `v2`, ...).

## Scenario file format

```toml
name = "scenario_name"
category = "matchmaker"  # optional — used for dashboard coloring

[config]
seed = 1999
season_days = 30
# ... only fields that differ from SimulationConfig defaults
```

You only need to set the fields you care about for that scenario;
everything else falls through to `defaults.toml` and then to the
pydantic defaults in `src/mm_sim/config.py`.

## Current scenarios

Defaults use the multi-team extraction mode (4 teams × 3 players) with
a two-stage matchmaker (team formation + lobby assembly), extract-Elo
rating, and skill/gear/season progression all enabled.

- **`skill_only`** — Pure skill weights at both matchmaker stages.
  Baseline for "good MM."
- **`experience_only`** — Experience-based matchmaking at both stages
  (no skill signal). Produces the feedback loop.
- **`random_mm`** — No matchmaking at all. Floor of match quality —
  calibrates how much MM contributes.

## Sweeps

Sweep files drive parameter sweeps via `just sweep <name>`. They inherit
from a `base_scenario` and enumerate parameter values as a grid or a
zipped set of points.

- **`sweep_skill_weight`** — 2D grid over team-formation and
  lobby-assembly skill weights.
- **`sweep_mm_skill_weight`** — 1D sweep of matchmaker skill weight,
  designed for overlay with reference scenarios.
- **`sweep_skill_gear_grid`** — 2D grid over skill and gear weights.

See `../README.md` for the sweep CLI (`just sweep`, `just sweep-compare`,
`just sweep-overlay`).

## Adding a new scenario

1. Copy an existing `.toml` file to a new name
2. Edit `name` to match the filename (without `.toml`)
3. Override the config fields that matter for your experiment
4. Run `just scenario <name>`
5. Inspect with `just experiment <name>`, `just dashboard`, or load it
   in Python with `from mm_sim.experiments import load_experiment`

## Adding a new sweep

1. Create `scenarios/sweep_<name>.toml` with:

   ```toml
   name = "sweep_<name>"
   base_scenario = "defaults"  # or any other scenario name

   [[sweep.grid]]
   parameter = "config.some.dotted.path"
   values = [0.1, 0.2, 0.3]
   ```

   Use `[[sweep.grid]]` blocks for full-factorial grids, or
   `[[sweep.zip]]` blocks for parallel-arrays (same length, walked in
   lockstep).
2. Run `just sweep sweep_<name>`
3. Inspect with `just sweep-compare sweep_<name>` or the dashboard.
