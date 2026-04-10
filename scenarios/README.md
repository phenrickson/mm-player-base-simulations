# Scenarios

Research recipes for the matchmaking simulation. Each `.toml` file is a
named configuration with an attached hypothesis. Running a scenario
produces an experiment in `../experiments/` via the experiment tracker.

## Running scenarios

```bash
# run a single scenario
just scenario skill_only

# run every scenario in this directory
just scenarios

# list what scenarios are available
just scenarios-list
```

Re-running a scenario does not overwrite previous experiments; the
experiment tracker auto-versions them (`skill_only`, `skill_only_v2`, ...).

## Scenario file format

```toml
name = "scenario_name"
hypothesis = """
What you expect to happen and why. This is saved into the experiment's
metadata.json so future-you knows what past-you was testing.
"""

[config]
seed = 1999
season_days = 30
# ... only fields that differ from SimulationConfig defaults
```

You only need to set the fields you care about for that scenario;
everything else falls through to the defaults in `src/mm_sim/config.py`.

## Current scenarios

- **`skill_only`** — Baseline. Pure skill-based composite matchmaker, solo
  parties. Expected: Elo learns quickly, blowouts die off, no feedback loop.
- **`experience_only`** — Level-based MM (no skill signal). Expected to
  produce the feedback loop because low- and high-skill players will be
  matched together based on time played, not ability.
- **`random_mm`** — No matchmaking at all. The absolute floor of match
  quality. Useful for calibrating how much MM contributes.
- **`homogeneous_trios`** — Skill-based MM but parties are tight-knit
  high-skill trios. Tests whether party stacking alone produces the
  feedback loop even when individual skill estimates are good.

## Adding a new scenario

1. Copy an existing `.toml` file to a new name
2. Edit `name` to match the filename (without `.toml`)
3. Write a hypothesis — what are you testing? What do you expect?
4. Override the config fields that matter for your experiment
5. Run `just scenario <name>`
6. Inspect with `just experiment <name>` or load it in Python with
   `from mm_sim.experiments import load_experiment`
