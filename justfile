_default:
    @just --list

# run the full test suite
test:
    uv run pytest

# run tests and stop on first failure
test-fast:
    uv run pytest -x

# run a single test file or node id
test-one target:
    uv run pytest {{target}} -v

# run a default simulation via the mm_sim CLI
sim:
    uv run python -m mm_sim.cli

# list all saved experiments
experiments:
    uv run python -c "from mm_sim.experiments import list_experiments; import polars as pl; pl.Config.set_tbl_rows(100); print(list_experiments())"

# show the aggregate snapshot of one experiment
experiment NAME:
    uv run python -c "from mm_sim.experiments import load_experiment; import polars as pl; pl.Config.set_tbl_rows(100); e = load_experiment('{{NAME}}'); print(e.metadata); print(e.aggregate)"

# run a single scenario by name (looks up scenarios/NAME.toml)
scenario NAME:
    uv run python -c "from mm_sim.scenarios import run_scenario; e = run_scenario('{{NAME}}'); print(f'saved: {e.metadata.name} ({e.metadata.elapsed_seconds}s)')"

# run every scenario in the scenarios/ directory
scenarios:
    uv run python -c "from mm_sim.scenarios import run_all_scenarios; exps = run_all_scenarios(); [print(f'saved: {e.metadata.name} ({e.metadata.elapsed_seconds}s)') for e in exps]"

# list scenario files available to run
scenarios-list:
    @ls scenarios/*.toml 2>/dev/null | sed 's|scenarios/||; s|\.toml||' || echo "no scenarios/ directory yet"

# regenerate plots for a saved experiment
plots NAME:
    uv run python -c "from pathlib import Path; from mm_sim.plots import generate_plots_for_experiment_dir; paths = generate_plots_for_experiment_dir(Path('experiments') / '{{NAME}}'); print('\n'.join(str(p) for p in paths))"

# refresh the uv lock file
lock:
    uv lock
