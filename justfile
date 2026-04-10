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

# refresh the uv lock file
lock:
    uv lock
