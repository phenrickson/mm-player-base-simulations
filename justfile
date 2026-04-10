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

# refresh the uv lock file
lock:
    uv lock
