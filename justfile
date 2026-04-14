set shell := ["pwsh.exe", "-c"]

_default:
    @just --list

# install dependencies and build the local .venv
setup:
    uv sync

# run the full test suite
test:
    uv run pytest

# run tests and stop on first failure
test-fast:
    uv run pytest -x

# run a single test file or node id
test-one target:
    uv run pytest {{target}} -v

# list all saved experiments (one row per version)
experiments:
    uv run python -m mm_sim.cli experiments

# show an experiment (latest version by default, or pass --version v2)
experiment NAME *ARGS:
    uv run python -m mm_sim.cli experiment {{NAME}} {{ARGS}}

# run a single scenario by name (looks up scenarios/NAME.toml)
scenario NAME:
    uv run python -m mm_sim.cli scenario {{NAME}}

# run every scenario in the scenarios/ directory
scenarios:
    uv run python -m mm_sim.cli scenarios

# list scenario files available to run
scenarios-list:
    Get-ChildItem scenarios -Filter *.toml -ErrorAction SilentlyContinue |
    ForEach-Object { $_.BaseName } `
    || echo "no scenarios/ directory yet"

# regenerate plots for a saved experiment (latest version by default, or pass --version v2)
plots NAME *ARGS:
    uv run python -m mm_sim.cli plots {{NAME}} {{ARGS}}

# generate cross-scenario comparison plots. first arg can be a season name;
# remaining args are scenario names. with no args, uses the current season.
compare *ARGS:
    uv run python -m mm_sim.cli compare {{ARGS}}

# delete ALL saved experiments (prompts for confirmation)
clean-experiments:
    @printf "delete everything under experiments/? [y/N] " && read ans && [ "$ans" = "y" ] && rm -rf experiments && echo "deleted." || echo "aborted."

# refresh the uv lock file
lock:
    uv lock
