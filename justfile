set windows-shell := ["pwsh.exe", "-c"]

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
    @uv run python -c "import glob, os, tomllib; names=[]; [names.append(os.path.splitext(os.path.basename(p))[0]) for p in sorted(glob.glob('scenarios/*.toml')) if os.path.basename(p) != 'defaults.toml' and 'sweep' not in tomllib.loads(open(p).read())]; print('\n'.join(names) if names else 'no scenarios/ directory yet')"

# regenerate plots for a saved experiment (latest version by default, or pass --version v2)
plots NAME *ARGS:
    uv run python -m mm_sim.cli plots {{NAME}} {{ARGS}}

# generate cross-scenario comparison plots. first arg can be a season name;
# remaining args are scenario names. with no args, uses the current season.
compare *ARGS:
    uv run python -m mm_sim.cli compare {{ARGS}}

# run a parameter sweep by name (looks up scenarios/NAME.toml)
sweep NAME:
    uv run python -m mm_sim.cli sweep {{NAME}}

# list sweep files available to run
sweeps-list:
    @uv run python -m mm_sim.cli sweeps

# regenerate sweep comparison plots (latest version by default, or pass --version v2)
sweep-compare NAME *ARGS:
    uv run python -m mm_sim.cli sweep-compare {{NAME}} {{ARGS}}

# overlay a sweep with named reference scenarios, e.g.
#   just sweep-overlay sweep_mm_skill_weight --reference random_mm
sweep-overlay NAME *ARGS:
    uv run python -m mm_sim.cli sweep-overlay {{NAME}} {{ARGS}}

# delete ALL saved experiments (prompts for confirmation)
clean-experiments:
    @printf "delete everything under experiments/? [y/N] " && read ans && [ "$ans" = "y" ] && rm -rf experiments && echo "deleted." || echo "aborted."

# refresh the uv lock file
lock:
    uv lock

# Launch the Streamlit dashboard
dashboard:
    uv run streamlit run src/mm_sim/dashboard/app.py
