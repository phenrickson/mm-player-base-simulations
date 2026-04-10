# CLI Orchestration Design

**Date:** 2026-04-10
**Status:** Approved

## Problem

The justfile embeds Python source code in `uv run python -c "..."` strings. This is untestable, unlintable, duplicates logic across `experiment`/`experiment-version` and `plots`/`plots-version`, and reaches into a private helper (`_latest_version_dir`). The `sim` recipe at `justfile:18` references a `mm_sim.cli` module that does not exist, so that recipe is already broken.

## Goal

Replace the inlined Python with a real CLI module. The justfile remains the orchestration layer; recipes become thin one-liners that invoke `python -m mm_sim.cli <subcommand>`.

## Design

### New file: `src/mm_sim/cli.py`

A single module containing:

- One `cmd_*` function per subcommand, each a thin wrapper around existing functions in `experiments.py`, `scenarios.py`, `plots.py`.
- A `main()` that builds an `argparse.ArgumentParser` with flat subcommands and dispatches via `args.func(args)`.
- `if __name__ == "__main__": main()` so `python -m mm_sim.cli` works.

Uses `argparse` (stdlib, no new deps).

### Subcommands

Flat layout — no nested command groups.

| Subcommand | Args | Underlying call |
|---|---|---|
| `sim` | — | runs default simulation |
| `experiments` | — | `list_experiments()` |
| `experiment` | `NAME [--version V]` | `load_experiment(NAME, version=V)` |
| `scenario` | `NAME` | `run_scenario(NAME)` |
| `scenarios` | — | `run_all_scenarios()` |
| `plots` | `NAME [--version V]` | resolves version dir, calls `generate_plots_for_experiment_dir` |

`experiment` and `plots` each take an optional `--version` flag, collapsing the old `-version` recipe pairs into single commands.

### Supporting change: `src/mm_sim/experiments.py`

Rename `_latest_version_dir` → `latest_version_dir` (drop the leading underscore). Update internal callers. This lets `cli.py` call it without reaching into private API.

### Justfile after

```make
sim:
    uv run python -m mm_sim.cli sim

experiments:
    uv run python -m mm_sim.cli experiments

experiment NAME *ARGS:
    uv run python -m mm_sim.cli experiment {{NAME}} {{ARGS}}

scenario NAME:
    uv run python -m mm_sim.cli scenario {{NAME}}

scenarios:
    uv run python -m mm_sim.cli scenarios

scenarios-list:
    @ls scenarios/*.toml 2>/dev/null | sed 's|scenarios/||; s|\.toml||'

plots NAME *ARGS:
    uv run python -m mm_sim.cli plots {{NAME}} {{ARGS}}
```

The `experiment-version` and `plots-version` recipes are removed. Callers pass `--version` through: `just experiment foo --version v2`.

The `test`, `test-fast`, `test-one`, and `lock` recipes are unchanged.

### Tests: `tests/test_cli.py`

One test per subcommand. Each test patches `sys.argv`, calls `cli.main()`, and asserts the underlying function was called with the expected arguments (mocked at the `mm_sim.cli` module boundary with `unittest.mock.patch`). No real simulations run; tests stay fast.

## Out of scope

- No Typer / Click / other arg-parsing library.
- No console_scripts entry point in `pyproject.toml` — `python -m mm_sim.cli` is enough.
- No restructuring of `experiments.py` / `scenarios.py` / `plots.py` beyond the one rename.
- No nested subcommand groups.
