"""Command-line interface for mm_sim.

Invoked as `python -m mm_sim.cli <subcommand> [args]`. The justfile
wraps these as recipes; this module is the single place where those
commands actually live.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from mm_sim.compare import compare_scenarios
from mm_sim.experiments import (
    DEFAULT_EXPERIMENTS_DIR,
    _find_latest_season_for_experiment,
    latest_version_dir,
    list_experiments,
    load_experiment,
)
from mm_sim.plots import generate_plots_for_experiment_dir
from mm_sim.scenarios import (
    DEFAULT_SCENARIOS_DIR,
    load_season_name,
    run_all_scenarios,
    run_scenario,
)
from mm_sim.sweep_plots import plot_sweep
from mm_sim.sweeps import list_sweeps, run_sweep


def cmd_experiments(args: argparse.Namespace) -> None:
    pl.Config.set_tbl_rows(100)
    print(list_experiments())


def cmd_experiment(args: argparse.Namespace) -> None:
    pl.Config.set_tbl_rows(100)
    exp = load_experiment(args.name, version=args.version)
    print(exp.metadata)
    print(exp.aggregate)


def cmd_scenario(args: argparse.Namespace) -> None:
    exp = run_scenario(args.name)
    print(f"saved: {exp.metadata.name} ({exp.metadata.elapsed_seconds}s)")


def cmd_scenarios(args: argparse.Namespace) -> None:
    experiments = run_all_scenarios()
    for exp in experiments:
        print(f"saved: {exp.metadata.name} ({exp.metadata.elapsed_seconds}s)")


def cmd_compare(args: argparse.Namespace) -> None:
    # First positional arg is treated as the season if it matches a
    # directory under experiments/. Otherwise all positional args are
    # scenario names in the current season.
    season: str | None = None
    names: list[str] = list(args.args)
    if names and (Path(DEFAULT_EXPERIMENTS_DIR) / names[0]).is_dir():
        season = names.pop(0)
    paths = compare_scenarios(
        names=names if names else None, season=season
    )
    for path in paths:
        print(path)


def cmd_plots(args: argparse.Namespace) -> None:
    base = Path(DEFAULT_EXPERIMENTS_DIR)
    season_dir = (
        base
        if (base / args.name).exists()
        else _find_latest_season_for_experiment(base, args.name)
    )
    if args.version is None:
        version_dir = latest_version_dir(season_dir, args.name)
    else:
        version_dir = season_dir / args.name / args.version
        if not version_dir.exists():
            raise FileNotFoundError(f"version not found: {version_dir}")
    paths = generate_plots_for_experiment_dir(version_dir)
    for path in paths:
        print(path)


def cmd_sweep(args: argparse.Namespace) -> None:
    result = run_sweep(args.name)
    plot_sweep(result.sweep_dir)
    print(f"sweep saved: {result.sweep_dir}")
    print(f"points: {len(result.point_experiments)}")


def cmd_sweeps(args: argparse.Namespace) -> None:
    for name in list_sweeps():
        print(name)


def cmd_sweep_compare(args: argparse.Namespace) -> None:
    season = load_season_name(DEFAULT_SCENARIOS_DIR)
    sweep_parent = Path(DEFAULT_EXPERIMENTS_DIR) / season / args.name
    if not sweep_parent.exists():
        raise FileNotFoundError(f"sweep not found: {sweep_parent}")
    if args.version is None:
        versions = sorted(
            p for p in sweep_parent.iterdir() if p.name.startswith("v")
        )
        if not versions:
            raise FileNotFoundError(f"no versions under {sweep_parent}")
        sweep_dir = versions[-1]
    else:
        sweep_dir = sweep_parent / args.version
        if not sweep_dir.exists():
            raise FileNotFoundError(f"version not found: {sweep_dir}")
    plot_sweep(sweep_dir)
    print(f"regenerated plots: {sweep_dir / 'plots'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mm_sim")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("experiments", help="list all saved experiments")
    p.set_defaults(func=cmd_experiments)

    p = sub.add_parser("experiment", help="show one experiment")
    p.add_argument("name")
    p.add_argument("--version", default=None)
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("scenario", help="run one scenario by name")
    p.add_argument("name")
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("scenarios", help="run every scenario in scenarios/")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("plots", help="regenerate plots for a saved experiment")
    p.add_argument("name")
    p.add_argument("--version", default=None)
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser(
        "compare",
        help="generate cross-scenario comparison plots for a season",
    )
    p.add_argument(
        "args",
        nargs="*",
        help=(
            "optional season name (first arg, if it matches a directory "
            "under experiments/) followed by scenario names. "
            "defaults to all scenarios in the current season."
        ),
    )
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="run a parameter sweep by name")
    p.add_argument("name")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("sweeps", help="list sweep files in scenarios/")
    p.set_defaults(func=cmd_sweeps)

    p = sub.add_parser(
        "sweep-compare",
        help="regenerate sweep comparison plots (latest version by default)",
    )
    p.add_argument("name")
    p.add_argument("--version", default=None)
    p.set_defaults(func=cmd_sweep_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
