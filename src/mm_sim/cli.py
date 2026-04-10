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

from mm_sim.experiments import (
    DEFAULT_EXPERIMENTS_DIR,
    _find_latest_season_for_experiment,
    latest_version_dir,
    list_experiments,
    load_experiment,
)
from mm_sim.plots import generate_plots_for_experiment_dir
from mm_sim.scenarios import run_all_scenarios, run_scenario


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
