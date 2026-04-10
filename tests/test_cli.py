"""Smoke tests for the mm_sim.cli dispatcher.

Each test patches the underlying function at the cli module boundary
and asserts the CLI parsed args into the expected call. No real
simulations run here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mm_sim import cli


def test_experiments_subcommand_calls_list_experiments():
    with patch.object(cli, "list_experiments") as mock_list:
        mock_list.return_value = "fake-df"
        cli.main(["experiments"])
        mock_list.assert_called_once_with()


def test_experiment_subcommand_default_version():
    with patch.object(cli, "load_experiment") as mock_load:
        mock_load.return_value = MagicMock(metadata="m", aggregate="a")
        cli.main(["experiment", "foo"])
        mock_load.assert_called_once_with("foo", version=None)


def test_experiment_subcommand_explicit_version():
    with patch.object(cli, "load_experiment") as mock_load:
        mock_load.return_value = MagicMock(metadata="m", aggregate="a")
        cli.main(["experiment", "foo", "--version", "v3"])
        mock_load.assert_called_once_with("foo", version="v3")


def test_scenario_subcommand_calls_run_scenario():
    with patch.object(cli, "run_scenario") as mock_run:
        mock_run.return_value = MagicMock(
            metadata=MagicMock(name="foo", elapsed_seconds=1.0)
        )
        cli.main(["scenario", "foo"])
        mock_run.assert_called_once_with("foo")


def test_scenarios_subcommand_calls_run_all_scenarios():
    with patch.object(cli, "run_all_scenarios") as mock_all:
        mock_all.return_value = []
        cli.main(["scenarios"])
        mock_all.assert_called_once_with()


def test_plots_subcommand_default_version_uses_latest(tmp_path):
    season_dir = tmp_path
    (season_dir / "foo").mkdir()
    with (
        patch.object(cli, "DEFAULT_EXPERIMENTS_DIR", season_dir),
        patch.object(cli, "latest_version_dir") as mock_latest,
        patch.object(cli, "generate_plots_for_experiment_dir") as mock_plots,
    ):
        mock_latest.return_value = season_dir / "foo" / "v7"
        mock_plots.return_value = []
        cli.main(["plots", "foo"])
        mock_latest.assert_called_once_with(season_dir, "foo")
        mock_plots.assert_called_once_with(season_dir / "foo" / "v7")


def test_plots_subcommand_explicit_version(tmp_path):
    season_dir = tmp_path
    version_dir = season_dir / "foo" / "v2"
    version_dir.mkdir(parents=True)
    with (
        patch.object(cli, "DEFAULT_EXPERIMENTS_DIR", season_dir),
        patch.object(cli, "generate_plots_for_experiment_dir") as mock_plots,
    ):
        mock_plots.return_value = []
        cli.main(["plots", "foo", "--version", "v2"])
        mock_plots.assert_called_once_with(version_dir)


def test_no_subcommand_is_an_error():
    with pytest.raises(SystemExit):
        cli.main([])
