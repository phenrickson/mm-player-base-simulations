"""Tests for the dashboard loader module."""
from __future__ import annotations

from pathlib import Path

import pytest

from mm_sim.dashboard import loader


@pytest.fixture
def experiments_tree(tmp_path: Path) -> Path:
    """Build a minimal experiments/ tree with two seasons and two scenarios."""
    root = tmp_path / "experiments"
    for season in ("season-a", "season-b"):
        for scen in ("skill_only", "random_mm"):
            for ver in ("v1", "v2"):
                d = root / season / scen / ver
                d.mkdir(parents=True)
                (d / "metadata.json").write_text("{}")
    return root


def test_list_seasons_returns_sorted_season_names(experiments_tree: Path):
    assert loader.list_seasons(experiments_tree) == ["season-a", "season-b"]


def test_list_seasons_returns_empty_when_missing(tmp_path: Path):
    assert loader.list_seasons(tmp_path / "missing") == []
