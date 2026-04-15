"""Tests for parameter sweeps."""

from __future__ import annotations

import pytest

from mm_sim.sweeps import set_nested


def test_set_nested_single_level():
    d = {"a": 1, "b": 2}
    set_nested(d, "a", 99)
    assert d == {"a": 99, "b": 2}


def test_set_nested_deep():
    d = {"config": {"matchmaker": {"kind": "composite"}}}
    set_nested(d, "config.matchmaker.kind", "random")
    assert d["config"]["matchmaker"]["kind"] == "random"


def test_set_nested_creates_missing_dicts():
    d = {}
    set_nested(d, "config.gear.transfer_rate", 0.1)
    assert d == {"config": {"gear": {"transfer_rate": 0.1}}}


def test_set_nested_into_existing_dict_leaf():
    d = {"config": {"matchmaker": {"composite_weights": {"skill": 1.0, "gear": 0.0}}}}
    set_nested(d, "config.matchmaker.composite_weights.skill", 0.5)
    assert d["config"]["matchmaker"]["composite_weights"]["skill"] == 0.5


def test_set_nested_rejects_empty_path():
    with pytest.raises(ValueError):
        set_nested({}, "", 1)


def test_sweep_grid_expansion(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    toml.write_text("""
name = "sk_gear"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.skill"
values = [0.0, 1.0]

[[sweep.grid]]
parameter = "config.matchmaker.composite_weights.gear"
values = [0.0, 0.5]
""")
    spec = SweepSpec.from_toml_file(toml)
    points = list(spec.iter_points())
    assert len(points) == 4
    labels = [p.label for p in points]
    assert labels == [
        "p0000_skill=0_gear=0",
        "p0001_skill=0_gear=0.5",
        "p0002_skill=1_gear=0",
        "p0003_skill=1_gear=0.5",
    ]
    assert points[1].overrides == {
        "config.matchmaker.composite_weights.skill": 0.0,
        "config.matchmaker.composite_weights.gear": 0.5,
    }


def test_sweep_zip_expansion(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    toml.write_text("""
name = "tradeoff"
base_scenario = "defaults"

[[sweep.zip]]
parameter = "config.matchmaker.composite_weights.skill"
values = [1.0, 0.5, 0.0]

[[sweep.zip]]
parameter = "config.matchmaker.composite_weights.gear"
values = [0.0, 0.5, 1.0]
""")
    spec = SweepSpec.from_toml_file(toml)
    points = list(spec.iter_points())
    assert len(points) == 3
    assert points[0].overrides == {
        "config.matchmaker.composite_weights.skill": 1.0,
        "config.matchmaker.composite_weights.gear": 0.0,
    }


def test_sweep_zip_rejects_unequal_lengths(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    toml.write_text("""
name = "bad"
base_scenario = "defaults"

[[sweep.zip]]
parameter = "a.b"
values = [1, 2, 3]

[[sweep.zip]]
parameter = "c.d"
values = [1, 2]
""")
    with pytest.raises(ValueError, match="zip"):
        SweepSpec.from_toml_file(toml)


def test_sweep_requires_exactly_one_mode(tmp_path):
    from mm_sim.sweeps import SweepSpec
    toml = tmp_path / "s.toml"
    toml.write_text("""
name = "bad"
base_scenario = "defaults"

[[sweep.grid]]
parameter = "a.b"
values = [1, 2]

[[sweep.zip]]
parameter = "c.d"
values = [1, 2]
""")
    with pytest.raises(ValueError):
        SweepSpec.from_toml_file(toml)


def test_materialize_point_applies_overrides():
    from mm_sim.sweeps import SweepPoint, materialize_point

    base_dict = {"config": {"matchmaker": {
        "kind": "composite",
        "composite_weights": {"skill": 1.0, "experience": 0.0, "gear": 0.0},
    }}}
    point = SweepPoint(
        index=0,
        label="p0000_skill=0.5",
        overrides={"config.matchmaker.composite_weights.skill": 0.5},
    )
    cfg = materialize_point(base_dict, point)
    assert cfg.matchmaker.composite_weights == {
        "skill": 0.5, "experience": 0.0, "gear": 0.0,
    }


def test_materialize_point_does_not_mutate_base():
    from mm_sim.sweeps import SweepPoint, materialize_point

    base_dict = {"config": {"matchmaker": {
        "kind": "composite",
        "composite_weights": {"skill": 1.0, "gear": 0.0},
    }}}
    point = SweepPoint(
        index=0, label="x",
        overrides={"config.matchmaker.composite_weights.skill": 0.1},
    )
    _ = materialize_point(base_dict, point)
    assert base_dict["config"]["matchmaker"]["composite_weights"]["skill"] == 1.0
