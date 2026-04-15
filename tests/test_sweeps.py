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
