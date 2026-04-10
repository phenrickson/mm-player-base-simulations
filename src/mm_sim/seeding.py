"""Deterministic RNG factory. All random draws must flow through here."""

from __future__ import annotations

import hashlib
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a seeded numpy Generator."""
    return np.random.default_rng(seed)


def spawn_child(parent: np.random.Generator, name: str) -> np.random.Generator:
    """Spawn a named child RNG from a parent.

    Naming the child keeps streams independent and reproducible: two calls
    with the same parent state and the same name return the same stream.
    """
    name_hash = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    parent_draw = int(parent.integers(0, 2**63 - 1))
    child_seed = (parent_draw ^ name_hash) & ((1 << 63) - 1)
    return np.random.default_rng(child_seed)
