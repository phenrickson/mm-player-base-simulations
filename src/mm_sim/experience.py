"""Experience: monotonic, normalized to [0, 1] via a reference max."""

from __future__ import annotations

import numpy as np

from mm_sim.population import Population


def apply_experience_update(
    pop: Population,
    matches_played_this_tick: np.ndarray,
    normalization_max_matches: int,
) -> None:
    pop.matches_played += matches_played_this_tick
    pop.experience = np.clip(
        pop.matches_played / float(normalization_max_matches), 0.0, 1.0
    ).astype(np.float32)
