"""Softmax-based winner-sampling helpers for extraction outcomes.

Two pure functions, no state:

- ``sample_extractor_count`` draws k (teams that extract this match) from a
  three-component mixture: a small P(k=0) spike, a small P(k=n_teams) spike,
  and a shifted Binomial in between, parameterized to hit a target mean.

- ``plackett_luce_marginals`` computes, for each team, the probability that
  it is among the top-k winners when k teams are sampled without replacement
  from ``softmax(beta * strengths)``. Exact via ordering enumeration; cheap
  for ``n_teams <= 8``.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def sample_extractor_count(
    n_teams: int,
    mean_extractors: float,
    p_zero: float,
    p_all: float,
    rng: np.random.Generator,
) -> int:
    """Draw k, the number of extracting teams this match.

    Three-component mixture:
      - with probability ``p_zero``: k = 0
      - with probability ``p_all``:  k = n_teams
      - with the remaining mass:     k ~ Binomial(n_teams - 1, p) + 1

    The Binomial ``p`` is solved so that the overall expected k equals
    ``mean_extractors``.
    """
    r = rng.random()
    if r < p_zero:
        return 0
    if r < p_zero + p_all:
        return n_teams
    p_mid_mass = 1.0 - p_zero - p_all
    if p_mid_mass <= 0.0:
        return 1
    mid_mean = (mean_extractors - p_all * n_teams) / p_mid_mass
    p = (mid_mean - 1.0) / (n_teams - 1)
    p = max(0.0, min(1.0, p))
    return int(rng.binomial(n_teams - 1, p)) + 1


def plackett_luce_marginals(
    strengths: np.ndarray,
    k: int,
    beta: float,
) -> np.ndarray:
    """Per-team probability of being among the top-k winners.

    Winners are sampled without replacement from ``softmax(beta * strengths)``.
    This function exactly enumerates every ordered length-k pick.
    """
    n = int(strengths.shape[0])
    if k <= 0:
        return np.zeros(n, dtype=np.float64)
    if k >= n:
        return np.ones(n, dtype=np.float64)

    shifted = beta * strengths.astype(np.float64)
    shifted = shifted - shifted.max()
    w = np.exp(shifted)

    marginals = np.zeros(n, dtype=np.float64)
    for ordering in permutations(range(n), k):
        remaining_mask = np.ones(n, dtype=bool)
        prob = 1.0
        for pick in ordering:
            total = float(w[remaining_mask].sum())
            prob *= float(w[pick]) / total
            remaining_mask[pick] = False
        for t in ordering:
            marginals[t] += prob
    return marginals
