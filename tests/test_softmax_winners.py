"""Tests for softmax-based winner sampling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mm_sim.outcomes.softmax_winners import (
    plackett_luce_marginals,
    sample_extractor_count,
)


def test_marginals_sum_to_k():
    strengths = np.array([0.2, 0.5, 0.8, 1.1], dtype=np.float32)
    for k in range(5):
        m = plackett_luce_marginals(strengths, k=k, beta=2.0)
        assert m.shape == (4,)
        assert pytest.approx(m.sum(), abs=1e-5) == float(k)


def test_marginals_stronger_team_higher():
    """With beta>0, a stronger team should have a higher marginal."""
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=2, beta=2.0)
    assert m[0] < m[1] < m[2] < m[3]


def test_marginals_beta_zero_is_uniform():
    """beta=0 means strengths are ignored; every team has marginal k/n."""
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=2, beta=0.0)
    np.testing.assert_allclose(m, [0.5, 0.5, 0.5, 0.5], atol=1e-6)


def test_marginals_k_zero_all_zero():
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=0, beta=2.0)
    np.testing.assert_allclose(m, [0.0, 0.0, 0.0, 0.0])


def test_marginals_k_equals_n_all_one():
    strengths = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=4, beta=2.0)
    np.testing.assert_allclose(m, [1.0, 1.0, 1.0, 1.0])


def test_marginals_hand_computed_values():
    """Reference values computed out of band:
    strengths = [0.2, 0.5, 0.8, 1.1], beta=1.0, k=2.
    Expected marginals (exact enumeration): [0.335, 0.437, 0.556, 0.672]."""
    strengths = np.array([0.2, 0.5, 0.8, 1.1], dtype=np.float32)
    m = plackett_luce_marginals(strengths, k=2, beta=1.0)
    np.testing.assert_allclose(
        m, [0.335, 0.437, 0.556, 0.672], atol=1e-3
    )


def test_sample_extractor_count_mean():
    """Over many draws, the sample mean should match the target."""
    rng = np.random.default_rng(42)
    draws = [
        sample_extractor_count(
            n_teams=4,
            mean_extractors=1.8,
            p_zero=0.01,
            p_all=0.03,
            rng=rng,
        )
        for _ in range(20_000)
    ]
    arr = np.array(draws)
    assert abs(arr.mean() - 1.8) < 0.03


def test_sample_extractor_count_tail_probs():
    """P(k=0) hits p_zero exactly; P(k=n_teams) is at least p_all (the
    shifted-binomial component also puts some mass at k=n_teams)."""
    rng = np.random.default_rng(42)
    draws = np.array([
        sample_extractor_count(
            n_teams=4,
            mean_extractors=1.8,
            p_zero=0.01,
            p_all=0.03,
            rng=rng,
        )
        for _ in range(50_000)
    ])
    p0 = float((draws == 0).mean())
    pn = float((draws == 4).mean())
    # p_zero is a hard floor — no other component contributes to k=0.
    assert abs(p0 - 0.01) < 0.005
    # p_all is a lower bound; binomial tail adds a bit more.
    assert pn >= 0.03
    assert pn < 0.06


def test_sample_extractor_count_support_in_range():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        k = sample_extractor_count(
            n_teams=4, mean_extractors=1.8, p_zero=0.01, p_all=0.03, rng=rng,
        )
        assert 0 <= k <= 4
