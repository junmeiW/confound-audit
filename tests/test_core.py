"""Tests for the core statistic, with hand-computable cases."""

from __future__ import annotations

import numpy as np

from confound_audit.core import (CLASSES, bootstrap_ci, effect_from_vectors,
                                 mean_direction, normed, one_sample_test,
                                 paired_test, per_assay_effects)


def test_normed_unit_length():
    v = np.array([3.0, 4.0])
    assert np.allclose(np.linalg.norm(normed(v)), 1.0)
    assert np.allclose(normed(v), [0.6, 0.8])


def test_normed_zero_vector_is_safe():
    """A zero displacement must not produce NaN."""
    out = normed(np.zeros(5))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_mean_direction_is_normalised_and_magnitude_blind():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = a * 1000.0  # same directions, wildly different magnitudes
    assert np.allclose(mean_direction(a), mean_direction(b))
    assert np.isclose(np.linalg.norm(mean_direction(a)), 1.0)


def test_effect_hand_computed():
    """A minimal two-class-collinear construction with a known answer.

    highly and moderately point the same way; unfit points elsewhere.
    cos(mod, unfit) == cos(high, unfit), so the effect must be exactly 0.
    """
    vecs = np.array([
        [1.0, 0.0], [1.0, 0.0],      # highly
        [1.0, 0.0], [1.0, 0.0],      # moderately (identical direction)
        [0.0, 1.0], [0.0, 1.0],      # unfit
    ])
    cls = np.array(["highly"] * 2 + ["moderately"] * 2 + ["unfit"] * 2)
    assay = np.array(["a"] * 6)
    e, n = effect_from_vectors(assay, cls, vecs)
    assert n == 1
    assert abs(e) < 1e-12


def test_effect_sign_is_positive_when_moderately_aligns_with_unfit():
    """moderately closer to unfit than highly -> positive effect."""
    vecs = np.array([
        [1.0, 0.0], [1.0, 0.0],      # highly, orthogonal to unfit
        [0.0, 1.0], [0.0, 1.0],      # moderately, parallel to unfit
        [0.0, 1.0], [0.0, 1.0],      # unfit
    ])
    cls = np.array(["highly"] * 2 + ["moderately"] * 2 + ["unfit"] * 2)
    assay = np.array(["a"] * 6)
    e, _ = effect_from_vectors(assay, cls, vecs)
    assert e > 0.99  # cos(m,u)=1, cos(h,u)=0 -> +1


def test_effect_is_averaged_over_assays_not_before():
    """Directions must never be averaged across assays."""
    vecs = np.array([
        # assay a: positive effect (+1)
        [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0],
        [0.0, 1.0], [0.0, 1.0],
        # assay b: negative effect (-1)
        [0.0, 1.0], [0.0, 1.0],
        [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0],
    ])
    cls = np.array(["highly"] * 2 + ["moderately"] * 2 + ["unfit"] * 2 +
                   ["highly"] * 2 + ["moderately"] * 2 + ["unfit"] * 2)
    assay = np.array(["a"] * 6 + ["b"] * 6)
    e, n = effect_from_vectors(assay, cls, vecs)
    assert n == 2
    assert abs(e) < 1e-12  # (+1 + -1)/2


def test_per_assay_effects_returns_two_cosines():
    a = np.array(["x"] * 6)
    c = np.array(["highly"] * 2 + ["moderately"] * 2 + ["unfit"] * 2)
    v = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]])
    rows = per_assay_effects(a, c, v)
    assert len(rows) == 1
    r = rows[0]
    assert np.isclose(r["effect"],
                      r["sim_moderately_unfit"] - r["sim_highly_unfit"])


def test_per_assay_effects_skips_underpopulated_class():
    a = np.array(["x"] * 5)
    c = np.array(["highly"] + ["moderately"] * 2 + ["unfit"] * 2)
    v = np.random.default_rng(0).normal(0, 1, (5, 4))
    assert per_assay_effects(a, c, v) == []  # highly has only 1 member


def test_paired_test_returns_both_sided_values():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.5, 50)
    b = rng.normal(0.0, 0.5, 50)
    r = paired_test(a, b)
    assert r["n"] == 50
    assert r["p_two"] > r["p"]          # two-sided is the larger of the two
    assert np.isclose(r["p"], r["p_two"] / 2, rtol=1e-9)
    assert r["p_two"] <= 1.0 and r["p"] >= 0.0


def test_paired_test_handles_small_n():
    r = paired_test(np.array([1.0]), np.array([0.0]))
    assert np.isnan(r["p_two"])


def test_paired_test_ignores_non_finite_pairs():
    a = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    b = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    r = paired_test(a, b)
    assert r["n"] == 4  # the NaN pair is dropped, not counted


def test_one_sample_test_matches_scipy_two_sided():
    from scipy import stats
    x = np.random.default_rng(3).normal(0.4, 1.0, 40)
    r = one_sample_test(x)
    assert np.isclose(r["p_two"], stats.ttest_1samp(x, 0)[1])


def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(7).normal(2.0, 1.0, 200)
    m, lo, hi = bootstrap_ci(x, n_boot=2000, seed=1)
    assert lo < m < hi
    assert abs(m - 2.0) < 0.3


def test_bootstrap_ci_is_reproducible():
    x = np.random.default_rng(7).normal(2.0, 1.0, 100)
    assert bootstrap_ci(x, n_boot=500, seed=42) == bootstrap_ci(
        x, n_boot=500, seed=42)


def test_bootstrap_ci_small_n_is_nan():
    m, lo, hi = bootstrap_ci(np.array([1.0, 2.0]))
    assert np.isnan(lo) and np.isnan(hi)


def test_classes_constant_is_canonical_order():
    assert CLASSES == ("highly", "moderately", "unfit")
