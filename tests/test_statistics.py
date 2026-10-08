"""Tests for baselines, matching, permutation and clustering."""

from __future__ import annotations

import numpy as np
import pytest

from confound_audit.baselines import (AA_ORDER, N_AA, onehot_delta,
                                      random_lookup, replacement_pair,
                                      unknown_fraction)
from confound_audit.clustering import (cluster_bootstrap_ci,
                                       cluster_one_sample_test,
                                       cluster_paired_test, per_protein,
                                       uniprot_of)
from confound_audit.core import CLASSES
from confound_audit.matching import match_indices, matched_effect
from confound_audit.permutation import permutation_null, stratified_null_report


# ---------------------------------------------------------------- baselines
def test_onehot_delta_is_mutant_minus_wildtype():
    v = onehot_delta(["A"], ["C"])
    iA, iC = AA_ORDER.index("A"), AA_ORDER.index("C")
    assert v.shape == (1, N_AA)
    assert v[0, iC] == 1.0 and v[0, iA] == -1.0
    assert np.isclose(v.sum(), 0.0)


def test_onehot_has_zero_parameters():
    """The encoding is fully determined by residue identity, nothing else."""
    a = onehot_delta(["A", "W"], ["C", "Y"])
    b = onehot_delta(["A", "W"], ["C", "Y"])
    assert np.array_equal(a, b)


def test_unknown_residues_are_zero_not_crash():
    v = onehot_delta(["A", "X"], ["C", "Y"])
    assert np.allclose(v[1], 0.0)
    assert np.isclose(unknown_fraction(["A", "X"], ["C", "Y"]), 0.5)


def test_replacement_pair_has_one_hot_per_type():
    v = replacement_pair(["A", "A"], ["C", "C"])
    assert v.shape == (2, N_AA * N_AA)
    assert np.allclose(v[0], v[1])       # same substitution -> same vector
    assert np.isclose(v[0].sum(), 1.0)


def test_random_lookup_is_seed_reproducible_and_dimension_scalable():
    a = random_lookup(["A"], ["C"], dim=8, seed=5)
    b = random_lookup(["A"], ["C"], dim=8, seed=5)
    c = random_lookup(["A"], ["C"], dim=8, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert random_lookup(["A"], ["C"], dim=64, seed=5).shape == (1, 64)


def test_random_lookup_is_difference_of_two_table_rows():
    table = np.arange(20 * 3, dtype=float).reshape(20, 3)
    v = random_lookup(["A"], ["C"], dim=3, table=table)
    expect = table[AA_ORDER.index("C")] - table[AA_ORDER.index("A")]
    assert np.allclose(v[0], expect)


def test_random_lookup_rejects_bad_dim_and_table():
    with pytest.raises(ValueError):
        random_lookup(["A"], ["C"], dim=0)
    with pytest.raises(ValueError):
        random_lookup(["A"], ["C"], dim=3, table=np.zeros((5, 3)))


def test_baseline_length_mismatch_raises():
    with pytest.raises(ValueError):
        onehot_delta(["A", "C"], ["C"])


# ----------------------------------------------------------------- matching
def test_match_indices_pair_balances_classes():
    rng = np.random.default_rng(0)
    n = 200
    assay = np.array(["a"] * n)
    cls = rng.choice(list(CLASSES), n)
    wt = rng.choice(list(AA_ORDER), n)
    mut = rng.choice(list(AA_ORDER), n)
    m = match_indices(assay, cls, wt, mut, how="pair", seed=1)
    assert m["n_input_assays"] == 1
    assert m["n_kept_mutations"] > 0


def test_match_indices_is_reproducible():
    rng = np.random.default_rng(2)
    n = 100
    args = (np.array(["a"] * n), rng.choice(list(CLASSES), n),
            rng.choice(list(AA_ORDER), n), rng.choice(list(AA_ORDER), n))
    a = match_indices(*args, how="pair", seed=7)
    b = match_indices(*args, how="pair", seed=7)
    assert a["n_kept_mutations"] == b["n_kept_mutations"]
    assert [i.tolist() for _, i in a["kept"]] == [i.tolist() for _, i in b["kept"]]


def _paired_types_data(rng, n_assay=4, n_types=8, n_per=4):
    """Build data in which each assay repeats the *same* substitution types
    across all three classes, so exact-pair matching retains samples.

    Random (wt, mut) draws almost never recur across classes, so matching
    would discard everything; a real DMS assay is enriched for shared types,
    and this fixture reproduces that structure deliberately.
    """
    types = [(AA_ORDER[i], AA_ORDER[(i + 7) % N_AA]) for i in range(n_types)]
    assay, cls, wt, mut = [], [], [], []
    for a in range(n_assay):
        for k in CLASSES:
            for (w, m) in types:
                for _ in range(n_per):
                    assay.append(f"a{a}")
                    cls.append(k)
                    wt.append(w)
                    mut.append(m)
    return np.array(assay), np.array(cls), np.array(wt), np.array(mut)


def test_matched_effect_pair_on_onehot_is_constructive_zero():
    """The central guard: one-hot + exact-pair matching must give exactly 0,
    and the caller must acknowledge that it is constructive."""
    rng = np.random.default_rng(3)
    assay, cls, wt, mut = _paired_types_data(rng)
    vecs = onehot_delta(wt, mut).astype(float)

    # without acknowledgement -> loud error, not a silent zero
    with pytest.raises(ValueError, match="constructive"):
        matched_effect(assay, cls, vecs, wt, mut, how="pair")

    # with acknowledgement -> flagged degenerate
    r = matched_effect(assay, cls, vecs, wt, mut, how="pair",
                       expect_degenerate=True)
    assert r["degenerate"] is True
    assert r["effect"] == 0.0
    assert r["n_assay"] > 0


def test_matched_effect_on_random_vectors_is_not_degenerate():
    """Random vectors carry no (wt,mut) structure, so exact-pair matching must
    NOT drive them to zero; this proves the guard is not over-triggering."""
    rng = np.random.default_rng(4)
    assay, cls, wt, mut = _paired_types_data(rng)
    vecs = rng.normal(0, 1, (len(assay), 24))
    r = matched_effect(assay, cls, vecs, wt, mut, how="pair")
    assert r["degenerate"] is False
    assert np.isfinite(r["effect"])
    assert r["n_assay"] > 0


def test_matching_none_keeps_everything_usable():
    rng = np.random.default_rng(5)
    assay, cls, wt, mut = _paired_types_data(rng)
    vecs = rng.normal(0, 1, (len(assay), 12))
    r_none = matched_effect(assay, cls, vecs, wt, mut, how="none")
    r_pair = matched_effect(assay, cls, vecs, wt, mut, how="pair")
    assert r_none["n_kept_mutations"] >= r_pair["n_kept_mutations"]


def test_match_indices_rejects_bad_how():
    with pytest.raises(ValueError):
        match_indices(["a"], ["highly"], ["A"], ["C"], how="bogus")


# -------------------------------------------------------------- permutation
def test_permutation_null_preserves_class_sizes_within_stratum():
    """Permuting labels must not change how many of each class a stratum has."""
    rng = np.random.default_rng(5)
    n = 120
    assay = np.array(["a"] * n)
    cls = rng.choice(list(CLASSES), n)
    vecs = rng.normal(0, 1, (n, 8))
    pos = np.repeat(np.arange(12), 10)
    nl = permutation_null(assay, cls, vecs, n_perm=30, strat="position",
                          seed=1, pos=pos)
    assert len(nl) == 30
    assert np.all(np.isfinite(nl))


def test_within_assay_null_is_smaller_than_within_position_null():
    """With a purely compositional vector set, the position-stratified null
    must retain far more of the statistic than the assay-stratified null."""
    rng = np.random.default_rng(6)
    n = 600
    assay = np.array([f"a{i%5}" for i in range(n)])
    cls = rng.choice(list(CLASSES), n)
    wt = rng.choice(list(AA_ORDER), n)
    mut = rng.choice(list(AA_ORDER), n)
    vecs = onehot_delta(wt, mut).astype(float)
    pos = np.repeat(np.arange(60), 10)
    rep = stratified_null_report(assay, cls, vecs, n_perm=40, seed=1, pos=pos)
    assert rep["strata"]["position"]["retained"] > \
        rep["strata"]["assay"]["retained"]


def test_permutation_requires_arguments_for_stratum():
    with pytest.raises(ValueError, match="pos"):
        permutation_null(["a"], ["highly"], np.zeros((1, 2)), strat="position")
    with pytest.raises(ValueError, match="wt"):
        permutation_null(["a"], ["highly"], np.zeros((1, 2)), strat="wt_aa")


def test_permutation_rejects_bad_stratum():
    with pytest.raises(ValueError, match="unknown stratum"):
        permutation_null(["a"], ["highly"], np.zeros((1, 2)), strat="bogus")


def test_permutation_n_perm_validated():
    with pytest.raises(ValueError):
        permutation_null(["a"], ["highly"], np.zeros((1, 2)), n_perm=0)


# --------------------------------------------------------------- clustering
def test_uniprot_of_extracts_accession():
    assert uniprot_of("A0A140D2T1_ZIKV_Sourisseau_2019") == "A0A140D2T1"
    assert uniprot_of("single") == "single"


def test_per_protein_collapses_to_one_row_per_group():
    v = np.array([1.0, 3.0, 10.0, 20.0])
    g = np.array(["p1", "p1", "p2", "p2"])
    pv, uniq = per_protein(v, g)
    assert len(pv) == 2
    assert np.allclose(sorted(pv), [2.0, 15.0])


def test_cluster_bootstrap_ignores_within_cluster_correlation():
    """A single cluster containing everything must give a near-degenerate CI,
    because resampling clusters then resamples the same unit repeatedly."""
    rng = np.random.default_rng(8)
    v = rng.normal(1.0, 1.0, 100)
    g = np.array(["only"] * 100)
    r = cluster_bootstrap_ci(v, g, n_boot=200, seed=1)
    assert r["n_cluster"] == 1
    assert np.isnan(r["lo"]) or np.isclose(r["lo"], v.mean())


def one_sample_naive(v):
    from scipy import stats
    return stats.ttest_1samp(v, 0)[1]


def test_clustering_corrects_false_positives_when_units_are_correlated():
    """The reason clustering matters: if units within a cluster are strongly
    correlated (as repeated assays of one protein are), treating them as
    independent produces a wildly overconfident p-value.

    Note the correction is *not* monotone in cluster count. With loosely
    clustered units, collapsing to cluster means can even shrink p, because
    each mean is estimated far more precisely than a single unit. The
    pathological case -- and the one this package guards against -- is
    within-cluster correlation with intact between-cluster spread.
    """
    rng = np.random.default_rng(9)
    n_clusters, per = 5, 50
    # Strongly correlated within cluster, large between-cluster offsets.
    v = np.concatenate([rng.normal(0, 0.01, per) + c for c in range(n_clusters)])
    g = np.repeat([f"q{i}" for i in range(n_clusters)], per)

    naive = one_sample_naive(v)
    clustered = cluster_one_sample_test(v, g)["p_two"]

    assert naive < 1e-10, "naive test should be wildly overconfident here"
    assert clustered > naive * 1e6, "clustering must inflate p substantially"
    assert clustered > 1e-3, "the clustered p should not be tiny"


def test_clustered_and_naive_agree_when_every_unit_is_its_own_cluster():
    """With no clustering structure the two must coincide."""
    rng = np.random.default_rng(12)
    v = rng.normal(0.3, 1.0, 30)
    g = np.array([f"c{i}" for i in range(30)])
    clustered = cluster_one_sample_test(v, g)["p_two"]
    naive = one_sample_naive(v)
    assert np.isclose(clustered, naive, rtol=1e-9)


def test_cluster_paired_test_reports_cluster_count():
    rng = np.random.default_rng(10)
    g = np.repeat([f"p{i}" for i in range(6)], 20)
    d = rng.normal(0.5, 1.0, len(g))
    r = cluster_paired_test(d, g)
    assert r["n_cluster"] == 6
    assert r["n_unit"] == len(g)
    assert 0 <= r["frac_positive"] <= 1


def test_cluster_functions_ignore_non_finite():
    v = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    r = cluster_bootstrap_ci(v, g, n_boot=100, seed=1)
    assert r["n_unit"] == 5


def test_cluster_bootstrap_is_reproducible():
    rng = np.random.default_rng(11)
    g = np.repeat([f"p{i}" for i in range(8)], 15)
    v = rng.normal(0.3, 1.0, len(g))
    assert cluster_bootstrap_ci(v, g, n_boot=500, seed=3) == \
        cluster_bootstrap_ci(v, g, n_boot=500, seed=3)
