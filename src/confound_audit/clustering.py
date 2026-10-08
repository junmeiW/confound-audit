"""
Protein-clustered inference.

Multiple assays frequently derive from the same protein (UniProt entry), so
assays are **not** independent replicates. Treating them as independent is
the single easiest way to obtain a wildly overconfident p-value: in the
accompanying analysis a naive Spearman p of 1.9e-9 corresponded to a
protein-clustered p of 0.87 at the same data size.

Two primitives are provided:

* :func:`per_protein`  -- collapse each cluster to one observation (mean).
* :func:`cluster_bootstrap_ci` -- resample **clusters**, not rows.

Both take an explicit ``groups`` array so they work with any clustering
(UniProt, Pfam family, organism, ...).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats

__all__ = ["uniprot_of", "per_protein", "cluster_bootstrap_ci",
           "cluster_paired_test", "cluster_one_sample_test"]


def uniprot_of(assay_id: str) -> str:
    """Extract the UniProt accessions prefix from a ProteinGym-style ID.

    ProteinGym assay IDs look like ``A0A140D2T1_ZIKV_Sourisseau_2019``; the
    first underscore-separated token is the UniProt entry. Adapt this for
    other ID conventions.
    """
    parts = str(assay_id).split("_")
    return parts[0] if len(parts) > 1 else str(assay_id)


def per_protein(values, groups) -> Tuple[np.ndarray, np.ndarray]:
    """Average ``values`` within each group; drop non-finite entries."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    uniq = np.unique(groups)
    out = np.array([values[groups == g].mean() for g in uniq]) if len(uniq) \
        else np.array([])
    return out, uniq


def cluster_bootstrap_ci(values, groups, n_boot: int = 10000, seed: int = 42,
                         alpha: float = 0.05) -> Dict[str, float]:
    """Percentile CI of the mean, resampling whole clusters with replacement.

    Every member of a drawn cluster is included, so within-cluster
    correlation is preserved.
    """
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return dict(mean=float(values.mean()) if len(values) else float("nan"),
                    lo=float("nan"), hi=float("nan"),
                    n_unit=int(len(values)), n_cluster=int(len(uniq)))
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    k = len(uniq)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=k, replace=True)
        means[i] = values[np.concatenate([idx_by_g[g] for g in pick])].mean()
    return dict(mean=float(values.mean()),
                lo=float(np.quantile(means, alpha / 2)),
                hi=float(np.quantile(means, 1 - alpha / 2)),
                n_unit=int(len(values)), n_cluster=int(k))


def cluster_paired_test(values, groups) -> Dict[str, float]:
    """Two-sided paired test for a **difference** series, clustered.

    ``values`` are per-unit differences (e.g. E_pretrained - E_random for each
    assay). Clusters are collapsed to their mean difference first, giving one
    observation per cluster, then tested against zero.
    """
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    pv, uniq = per_protein(values, groups)
    if len(pv) < 3:
        return dict(effect=float(values.mean()) if len(values) else float("nan"),
                    p_two=float("nan"), n_unit=int(len(values)),
                    n_cluster=int(len(uniq)))
    t, p_two = stats.ttest_1samp(pv, 0.0)
    ci = cluster_bootstrap_ci(values, groups)
    return dict(effect=float(values.mean()), effect_per_cluster=float(pv.mean()),
                p_two=float(p_two), t=float(t),
                ci_lo=ci["lo"], ci_hi=ci["hi"],
                n_unit=int(len(values)), n_cluster=int(len(uniq)),
                frac_positive=float(np.mean(pv > 0)))


def cluster_one_sample_test(values, groups) -> Dict[str, float]:
    """Cluster-collapsed one-sample test of an effect against zero.

    Reports both one- and two-sided p-values; the two-sided value is the
    reporting default.
    """
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    pv, uniq = per_protein(values, groups)
    if len(pv) < 3:
        return dict(effect=float(values.mean()) if len(values) else float("nan"),
                    p=float("nan"), p_two=float("nan"), n_unit=int(len(values)),
                    n_cluster=int(len(uniq)))
    t, p_two = stats.ttest_1samp(pv, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    ci = cluster_bootstrap_ci(values, groups)
    return dict(effect=float(values.mean()), effect_per_cluster=float(pv.mean()),
                p=float(p_one), p_two=float(p_two), t=float(t),
                ci_lo=ci["lo"], ci_hi=ci["hi"],
                n_unit=int(len(values)), n_cluster=int(len(uniq)))
