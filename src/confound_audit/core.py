"""
Core statistic and directional geometry.

This module is deliberately free of any dependency on a particular dataset,
model, or file format. Everything operates on three arrays:

    assay : (n,)   group label, e.g. assay / dataset identifier
    cls   : (n,)   fitness class, one of {"highly", "moderately", "unfit"}
    vecs  : (n, d) displacement vectors, i.e. h(mutant) - h(wild-type)

The same code therefore applies to protein language models, one-hot
encodings, random lookup tables, or any other vector representation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "CLASSES",
    "normed",
    "mean_direction",
    "assay_directions",
    "effect_from_vectors",
    "per_assay_effects",
    "paired_test",
    "one_sample_test",
    "bootstrap_ci",
]

#: The three fitness classes, in the canonical order used throughout.
CLASSES: Tuple[str, str, str] = ("highly", "moderately", "unfit")


def normed(v: np.ndarray) -> np.ndarray:
    """Unit-normalise ``v``; return ``v`` unchanged if its norm is ~0.

    The tolerance is ``1e-8``, matching the reference implementation, so that
    zero vectors (e.g. a self-substitution) do not produce NaNs.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def mean_direction(sub: np.ndarray) -> np.ndarray:
    """Mean of the unit-normalised rows of ``sub``, re-normalised.

    This is the ``d_c = normalize(mean_i normalize(delta_i))`` of the paper:
    each mutation contributes a direction, not a magnitude.
    """
    sub = np.atleast_2d(np.asarray(sub, dtype=float))
    u = np.stack([normed(x) for x in sub])
    return normed(u.mean(0))


def assay_directions(cls: np.ndarray, vecs: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean direction per fitness class within a *single* assay's data.

    Returns a dict keyed by class name. Classes with fewer than 2 mutations
    are omitted. Callers should check for all three keys; the reference
    implementation requires all three to compute an effect.
    """
    vecs = np.asarray(vecs, dtype=float)
    cls = np.asarray(cls)
    dirs: Dict[str, np.ndarray] = {}
    for k in CLASSES:
        sub = vecs[cls == k]
        if len(sub) < 2:
            continue
        dirs[k] = mean_direction(sub)
    return dirs


def _effect_one_assay(assay: np.ndarray, cls: np.ndarray, vecs: np.ndarray
                      ) -> Optional[float]:
    """Effect for one assay, or ``None`` if a class is under-populated."""
    dirs = assay_directions(cls, vecs)
    if len(dirs) < 3:
        return None
    return float(dirs["moderately"] @ dirs["unfit"]
                 - dirs["highly"] @ dirs["unfit"])


def effect_from_vectors(assay: np.ndarray, cls: np.ndarray, vecs: np.ndarray
                        ) -> Tuple[float, int]:
    """Effect statistic averaged over assays.

    Computes, per assay::

        effect_a = cos(d_moderately, d_unfit) - cos(d_highly, d_unfit)

    and returns ``(mean over assays, number of usable assays)``.

    Directions are never averaged across assays (in high dimensions the
    expected cosine between independent directions is 0, so cross-assay
    averaging cancels the signal). Only the scalar effects are averaged.
    """
    assay = np.asarray(assay)
    pairs: List[float] = []
    for aid in np.unique(assay):
        sel = assay == aid
        e = _effect_one_assay(assay[sel], cls[sel], vecs[sel])
        if e is not None:
            pairs.append(e)
    if not pairs:
        return float("nan"), 0
    return float(np.mean(pairs)), len(pairs)


def per_assay_effects(assay: np.ndarray, cls: np.ndarray, vecs: np.ndarray,
                      mag: Optional[np.ndarray] = None,
                      lo_mag: bool = False) -> List[Dict]:
    """Per-assay detail: the two cosines and their difference.

    ``mag``/``lo_mag`` optionally restrict to the lower third of a magnitude
    covariate (used to probe magnitude confounding); when ``len(mag) < 12``
    the restriction is skipped, matching the reference implementation.
    """
    assay = np.asarray(assay)
    rows: List[Dict] = []
    for aid in np.unique(assay):
        sel = assay == aid
        c_, v_ = np.asarray(cls)[sel], np.asarray(vecs)[sel]
        if lo_mag and mag is not None:
            m_ = np.asarray(mag)[sel]
            if len(m_) >= 12:
                keep = m_ <= np.quantile(m_, 0.33)
                c_, v_ = c_[keep], v_[keep]
        dirs = assay_directions(c_, v_)
        if len(dirs) < 3:
            continue
        hi = float(dirs["highly"] @ dirs["unfit"])
        md = float(dirs["moderately"] @ dirs["unfit"])
        rows.append(dict(assay_ID=aid, sim_highly_unfit=hi,
                         sim_moderately_unfit=md, effect=md - hi))
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired t-test of ``a`` vs ``b`` across paired units (e.g. assays).

    Returns **both** the one-sided (``p``) and two-sided (``p_two``) p-value,
    plus Cohen's dz. Reporting the two-sided value is the default in the
    accompanying paper; ``p`` is retained for backward comparability and is
    explicitly labelled.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return dict(effect=float("nan"), p=float("nan"), p_two=float("nan"),
                    dz=float("nan"), n=int(len(a)))
    diff = a - b
    t, p_two = stats.ttest_rel(a, b)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 1e-12 else float("nan")
    return dict(effect=float(diff.mean()), p=float(p_one),
                p_two=float(p_two), dz=dz, n=int(len(a)))


def one_sample_test(x: np.ndarray) -> Dict[str, float]:
    """One-sample t-test of ``x`` against zero, one- and two-sided."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return dict(effect=float("nan"), p=float("nan"), p_two=float("nan"),
                    t=float("nan"), n=int(len(x)))
    t, p_two = stats.ttest_1samp(x, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return dict(effect=float(x.mean()), p=float(p_one), p_two=float(p_two),
                t=float(t), n=int(len(x)))


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of the mean, resampling **units**.

    ``units`` here are whatever the caller passes: assay-level values give an
    assay-level CI. For protein-clustered inference use
    :func:`confound_audit.clustering.cluster_bootstrap_ci` instead, which
    resamples clusters rather than rows.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(x.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def collinearity_v(a: Sequence, b: Sequence) -> Dict[str, float]:
    """Cramér's V between two categorical variables.

    Used to quantify how imbalanced substitution composition is with respect
    to fitness class; a value well above 0 means the classes contain
    different substitution types, which is the confound this package targets.
    """
    import pandas as pd
    tab = pd.crosstab(pd.Series(a), pd.Series(b))
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return dict(chi2=float("nan"), p=float("nan"), v=float("nan"))
    chi2, p, _, _ = stats.chi2_contingency(tab)
    n = tab.values.sum()
    v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))
    return dict(chi2=float(chi2), p=float(p), v=v)
