"""
Stratified permutation nulls.

Permuting class labels *within* a stratum holds the stratifying variable
fixed while destroying any class-specific signal. Two strata matter here:

* ``"assay"``    : within assay only. This is the **weak** null and is the
  one that typically gets reported. It is near zero even for a purely
  compositional statistic, so a significant result against it is not
  evidence of a model-internal mechanism.
* ``"position"`` : within assay × mutated position. Class sizes are
  preserved inside each stratum. This is the **diagnostic** null: if it
  retains most of the observed statistic, the statistic is largely a
  property of the component shared by all mutations at a site.
* ``"wt_aa"``    : within assay × wild-type residue identity.

The package reports the *retained fraction* ``null / observed`` rather than
calling it a variance decomposition. A conditional permutation null is not a
decomposition, and the wording matters.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .core import effect_from_vectors

__all__ = ["permutation_null", "stratified_null_report"]

_STRATA = ("assay", "position", "wt_aa")


def _keys(assay, strat: str, pos=None, wt=None) -> np.ndarray:
    assay = np.asarray(assay, dtype=str)
    if strat == "assay":
        return assay
    if strat == "position":
        if pos is None:
            raise ValueError("strat='position' requires the pos argument")
        pos = np.asarray(pos).astype(str)
        return np.char.add(np.char.add(assay, "|"), pos)
    if strat == "wt_aa":
        if wt is None:
            raise ValueError("strat='wt_aa' requires the wt argument")
        return np.char.add(np.char.add(assay, "|"), np.asarray(wt, dtype=str))
    raise ValueError(f"unknown stratum {strat!r}; expected one of {_STRATA}")


def permutation_null(assay, cls, vecs, n_perm: int = 200, strat: str = "position",
                     seed: int = 0, pos=None, wt=None) -> np.ndarray:
    """Permute class labels within ``strat`` and recompute the statistic.

    Class sizes are preserved within each stratum (labels are permuted, not
    resampled), so the null answers: *what does this statistic look like when
    class membership carries no information beyond the stratum?*

    Returns an array of ``n_perm`` null values.
    """
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1; got {n_perm}")
    assay = np.asarray(assay)
    cls = np.asarray(cls)
    vecs = np.asarray(vecs, dtype=float)
    keys = _keys(assay, strat, pos=pos, wt=wt)
    rng = np.random.default_rng(seed)

    uniq = np.unique(keys)
    buckets = [np.where(keys == k)[0] for k in uniq]
    buckets = [b for b in buckets if len(b) >= 2]

    null = np.empty(n_perm, dtype=float)
    for pi in range(n_perm):
        newc = cls.copy()
        for b in buckets:
            newc[b] = rng.permutation(cls[b])
        null[pi] = effect_from_vectors(assay, newc, vecs)[0]
    return null


def stratified_null_report(assay, cls, vecs, observed: Optional[float] = None,
                           n_perm: int = 200, seed: int = 0,
                           pos=None, wt=None,
                           strata: Sequence[str] = _STRATA) -> Dict:
    """Run every available stratum and report the retained fraction.

    ``retained = null_mean / observed`` is reported as a *retained fraction of
    the observed statistic*, not as "explained variance".
    """
    if observed is None:
        observed, _ = effect_from_vectors(assay, cls, vecs)
    out: Dict = {"observed": float(observed), "n_perm": int(n_perm),
                 "strata": {}}
    for s in strata:
        try:
            nl = permutation_null(assay, cls, vecs, n_perm=n_perm, strat=s,
                                  seed=seed, pos=pos, wt=wt)
        except ValueError:
            continue
        m = float(np.mean(nl))
        out["strata"][s] = dict(
            null_mean=m, null_sd=float(np.std(nl, ddof=1)),
            null_q95=float(np.quantile(nl, 0.95)),
            retained=(m / observed) if abs(observed) > 1e-12 else float("nan"),
            null=nl,
        )
    return out
