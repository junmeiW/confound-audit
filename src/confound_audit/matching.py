"""
Strict substitution-type matching.

The confound this package targets arises because fitness classes contain
*different substitutions at different positions*. Matching makes the classes
comparable by construction:

* ``"wt"``   : within assay × wild-type residue
* ``"pair"`` : within assay × exact ``(wt, mut)`` pair

Matching is performed **within assay** and subsampled to equal counts per
class, so that class-mean directions are computed from balanced data.

A note on the constructive zero
-------------------------------
For the 20-dimensional one-hot baseline, exact ``(wt, mut)`` matching drives
the statistic to **exactly zero**, because its displacement depends only on
the residue pair and therefore carries no positional information. That is a
mathematical property of that encoding. It is *not* evidence that a
pretrained model behaves the same way, and must never be extrapolated. Pass
``expect_degenerate=True`` only when you know you are matching a
composition-only encoding.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .core import CLASSES, bootstrap_ci, mean_direction

__all__ = ["match_indices", "matched_effect"]


def match_indices(assay, cls, wt, mut, how: str = "pair", seed: int = 0,
                  min_kept: int = 6) -> Dict:
    """Select balanced, substitution-matched indices within each assay.

    Parameters
    ----------
    assay, cls, wt, mut : array-like
        Group label, fitness class, wild-type residue, mutant residue.
    how : {"pair", "wt", "none"}
        Matching key. ``"none"`` keeps everything (exact-balance subsampling
        still applies).
    seed : int
        Seed for the equal-count subsampling; fixed for reproducibility.
    min_kept : int
        Assays with fewer than this many retained mutations are dropped.

    Returns
    -------
    dict
        ``kept`` : list of ``(assay_id, index_array)``
        ``dropped_assays`` : assays lost to the ``min_kept`` rule
        ``n_input_assays``, ``n_kept_mutations``, ``n_dropped_small``
    """
    if how not in ("pair", "wt", "none"):
        raise ValueError(f"how must be 'pair', 'wt' or 'none'; got {how!r}")

    rng = np.random.default_rng(seed)
    assay = np.asarray(assay)
    cls = np.asarray(cls)
    wt = np.asarray(wt, dtype=str)
    mut = np.asarray(mut, dtype=str)

    if how == "pair":
        key_all = np.char.add(np.char.add(wt.astype(str), "|"), mut.astype(str))
    elif how == "wt":
        key_all = wt.astype(str)
    else:
        key_all = np.full(len(wt), "all", dtype=object)

    kept: List = []
    n_dropped_small = 0

    for aid in np.unique(assay):
        sel = assay == aid
        c = cls[sel]
        key = key_all[sel]
        idx_local = np.where(sel)[0]

        pick: List[int] = []
        for k in np.unique(key):
            same = key == k
            ih = idx_local[np.where(same & (c == "highly"))[0]]
            im = idx_local[np.where(same & (c == "moderately"))[0]]
            iu = idx_local[np.where(same & (c == "unfit"))[0]]
            if len(ih) == 0 or len(im) == 0:
                continue
            n = min(len(ih), len(im))
            pick += list(rng.choice(ih, n, replace=False))
            pick += list(rng.choice(im, n, replace=False))
            if len(iu) > 0:
                pick += list(rng.choice(iu, min(len(iu), n), replace=False))

        if len(pick) < min_kept:
            n_dropped_small += 1
            continue
        kept.append((aid, np.array(sorted(pick), dtype=int)))

    all_assays = set(np.unique(assay).tolist())
    kept_assays = {a for a, _ in kept}
    return dict(
        kept=kept,
        dropped_assays=sorted(all_assays - kept_assays),
        n_input_assays=int(len(all_assays)),
        n_kept_mutations=int(sum(len(i) for _, i in kept)),
        n_dropped_small=int(n_dropped_small),
    )


def _effect_on_subset(assay, cls, vecs, idx) -> Optional[float]:
    a_, c_, v_ = assay[idx], cls[idx], vecs[idx]
    dirs = {}
    for k in CLASSES:
        sub = v_[c_ == k]
        if len(sub) < 2:
            return None
        dirs[k] = mean_direction(sub)
    return float(dirs["moderately"] @ dirs["unfit"]
                 - dirs["highly"] @ dirs["unfit"])


def matched_effect(assay, cls, vecs, wt, mut, how: str = "pair", seed: int = 0,
                   expect_degenerate: bool = False) -> Dict:
    """Effect statistic after substitution-type matching.

    Returns a dict with ``effect``, ``p`` (one-sided), ``p_two``,
    ``ci_lo``/``ci_hi`` (assay-level bootstrap), ``n_assay``,
    ``n_kept_mutations``, ``per_assay`` and bookkeeping fields.

    Raises
    ------
    ValueError
        If a composition-only encoding yields exactly zero under
        ``how="pair"`` but ``expect_degenerate`` was not set. This is a guard
        against silently interpreting a constructive zero as a finding.
    """
    assay = np.asarray(assay)
    cls = np.asarray(cls)
    vecs = np.asarray(vecs, dtype=float)

    m = match_indices(assay, cls, wt, mut, how=how, seed=seed)
    per_assay: List[Dict] = []
    effs: List[float] = []
    for aid, idx in m["kept"]:
        e = _effect_on_subset(assay, cls, vecs, idx)
        if e is None:
            continue
        effs.append(e)
        per_assay.append(dict(assay_ID=aid, effect=e, n_kept=int(len(idx))))

    base = dict(
        n_assay_input=m["n_input_assays"],
        n_kept_mutations=m["n_kept_mutations"],
        n_dropped_small=m["n_dropped_small"],
        dropped_assays=m["dropped_assays"],
        per_assay=per_assay,
    )

    if len(effs) < 3:
        base.update(effect=float("nan"), p=float("nan"), p_two=float("nan"),
                    n_assay=len(effs),
                    note="too few assays survive matching (<3)")
        return base

    effs_arr = np.asarray(effs, float)

    if np.allclose(effs_arr, 0.0):
        if not expect_degenerate:
            raise ValueError(
                "Matched effect is exactly zero, but expect_degenerate=False. "
                "Matching a composition-only encoding (e.g. the 20-dim "
                "one-hot) on the exact (wt, mut) pair drives the statistic "
                "constructively to zero; that is a property of the encoding, "
                "NOT evidence about any model. Re-run with "
                "expect_degenerate=True if this is intended."
            )
        base.update(effect=0.0, p=float("nan"), p_two=float("nan"),
                    ci_lo=0.0, ci_hi=0.0, degenerate=True, n_assay=len(effs),
                    note=("constructive zero: a composition-only delta depends "
                          "only on (wt, mut), so exact-pair matching nullifies "
                          "it. Applies to this encoding only."))
        return base

    from scipy import stats
    t, p_two = stats.ttest_1samp(effs_arr, 0.0)
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    _, lo, hi = bootstrap_ci(effs_arr, seed=seed)
    base.update(effect=float(effs_arr.mean()), p=float(p_one),
                p_two=float(p_two), n_assay=int(len(effs_arr)),
                ci_lo=lo, ci_hi=hi, degenerate=False)
    return base
