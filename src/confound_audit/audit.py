"""
The audit: the three checks recommended by the accompanying paper.

Given displacement vectors and their metadata, :func:`audit` reports

1. a **zero-parameter baseline** — does one-hot / random lookup reproduce
   the statistic? (``baseline``)
2. **stratified permutation nulls** — how much of the statistic survives
   within-position label permutation? (``permutation``)
3. **strict substitution-type matching** — what remains when the classes are
   made compositionally identical? (``matching``)

Each is a different quantity and they are reported separately. In
particular the package never calls any of them "the residual" without
qualification, and never reports the retained permutation fraction as a
variance decomposition.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from . import baselines as _bl
from .core import effect_from_vectors, one_sample_test
from .clustering import cluster_one_sample_test
from .matching import matched_effect
from .permutation import stratified_null_report

__all__ = ["audit"]


def _bl_groups_default(assay_id) -> str:
    """Default clustering: one cluster per assay (i.e. no clustering)."""
    return str(assay_id)


def _as_mapping(groups, assay) -> Dict:
    """Accept either a per-mutation ``groups`` array or an ``{assay: cluster}``
    mapping and normalise to the latter."""
    if isinstance(groups, dict):
        return {str(k): str(v) for k, v in groups.items()}
    groups = np.asarray(groups)
    assay = np.asarray(assay)
    if groups.shape == assay.shape:
        return {str(a): str(g) for a, g in zip(assay, groups)}
    raise ValueError(
        "groups must be either a per-mutation array with the same length as "
        "assay, or a dict mapping assay id -> cluster id"
    )


def audit(assay, cls, vecs, *, wt=None, mut=None, pos=None, groups=None,
          n_perm: int = 200, seed: int = 0, n_boot: int = 5000,
          strata: Sequence[str] = ("assay", "position", "wt_aa"),
          run_matching: bool = True, run_baselines: bool = True) -> Dict:
    """Run the confound audit.

    Parameters
    ----------
    assay, cls, vecs : array-like
        Group label, fitness class (``"highly"``/``"moderately"``/``"unfit"``),
        and ``(n, d)`` displacement vectors (mutant minus wild-type).
    wt, mut : array-like, optional
        Residues. Required for matching and for the zero-parameter baselines.
    pos : array-like, optional
        Mutated position, required for the ``"position"`` null.
    groups : array-like, optional
        Cluster labels for protein-clustered inference; defaults to ``assay``.
    n_perm, seed, n_boot : int
        Permutation count, RNG seed, bootstrap replicates.
    strata : sequence of str
        Which permutation strata to run.
    run_matching, run_baselines : bool
        Set ``False`` to skip a check (e.g. when ``wt``/``mut`` are absent).

    Returns
    -------
    dict
        Nested report; all keys are JSON-serialisable except the per-stratum
        null arrays, which are numpy arrays.
    """
    assay = np.asarray(assay)
    cls = np.asarray(cls)
    vecs = np.asarray(vecs, dtype=float)
    if not (len(assay) == len(cls) == len(vecs)):
        raise ValueError("assay, cls and vecs must have equal length")

    observed, n_assay = effect_from_vectors(assay, cls, vecs)
    report: Dict = {
        "n_mutations": int(len(vecs)),
        "n_assays": int(n_assay),
        "observed_effect": float(observed),
        "dims": int(vecs.shape[1]) if vecs.ndim == 2 else None,
    }

    # --- assay-level and clustered inference on the observed statistic ------
    per_assay = []
    for aid in np.unique(assay):
        sel = assay == aid
        e, _ = effect_from_vectors(assay[sel], cls[sel], vecs[sel])
        if np.isfinite(e):
            per_assay.append((aid, e))
    if per_assay:
        ids = np.array([a for a, _ in per_assay])
        vals = np.array([v for _, v in per_assay])
        report["assay_level"] = one_sample_test(vals)
        # Cluster labels: either derived per assay, or supplied as a mapping.
        if groups is None:
            g = np.array([_bl_groups_default(a) for a in ids])
        else:
            mapping = _as_mapping(groups, assay)
            g = np.array([mapping.get(a, str(a)) for a in ids])
        report["clustered"] = cluster_one_sample_test(vals, g)

    # --- check 1: zero-parameter baselines ----------------------------------
    if run_baselines and wt is not None and mut is not None:
        unknown = _bl.unknown_fraction(wt, mut)
        bl: Dict = {"unknown_residue_fraction": unknown,
                    "note": ("zero-parameter and untrained; if these match the "
                             "observed statistic the effect is attributable to "
                             "input composition")}
        oh = _bl.onehot_delta(wt, mut)
        e_oh, n_oh = effect_from_vectors(assay, cls, oh)
        bl["onehot20"] = {"effect": float(e_oh), "n_assay": int(n_oh)}
        pr = _bl.replacement_pair(wt, mut)
        e_pr, n_pr = effect_from_vectors(assay, cls, pr)
        bl["pair400"] = {"effect": float(e_pr), "n_assay": int(n_pr)}
        rl = _bl.random_lookup(wt, mut, dim=20, seed=seed)
        e_rl, _ = effect_from_vectors(assay, cls, rl)
        bl["random_lookup_d20_seed0"] = {"effect": float(e_rl)}
        if np.isfinite(e_oh) and abs(observed) > 1e-12:
            bl["onehot_share_of_observed"] = float(e_oh / observed)
        report["baseline"] = bl
    elif run_baselines:
        report["baseline"] = {"skipped": "wt/mut not supplied"}

    # --- check 2: stratified permutation nulls ------------------------------
    report["permutation"] = stratified_null_report(
        assay, cls, vecs, observed=observed, n_perm=n_perm, seed=seed,
        pos=pos, wt=wt, strata=strata)
    for s, d in report["permutation"]["strata"].items():
        d.pop("null", None)  # keep the report small; arrays are rerunnable

    # --- check 3: strict substitution-type matching -------------------------
    if run_matching and wt is not None and mut is not None:
        m: Dict = {}
        for how in ("none", "wt", "pair"):
            try:
                res = matched_effect(assay, cls, vecs, wt, mut, how=how,
                                     seed=seed, expect_degenerate=(how == "pair"))
            except ValueError as exc:
                res = {"error": str(exc)}
            res.pop("per_assay", None)
            res.pop("dropped_assays", None)
            m[how] = res
        if np.isfinite(m.get("none", {}).get("effect", np.nan)) and \
                abs(m["none"]["effect"]) > 1e-12:
            for how in ("wt", "pair"):
                v = m.get(how, {}).get("effect")
                if v is not None and np.isfinite(v):
                    m[how]["relative_to_unmatched"] = float(
                        v / m["none"]["effect"])
        report["matching"] = m
    elif run_matching:
        report["matching"] = {"skipped": "wt/mut not supplied"}

    report["interpretation"] = _interpret(report)
    return report


def _interpret(report: Dict) -> Dict[str, str]:
    """Turn the numbers into the package's standing caveats."""
    out: Dict[str, str] = {}
    obs = report.get("observed_effect", float("nan"))
    perm = report.get("permutation", {}).get("strata", {})
    pos = perm.get("position", {}).get("retained")

    if pos is not None and np.isfinite(pos) and pos >= 1.0:
        out["permutation"] = (
            f"Within-position label permutation produces a null as large as "
            f"or larger than the observed statistic ({pos:.0%} of it). The "
            f"class labels therefore add nothing beyond the site-level "
            f"composition: the statistic is a property of the shared "
            f"per-site component. Note this is a conditional permutation "
            f"result, NOT a variance decomposition, and a ratio above 100% is "
            f"meaningful rather than an error."
        )
    elif pos is not None and np.isfinite(pos) and pos > 0.5:
        out["permutation"] = (
            f"Within-position label permutation retains {pos:.0%} of the "
            f"observed statistic. This is a conditional permutation result, "
            f"NOT a variance decomposition. It indicates the statistic is "
            f"largely a property of the component shared by all mutations at "
            f"a site, so a significant result against the weak (within-assay) "
            f"null is not evidence of a model-internal mechanism."
        )
    elif pos is not None and np.isfinite(pos):
        out["permutation"] = (
            f"Within-position permutation retains {pos:.0%} of the observed "
            f"statistic, so a substantial class-specific component remains."
        )

    bl = report.get("baseline", {})
    share = bl.get("onehot_share_of_observed")
    if share is not None and np.isfinite(share):
        out["baseline"] = (
            f"A 20-dimensional zero-parameter one-hot encoding yields "
            f"{share:.0%} of the observed statistic. Untrained encodings "
            f"reproducing the effect indicates input composition as the "
            f"dominant driver."
        )

    m = report.get("matching", {})
    if isinstance(m, dict) and m.get("pair", {}).get("effect") == 0.0:
        out["matching"] = (
            "The matched effect is exactly zero and was flagged "
            "constructive: for a composition-only encoding, exact "
            "(wt, mut) matching nullifies the statistic by construction. "
            "This must not be extrapolated to a trained model."
        )
    clust = report.get("clustered", {})
    if clust.get("n_cluster") and clust.get("n_unit"):
        if clust["n_cluster"] < clust["n_unit"]:
            out["clustering"] = (
                f"{clust['n_unit']} units fall into {clust['n_cluster']} "
                f"clusters. Inference is reported clustered, since units "
                f"within a cluster are not independent replicates."
            )
    return out
