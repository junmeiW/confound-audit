"""
Data loading.

The audit needs only ``(assay, cls, vecs)`` plus, optionally, ``wt``, ``mut``
and ``pos``. This module provides a small, explicit way to assemble those
from whatever you have, without imposing a directory layout.

Two entry points:

* :func:`from_arrays` -- you already have the arrays in memory.
* :func:`load_npz`    -- a ``.npz`` produced by the reference pipeline
  (``vecs`` / ``assay`` / ``cls`` / ``wt_aa`` / ``mut_aa`` / ``mut_pos``).

A ``.npz`` written by ``numpy.savez`` is the recommended interchange format;
a CSV of per-mutation metadata plus a ``.npy``/``.npz`` of vectors also works
via :func:`from_arrays`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = ["CLASSES", "from_arrays", "load_npz", "load_csv", "DataBundle"]

CLASSES = ("highly", "moderately", "unfit")


class DataBundle(dict):
    """A plain dict holding the arrays the audit consumes.

    Keys: ``assay``, ``cls``, ``vecs`` and optionally ``wt``, ``mut``,
    ``pos``, ``groups``. Exposed as a class mainly so that ``repr`` is
    informative and attribute-style access is available.
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    #: keys that :func:`confound_audit.audit` accepts as keyword arguments
    AUDIT_KEYS = ("assay", "cls", "vecs", "wt", "mut", "pos", "groups")

    def kwargs(self) -> dict:
        """Return only the keys ``audit()`` accepts (drops bookkeeping)."""
        return {k: self[k] for k in self.AUDIT_KEYS if k in self}

    def __repr__(self) -> str:
        n = len(self.get("assay", []))
        d = self["vecs"].shape[1] if "vecs" in self and np.ndim(self["vecs"]) == 2 else "?"
        keys = ", ".join(sorted(self.keys()))
        return f"DataBundle(n={n}, dims={d}, keys=[{keys}])"


def _validate(assay, cls, vecs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assay = np.asarray(assay)
    cls = np.asarray(cls)
    vecs = np.asarray(vecs, dtype=float)
    if vecs.ndim != 2:
        raise ValueError(f"vecs must be 2-D (n, d); got shape {vecs.shape}")
    if not (len(assay) == len(cls) == vecs.shape[0]):
        raise ValueError(
            f"length mismatch: assay={len(assay)}, cls={len(cls)}, "
            f"vecs={vecs.shape[0]}")
    unknown = set(np.unique(cls).tolist()) - set(CLASSES)
    if unknown:
        raise ValueError(
            f"unknown fitness classes {sorted(unknown)}; expected a subset of "
            f"{CLASSES}")
    return assay, cls, vecs


def from_arrays(assay, cls, vecs, wt=None, mut=None, pos=None,
                groups=None) -> DataBundle:
    """Assemble a validated :class:`DataBundle` from in-memory arrays."""
    assay, cls, vecs = _validate(assay, cls, vecs)
    b = DataBundle(assay=assay, cls=cls, vecs=vecs)
    for k, v in (("wt", wt), ("mut", mut), ("pos", pos), ("groups", groups)):
        if v is not None:
            b[k] = np.asarray(v)
    return b


def _first_present(z, *names) -> Optional[np.ndarray]:
    for n in names:
        if n in z:
            return z[n]
    return None


def load_npz(path) -> DataBundle:
    """Load a ``.npz`` of vectors + metadata.

    Recognised keys (aliases in parentheses):

    ============  =====================================================
    ``assay``     group label (``assay_ID``)
    ``cls``       fitness class (``class``, ``label``)
    ``vecs``      displacement vectors, shape ``(n, d)`` (``vectors``)
    ``wt``        wild-type residue (``wt_aa``)
    ``mut``       mutant residue (``mut_aa``)
    ``pos``       mutated position (``mut_pos``, ``position``)
    ``groups``    cluster labels (``uniprot``, ``cluster``)
    ============  =====================================================
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with np.load(path, allow_pickle=True) as z:
        assay = _first_present(z, "assay", "assay_ID")
        cls = _first_present(z, "cls", "class", "label")
        vecs = _first_present(z, "vecs", "vectors")
        missing = [n for n, v in (("assay", assay), ("cls", cls),
                                  ("vecs", vecs)) if v is None]
        if missing:
            raise KeyError(
                f"{path.name} is missing required key(s): {missing}. "
                f"Found: {sorted(z.files)}")
        return from_arrays(
            assay, cls, vecs,
            wt=_first_present(z, "wt", "wt_aa"),
            mut=_first_present(z, "mut", "mut_aa"),
            pos=_first_present(z, "pos", "mut_pos", "position"),
            groups=_first_present(z, "groups", "uniprot", "cluster"),
        )


def load_csv(path, vecs_path=None, vecs_cols=None, **kwargs) -> DataBundle:
    """Load per-mutation metadata from CSV, optionally with vectors elsewhere.

    Parameters
    ----------
    path : path
        CSV with one row per mutation. Required columns are matched case-
        insensitively against common aliases (see :func:`load_npz`).
    vecs_path : path, optional
        ``.npy``/``.npz`` holding the ``(n, d)`` vectors, in the same row
        order as the CSV. If omitted, ``vecs_cols`` selects vector columns
        from the CSV itself.
    vecs_cols : sequence of str, optional
        Names of the columns holding vector components, in order.
    """
    import pandas as pd

    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    def col(*names, required=False):
        for n in names:
            if n.lower() in lower:
                return df[lower[n.lower()]].values
        if required:
            raise KeyError(f"{Path(path).name}: none of {names} found; "
                           f"columns are {list(df.columns)}")
        return None

    assay = col("assay", "assay_id", "dataset", "dms_id", required=True)
    cls = col("cls", "class", "label", "fitness_class", required=True)
    wt = col("wt", "wt_aa", "wild_type")
    mut = col("mut", "mut_aa", "mutant_aa")
    pos = col("pos", "mut_pos", "position")
    groups = col("groups", "uniprot", "cluster")

    if vecs_path is not None:
        vp = Path(vecs_path)
        if vp.suffix == ".npz":
            with np.load(vp, allow_pickle=True) as z:
                vecs = _first_present(z, "vecs", "vectors")
        else:
            vecs = np.load(vp)
    elif vecs_cols:
        vecs = df[list(vecs_cols)].values
    else:
        raise ValueError("provide either vecs_path or vecs_cols")
    return from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos,
                       groups=groups, **kwargs)
