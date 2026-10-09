"""
Zero-parameter baselines.

These construct displacement vectors from **amino-acid identity alone**.
They contain no trained parameters and no positional information. If a
pretrained model's directional statistic is matched by one of them, the
statistic is attributable to input composition rather than to anything the
model learned.

All functions take equal-length arrays of wild-type and mutant residues and
return an ``(n, d)`` array of displacement vectors.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "AA_ORDER",
    "AA_INDEX",
    "N_AA",
    "onehot_delta",
    "replacement_pair",
    "random_lookup",
]

#: Canonical amino-acid order (identical to the reference analysis).
AA_ORDER: str = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}
N_AA: int = len(AA_ORDER)


def _indices(wt, mut) -> np.ndarray:
    """Map residue arrays to (wt_index, mut_index), allowing -1 for unknown."""
    wt = np.asarray(wt, dtype=str)
    mut = np.asarray(mut, dtype=str)
    if wt.shape != mut.shape:
        raise ValueError(f"wt and mut must have equal length; got {wt.shape} "
                         f"and {mut.shape}")
    iw = np.array([AA_INDEX.get(a, -1) for a in wt], dtype=int)
    im = np.array([AA_INDEX.get(a, -1) for a in mut], dtype=int)
    return np.stack([iw, im], axis=1)


def unknown_fraction(wt, mut) -> float:
    """Fraction of records whose residues are not canonical amino acids."""
    idx = _indices(wt, mut)
    return float(np.mean((idx[:, 0] < 0) | (idx[:, 1] < 0)))


def onehot_delta(wt, mut) -> np.ndarray:
    """20-dimensional signed one-hot delta, ``e(mut) - e(wt)``.

    Zero parameters. Records containing a non-canonical residue are returned
    as all-zero vectors (and can be dropped via :func:`unknown_fraction`).
    """
    idx = _indices(wt, mut)
    out = np.zeros((len(idx), N_AA), dtype=np.float32)
    for k, (iw, im) in enumerate(idx):
        if iw < 0 or im < 0:
            continue
        out[k, iw] -= 1.0
        out[k, im] += 1.0
    return out


def replacement_pair(wt, mut) -> np.ndarray:
    """400-dimensional replacement-pair one-hot.

    One dimension per ordered ``(wt, mut)`` pair. This is the strictest
    zero-parameter baseline: because it conditions on the full substitution
    type, exact substitution-type matching drives its statistic
    *constructively* to zero. That is a property of this encoding, not
    evidence about any model, and must not be extrapolated.
    """
    idx = _indices(wt, mut)
    out = np.zeros((len(idx), N_AA * N_AA), dtype=np.float32)
    for k, (iw, im) in enumerate(idx):
        if iw < 0 or im < 0:
            continue
        out[k, iw * N_AA + im] = 1.0
    return out


def random_lookup(wt, mut, dim: int, seed: int = 0,
                  table: np.ndarray | None = None) -> np.ndarray:
    """Random amino-acid lookup table, ``table[mut] - table[wt]``.

    Isomorphic to a neural embedding table, but untrained. ``table`` may be
    supplied to reuse a specific draw; otherwise it is sampled from
    ``N(0, 1)`` with the given ``seed``.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive; got {dim}")
    if table is None:
        rng = np.random.default_rng(seed)
        table = rng.normal(0.0, 1.0, size=(N_AA, dim))
    table = np.asarray(table, dtype=np.float32)
    if table.shape != (N_AA, dim):
        raise ValueError(f"table must have shape {(N_AA, dim)}; got "
                         f"{table.shape}")
    idx = _indices(wt, mut)
    out = np.zeros((len(idx), dim), dtype=np.float32)
    for k, (iw, im) in enumerate(idx):
        if iw < 0 or im < 0:
            continue
        out[k] = table[im] - table[iw]
    return out
