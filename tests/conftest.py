"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def simple_data(rng):
    """Small, well-populated dataset: 4 assays x 3 classes x 10 mutations."""
    assay, cls, wt, mut, pos = [], [], [], [], []
    for a in range(4):
        p = 0
        for k in ("highly", "moderately", "unfit"):
            for _ in range(10):
                w = rng.choice(list(AA))
                m = rng.choice([x for x in AA if x != w])
                assay.append(f"ds{a}")
                cls.append(k)
                wt.append(w)
                mut.append(m)
                pos.append(p)
                p += 1
    vecs = rng.normal(0, 1, (len(assay), 16))
    return (np.array(assay), np.array(cls), vecs, np.array(wt), np.array(mut),
            np.array(pos))


@pytest.fixture
def compositional_data(rng):
    """Data with a deliberate composition confound.

    Fitness class is drawn independently of the substitution type, but the
    wild-type distribution differs sharply by class. The effect is therefore
    driven by *which* substitutions fall in each class.
    """
    assay, cls, wt, mut, pos = [], [], [], [], []
    p_hi = np.array([.3, .3, .3] + [.01] * 17)
    p_hi = p_hi / p_hi.sum()
    for a in range(6):
        p = 0
        for _ in range(90):
            c = rng.choice(["highly", "moderately", "unfit"])
            if c == "highly":
                w = rng.choice(list(AA), p=p_hi)
            else:
                w = rng.choice(list(AA))
            m = rng.choice([x for x in AA if x != w])
            assay.append(f"ds{a}")
            cls.append(c)
            wt.append(w)
            mut.append(m)
            pos.append(p)
            p += 1
    from confound_audit.baselines import onehot_delta
    vecs = onehot_delta(wt, mut).astype(float)
    return (np.array(assay), np.array(cls), vecs, np.array(wt), np.array(mut),
            np.array(pos))
