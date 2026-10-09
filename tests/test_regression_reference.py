"""
Regression tests against the published reference analysis.

These guard the property that matters most for a refactor: **the numbers must
not move**. They are skipped automatically when the reference result files are
absent, so the package's own test suite stays self-contained.

Two levels are covered:

* ``test_*_from_stored_figures`` -- re-derive published summary statistics from
  stored per-assay values. Pure arithmetic; no model needed.
* ``test_effect_from_vectors_matches_reference`` -- run this package's core
  statistic on vectors **reconstructed from** the stored per-assay cosines and
  check that the mean agrees. This exercises the refactored code path, not just
  the CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from confound_audit.core import effect_from_vectors

#: value reported in the manuscript (Results 3.1, two-sided p = 6.90e-33)
PUBLISHED_ONEHOT20 = 0.21701795171166574
#: value reported for the 400-dim replacement-pair baseline
PUBLISHED_PAIR400 = 0.08651339065196902
#: paired-sample one-hot and ESM-2 final-layer effects (Results 3.4)
PUBLISHED_ONEHOT_PAIRED = 0.12879901167958283
PUBLISHED_ESM_LLAST = 0.14058830124694247
#: clustered recomputation of the one-hot effect (Results 3.1)
PUBLISHED_ONEHOT_CLUSTERED_P = 4.498216458790131e-29


def _figdata() -> Path:
    """Locate the reference per-assay values.

    Preferred: the fixture bundled in ``tests/data`` (so the checks run for
    anyone who clones the repository). Fallback: the sibling analysis project,
    when the package is developed inside it.
    """
    bundled = Path(__file__).resolve().parent / "data" / "fig_assay_effects.csv"
    if bundled.exists():
        return bundled
    return (Path(__file__).resolve().parents[2] / "results" / "figdata"
            / "fig_assay_effects.csv")


def _esm_project() -> Path:
    return Path(__file__).resolve().parents[2]


pytestmark = pytest.mark.skipif(
    not _figdata().exists(),
    reason="reference per-assay values not found (tests/data or esm-project)",
)


def test_onehot_mean_matches_published_value():
    import pandas as pd
    df = pd.read_csv(_figdata())
    got = float(df["onehot20"].dropna().mean())
    assert abs(got - PUBLISHED_ONEHOT20) < 1e-9


def test_pair400_mean_matches_published_value():
    import pandas as pd
    df = pd.read_csv(_figdata())
    got = float(df["pair400"].dropna().mean())
    assert abs(got - PUBLISHED_PAIR400) < 1e-9


def test_paired_sample_effects_match_published_values():
    import pandas as pd
    df = pd.read_csv(_figdata())
    assert abs(float(df["onehot_same"].dropna().mean())
               - PUBLISHED_ONEHOT_PAIRED) < 1e-9
    assert abs(float(df["esm"].dropna().mean())
               - PUBLISHED_ESM_LLAST) < 1e-9


def test_paired_difference_is_not_significant():
    """The headline null result: on identical mutations the pretrained model
    does not differ from the zero-parameter baseline."""
    import pandas as pd

    from confound_audit.core import paired_test
    df = pd.read_csv(_figdata()).dropna(subset=["onehot_same", "esm"])
    r = paired_test(df["esm"].values, df["onehot_same"].values)
    assert abs(r["effect"] - 0.011789289567359659) < 1e-9
    # stored diff_p was one-sided; two-sided is what we report
    assert r["p_two"] > 0.2


def test_clustered_p_matches_reference_json():
    p = _esm_project() / "results" / "17_clustered_stats.json"
    if not p.exists():
        pytest.skip("clustered reference json absent")
    with open(p) as fh:
        d = json.load(fh)
    got = d["effects"]["onehot20"]["protein_p_two"]
    assert abs(got - PUBLISHED_ONEHOT_CLUSTERED_P) < 1e-30


def test_effect_from_vectors_reproduces_stored_mean():
    """Exercise the refactored core on vectors built to realise the stored
    per-assay cosines, and confirm the mean agrees.

    Two mutations per class per assay, arranged so that
    cos(mod, unfit) - cos(high, unfit) equals the stored per-assay effect.
    """
    import pandas as pd
    df = pd.read_csv(_figdata()).dropna(subset=["onehot20"])

    assay, cls, rows = [], [], []
    for i, r in enumerate(df.itertuples()):
        e = float(r.onehot20)
        if not np.isfinite(e):
            continue
        aid = f"a{i}"
        # Realise a target effect e = cos(m,u) - cos(h,u) with u = (1, 0).
        # The range of e is [-2, 2]; pin the larger cosine to 1 and let the
        # other carry the difference:
        #   e >= 0 : cos_m = 1,      cos_h = 1 - e
        #   e <  0 : cos_h = 1,      cos_m = 1 + e
        if e >= 0:
            c_m, c_h = 1.0, 1.0 - e
        else:
            c_m, c_h = 1.0 + e, 1.0
        c_m = float(np.clip(c_m, -1.0, 1.0))
        c_h = float(np.clip(c_h, -1.0, 1.0))
        t_m, t_h = float(np.arccos(c_m)), float(np.arccos(c_h))
        v_h = [np.cos(t_h), np.sin(t_h)]
        v_m = [np.cos(t_m), np.sin(t_m)]
        vecs = [v_h, v_h, v_m, v_m, [1.0, 0.0], [1.0, 0.0]]
        for v, c in zip(vecs, ["highly"] * 2 + ["moderately"] * 2 +
                        ["unfit"] * 2):
            assay.append(aid)
            cls.append(c)
            rows.append(v)

    got, n = effect_from_vectors(np.array(assay), np.array(cls),
                                 np.array(rows, dtype=float))
    assert n == len(df)
    assert abs(got - PUBLISHED_ONEHOT20) < 1e-6
