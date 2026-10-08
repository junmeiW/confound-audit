"""Tests for the audit orchestration, IO, reporting and CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from confound_audit import (audit, from_arrays, load_csv, load_npz, to_markdown,
                            to_text)
from confound_audit.cli import main as cli_main


def test_audit_returns_all_three_checks(simple_data):
    assay, cls, vecs, wt, mut, pos = simple_data
    b = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    rep = audit(**b.kwargs(), n_perm=10, seed=1)
    assert "baseline" in rep
    assert "permutation" in rep
    assert "matching" in rep
    assert np.isfinite(rep["observed_effect"])


def test_audit_skips_checks_without_residues(simple_data):
    assay, cls, vecs, _, _, pos = simple_data
    b = from_arrays(assay, cls, vecs, pos=pos)
    rep = audit(**b.kwargs(), n_perm=10, seed=1)
    assert "skipped" in rep["baseline"]
    assert "skipped" in rep["matching"]
    assert "position" in rep["permutation"]["strata"]


def test_audit_detects_a_planted_confound(compositional_data):
    """On data whose classes contain different substitutions, the one-hot
    baseline must reproduce a substantial share of the effect."""
    assay, cls, vecs, wt, mut, pos = compositional_data
    b = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    rep = audit(**b.kwargs(), n_perm=20, seed=1)
    share = rep["baseline"]["onehot_share_of_observed"]
    # vecs ARE the one-hot encoding here, so the share is ~100%
    assert 0.9 < share < 1.1


def test_audit_flags_high_position_retention(compositional_data):
    assay, cls, vecs, wt, mut, pos = compositional_data
    b = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    rep = audit(**b.kwargs(), n_perm=30, seed=1)
    ret = rep["permutation"]["strata"]["position"]["retained"]
    assert ret > 0.5
    # wording must not overclaim
    assert "NOT a variance decomposition" in rep["interpretation"]["permutation"]


def test_audit_length_mismatch_raises():
    with pytest.raises(ValueError):
        from_arrays(["a", "b"], ["highly"], np.zeros((2, 3)))


def test_from_arrays_rejects_unknown_class():
    with pytest.raises(ValueError, match="unknown fitness classes"):
        from_arrays(["a", "a"], ["good", "bad"], np.zeros((2, 3)))


def test_from_arrays_rejects_1d_vectors():
    with pytest.raises(ValueError, match="2-D"):
        from_arrays(["a", "a"], ["highly", "unfit"], np.zeros(2))


def test_bundle_kwargs_excludes_bookkeeping(simple_data):
    assay, cls, vecs, wt, mut, pos = simple_data
    b = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    assert set(b.kwargs()) <= {"assay", "cls", "vecs", "wt", "mut", "pos",
                               "groups"}


def test_npz_roundtrip(tmp_path, simple_data):
    assay, cls, vecs, wt, mut, pos = simple_data
    f = tmp_path / "d.npz"
    np.savez(f, assay=assay, cls=cls, vecs=vecs, wt_aa=wt, mut_aa=mut,
             mut_pos=pos, uniprot=np.array([f"p{i}" for i in range(len(assay))]))
    b = load_npz(f)
    assert np.array_equal(b["assay"], assay)
    assert "wt" in b and "pos" in b and "groups" in b
    assert b["vecs"].shape == vecs.shape


def test_npz_alias_keys_accepted(tmp_path, simple_data):
    assay, cls, vecs, _, _, _ = simple_data
    f = tmp_path / "d.npz"
    np.savez(f, assay_ID=assay, label=cls, vectors=vecs)
    b = load_npz(f)
    assert np.array_equal(b["cls"], cls)


def test_npz_missing_key_names_the_missing_key(tmp_path, simple_data):
    assay, cls, vecs, _, _, _ = simple_data
    f = tmp_path / "d.npz"
    np.savez(f, assay=assay, vecs=vecs)
    with pytest.raises(KeyError, match="cls"):
        load_npz(f)


def test_npz_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_npz(tmp_path / "nope.npz")


def test_load_csv_with_sidecar_vectors(tmp_path, simple_data):
    import pandas as pd
    assay, cls, vecs, wt, mut, pos = simple_data
    df = pd.DataFrame(dict(assay_id=assay, class_=cls, wt_aa=wt, mut_aa=mut,
                           mut_pos=pos))
    df = df.rename(columns={"class_": "cls"})
    csv = tmp_path / "meta.csv"
    df.to_csv(csv, index=False)
    vecs_f = tmp_path / "vecs.npy"
    np.save(vecs_f, vecs)
    b = load_csv(csv, vecs_path=vecs_f)
    assert b["vecs"].shape == vecs.shape
    assert np.array_equal(b["wt"], wt)


def test_load_csv_requires_a_vector_source(tmp_path, simple_data):
    import pandas as pd
    assay, cls, _, _, _, _ = simple_data
    csv = tmp_path / "m.csv"
    pd.DataFrame(dict(assay_ID=assay, cls=cls)).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="vecs_path or vecs_cols"):
        load_csv(csv)


def test_report_is_rendered_in_both_formats(simple_data):
    assay, cls, vecs, wt, mut, pos = simple_data
    b = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    rep = audit(**b.kwargs(), n_perm=10, seed=1)
    t, m = to_text(rep), to_markdown(rep)
    assert "CONFOUND AUDIT" in t
    assert "check 1" in t and "check 2" in t and "check 3" in t
    assert m.startswith("# Confound audit report")
    assert "Check 1" in m and "Check 3" in m


def test_report_never_calls_retention_a_decomposition(compositional_data):
    assay, cls, vecs, wt, mut, pos = compositional_data
    b = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    rep = audit(**b.kwargs(), n_perm=10, seed=1)
    blob = (to_text(rep) + to_markdown(rep)).lower()
    for banned in ("explained variance", "variance explained",
                   "% of the effect originates", "originates from"):
        assert banned not in blob


# ---------------------------------------------------------------------- CLI
def test_cli_demo_runs_and_emits_report(capsys):
    rc = cli_main(["demo", "--n_perm", "10", "--format", "md"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Confound audit report" in out


def test_cli_demo_writes_file(tmp_path):
    out = tmp_path / "r.md"
    rc = cli_main(["demo", "--n_perm", "10", "--out", str(out)])
    assert rc == 0
    assert "CONFOUND AUDIT" in out.read_text()


def test_cli_run_writes_markdown_and_json(tmp_path, simple_data):
    assay, cls, vecs, wt, mut, pos = simple_data
    f = tmp_path / "d.npz"
    np.savez(f, assay=assay, cls=cls, vecs=vecs, wt_aa=wt, mut_aa=mut,
             mut_pos=pos)
    out = tmp_path / "rep.md"
    js = tmp_path / "rep.json"
    rc = cli_main(["run", str(f), "--n_perm", "10", "--format", "md",
                   "--out", str(out), "--json", str(js)])
    assert rc == 0
    assert "Check 1" in out.read_text()
    payload = json.loads(js.read_text())
    assert "observed_effect" in payload
    assert "permutation" in payload


def test_cli_json_drops_raw_arrays(tmp_path, simple_data):
    assay, cls, vecs, wt, mut, pos = simple_data
    f = tmp_path / "d.npz"
    np.savez(f, assay=assay, cls=cls, vecs=vecs, wt_aa=wt, mut_aa=mut,
             mut_pos=pos)
    js = tmp_path / "r.json"
    cli_main(["run", str(f), "--n_perm", "10", "--json", str(js)])
    payload = json.loads(js.read_text())
    assert "null" not in payload.get("permutation", {}).get("strata", {}).get(
        "position", {})


def test_interpretation_handles_retention_above_100pct(compositional_data):
    """A null larger than the observed statistic is possible and must be
    described, not reported as a nonsensical percentage."""
    assay, cls, vecs, wt, mut, pos = compositional_data
    b = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    rep = audit(**b.kwargs(), n_perm=30, seed=1)
    txt = rep["interpretation"]["permutation"]
    assert "NOT a variance decomposition" in txt
    if rep["permutation"]["strata"]["position"]["retained"] >= 1.0:
        assert "as large as or larger" in txt


def test_demo_reproduces_the_paper_signature(capsys):
    """The built-in demo must show the diagnostic pattern: one-hot reproduces
    the statistic and within-position permutation does not remove it."""
    from confound_audit.cli import _demo_data
    b = from_arrays(*_demo_data(seed=0)[:3], wt=_demo_data(seed=0)[3],
                    mut=_demo_data(seed=0)[4], pos=_demo_data(seed=0)[5])
    rep = audit(**b.kwargs(), n_perm=40, seed=0)
    assert rep["observed_effect"] > 0.05
    assert rep["baseline"]["onehot_share_of_observed"] > 0.9
    assert rep["permutation"]["strata"]["position"]["retained"] > 0.5
