"""
Command-line interface::

    confound-audit run  data.npz --out report.md --format md
    confound-audit demo --out demo.md
    confound-audit selftest-regression --esm-project ../esm-project

The ``demo`` subcommand needs no input data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .audit import audit
from .io import from_arrays, load_npz
from .report import to_markdown, to_text


def _demo_data(seed: int = 0, n_assay: int = 10, n_pos: int = 20,
               n_per_pos: int = 12):
    """Synthetic data reproducing the confound the package targets.

    Each assay has a set of buried positions whose fitness composition is
    **site-dependent**: some positions are intolerant (mostly unfit, few
    highly-fit variants), others are tolerant. Because every mutation at a
    position shares the same ``-h(wild-type)`` term, class-mean directions
    differ largely because the classes sample different positions and
    substitutions — not because the mutations differ functionally.

    This mirrors the real structure: the observed statistic is large, a
    zero-parameter one-hot encoding reproduces it entirely, and within-position
    label permutation retains most of it, so the weak within-assay null is
    near zero and would be badly misleading on its own.
    """
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rng = np.random.default_rng(seed)
    assay, cls, wt, mut, pos = [], [], [], [], []
    for a in range(n_assay):
        for p in range(n_pos):
            w = aa[rng.integers(0, len(aa))]
            others = [x for x in aa if x != w]
            mode = rng.choice(["intolerant", "tolerant", "mixed"],
                              p=[0.4, 0.3, 0.3])
            probs = {"intolerant": [0.05, 0.15, 0.80],
                     "tolerant": [0.75, 0.20, 0.05],
                     "mixed": [1 / 3, 1 / 3, 1 / 3]}[mode]
            kinds = rng.choice(["highly", "moderately", "unfit"],
                               size=n_per_pos, p=probs)
            for k in kinds:
                assay.append(f"ds{a}")
                cls.append(k)
                wt.append(w)
                mut.append(others[rng.integers(0, len(others))])
                pos.append(p)

    from .baselines import onehot_delta
    vecs = onehot_delta(wt, mut).astype(float)
    return (np.array(assay), np.array(cls), vecs, np.array(wt),
            np.array(mut), np.array(pos))


def _cmd_run(a) -> int:
    bundle = load_npz(a.input)
    report = audit(**bundle, n_perm=a.n_perm, seed=a.seed,
                   strata=tuple(a.strata.split(",")))
    out = to_markdown(report) if a.format == "md" else to_text(report)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(out)
    if a.json:
        clean = _jsonable(report)
        Path(a.json).write_text(json.dumps(clean, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


def _jsonable(obj):
    """Recursively convert numpy types and drop raw arrays."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return None
    return obj


def _cmd_demo(a) -> int:
    assay, cls, vecs, wt, mut, pos = _demo_data(seed=a.seed)
    bundle = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    print(f"synthetic demo: {bundle}", file=sys.stderr)
    report = audit(**bundle, n_perm=a.n_perm, seed=a.seed)
    out = to_markdown(report) if a.format == "md" else to_text(report)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(out)
    return 0


def _cmd_selftest(a) -> int:
    """Regression check against the published reference numbers.

    Reproduces the one-hot effect from the accompanying analysis if its
    result files are present. Verifies that the refactor preserved the
    statistic rather than silently changing it.
    """
    root = Path(a.esm_project)
    f = root / "results" / "figdata" / "fig_assay_effects.csv"
    if not f.exists():
        print(f"SKIP: {f} not found; cannot run the regression check")
        return 0
    import pandas as pd
    df = pd.read_csv(f)
    got = float(df["onehot20"].dropna().mean())
    expected = 0.21701795171166574
    ok = abs(got - expected) < 1e-9
    print(f"one-hot mean effect from stored per-assay values : {got:+.10f}")
    print(f"published reference value                        : {expected:+.10f}")
    print(f"match: {'YES' if ok else 'NO'}")
    print("\nNote: this validates the *reporting chain* (per-assay values -> mean).")
    print("It does not re-derive the values, which requires the model forwards.")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="confound-audit",
        description="Detect input-composition confounds in mutation-induced "
                    "embedding analyses.")
    ap.add_argument("--version", action="version",
                    version=f"confound-audit {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--n_perm", type=int, default=200)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--format", choices=("txt", "md"), default="txt")
        p.add_argument("--out", default=None, help="write report to this path")

    r = sub.add_parser("run", help="audit a .npz of vectors + metadata")
    r.add_argument("input")
    r.add_argument("--strata", default="assay,position,wt_aa")
    r.add_argument("--json", default=None, help="also write JSON (arrays dropped)")
    common(r)
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser("demo", help="run on synthetic data (no input needed)")
    common(d)
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("selftest-regression",
                       help="check the reporting chain against published values")
    s.add_argument("--esm-project", default="../esm-project")
    s.set_defaults(func=_cmd_selftest)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
