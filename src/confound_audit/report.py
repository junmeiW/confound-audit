"""
Report rendering.

Turns an :func:`confound_audit.audit.audit` result into human-readable text
or Markdown. The wording deliberately preserves the distinctions the package
exists to protect: a permutation null is not a variance decomposition, a
constructive zero is not a finding, and an association is not a prediction.
"""

from __future__ import annotations

__all__ = ["to_text", "to_markdown"]


def _fmt(x, nd=4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    return f"{v:+.{nd}f}"


def _fmt_p(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:.3g}"


def to_text(report: dict) -> str:
    """Plain-text report."""
    L = []
    add = L.append
    add("=" * 74)
    add(" CONFOUND AUDIT")
    add("=" * 74)
    add(f" mutations          : {report.get('n_mutations', 'n/a'):,}"
        if isinstance(report.get("n_mutations"), int)
        else f" mutations          : {report.get('n_mutations', 'n/a')}")
    add(f" assays/datasets    : {report.get('n_assays', 'n/a')}")
    add(f" embedding dims     : {report.get('dims', 'n/a')}")
    add(f" observed effect    : {_fmt(report.get('observed_effect'))}")

    al = report.get("assay_level")
    if al:
        add(f" assay-level        : p2 = {_fmt_p(al.get('p_two'))} "
            f"(n = {al.get('n')})")
    cl = report.get("clustered")
    if cl and cl.get("n_cluster"):
        add(f" clustered          : p2 = {_fmt_p(cl.get('p_two'))} "
            f"({cl['n_unit']} units / {cl['n_cluster']} clusters)")

    bl = report.get("baseline")
    if bl and "skipped" not in bl:
        add("")
        add("-- check 1: zero-parameter baselines " + "-" * 37)
        add(f" one-hot (20-d)     : {_fmt(bl['onehot20']['effect'])}")
        add(f" replacement-pair   : {_fmt(bl['pair400']['effect'])}")
        add(f" random lookup (20) : {_fmt(bl['random_lookup_d20_seed0']['effect'])}")
        if "onehot_share_of_observed" in bl:
            add(f" one-hot / observed : "
                f"{bl['onehot_share_of_observed']:.0%}")

    pm = report.get("permutation", {})
    if pm.get("strata"):
        add("")
        add("-- check 2: stratified permutation nulls " + "-" * 34)
        add(f" {'stratum':<10}{'null mean':>12}{'retained':>11}{'q95':>12}")
        for s, d in pm["strata"].items():
            ret = d.get("retained")
            add(f" {s:<10}{_fmt(d['null_mean']):>12}"
                f"{(f'{ret:.0%}' if ret == ret else 'n/a'):>11}"
                f"{_fmt(d['null_q95']):>12}")

    m = report.get("matching")
    if m and "skipped" not in m:
        add("")
        add("-- check 3: strict substitution-type matching " + "-" * 30)
        for how in ("none", "wt", "pair"):
            d = m.get(how, {})
            if "error" in d:
                add(f" {how:<6} : (guard) {d['error'][:60]}...")
                continue
            rel = d.get("relative_to_unmatched")
            add(f" {how:<6} : effect = {_fmt(d.get('effect'))}"
                + (f"   ({rel:.0%} of unmatched)" if rel is not None else "")
                + ("   [CONSTRUCTIVE ZERO]" if d.get("degenerate") else ""))

    iv = report.get("interpretation", {})
    if iv:
        add("")
        add("-- interpretation " + "-" * 57)
        for k in ("permutation", "baseline", "matching", "clustering"):
            if k in iv:
                add(f" * {iv[k]}")
    add("=" * 74)
    return "\n".join(L)


def to_markdown(report: dict) -> str:
    """Markdown report (suitable for supplementary material)."""
    L = []
    add = L.append
    add("# Confound audit report")
    add("")
    add(f"- mutations: **{report.get('n_mutations', 'n/a')}**")
    add(f"- assays/datasets: **{report.get('n_assays', 'n/a')}**")
    add(f"- embedding dims: **{report.get('dims', 'n/a')}**")
    add(f"- observed effect: **{_fmt(report.get('observed_effect'))}**")
    al = report.get("assay_level") or {}
    if al:
        add(f"- assay-level two-sided p: **{_fmt_p(al.get('p_two'))}** "
            f"(n = {al.get('n')})")
    cl = report.get("clustered") or {}
    if cl.get("n_cluster"):
        add(f"- protein-clustered two-sided p: **{_fmt_p(cl.get('p_two'))}** "
            f"({cl['n_unit']} units / {cl['n_cluster']} clusters)")

    bl = report.get("baseline")
    if bl and "skipped" not in bl:
        add("")
        add("## Check 1 — zero-parameter baselines")
        add("")
        add("| encoding | effect |")
        add("|---|---|")
        add(f"| one-hot (20-d) | {_fmt(bl['onehot20']['effect'])} |")
        add(f"| replacement-pair (400-d) | {_fmt(bl['pair400']['effect'])} |")
        add(f"| random lookup (d=20, seed 0) | "
            f"{_fmt(bl['random_lookup_d20_seed0']['effect'])} |")
        if "onehot_share_of_observed" in bl:
            add("")
            add(f"One-hot reproduces **{bl['onehot_share_of_observed']:.0%}** "
                f"of the observed statistic.")

    pm = report.get("permutation", {})
    if pm.get("strata"):
        add("")
        add("## Check 2 — stratified permutation nulls")
        add("")
        add("| stratum | null mean | retained | null 95th pct |")
        add("|---|---|---|---|")
        for s, d in pm["strata"].items():
            ret = d.get("retained")
            add(f"| {s} | {_fmt(d['null_mean'])} | "
                f"{f'{ret:.0%}' if ret == ret else 'n/a'} | "
                f"{_fmt(d['null_q95'])} |")
        add("")
        add("> The retained fraction is a property of the conditional "
            "permutation null, **not** a variance decomposition.")

    m = report.get("matching")
    if m and "skipped" not in m:
        add("")
        add("## Check 3 — strict substitution-type matching")
        add("")
        add("| matching | effect | relative |")
        add("|---|---|---|")
        for how in ("none", "wt", "pair"):
            d = m.get(how, {})
            if "error" in d:
                continue
            rel = d.get("relative_to_unmatched")
            add(f"| {how} | {_fmt(d.get('effect'))} | "
                f"{f'{rel:.0%}' if rel is not None else 'n/a'} |")

    iv = report.get("interpretation", {})
    if iv:
        add("")
        add("## Interpretation")
        add("")
        for k in ("permutation", "baseline", "matching", "clustering"):
            if k in iv:
                add(f"- {iv[k]}")
    return "\n".join(L) + "\n"
