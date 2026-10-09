# confound-audit

[![CI](https://github.com/junmeiW/confound-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/junmeiW/confound-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/junmeiW/confound-audit/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23253176.svg)](https://doi.org/10.5281/zenodo.23253176)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)

Detect **input-composition confounds** in mutation-induced embedding analyses
before interpreting them as model findings.

## The problem this solves

A common way to ask whether a protein language model (PLM) encodes something
functional is to compute, within each assay, the mean displacement direction
of each fitness class,

```
d_c = normalize( mean_{i in c} normalize( h(mutant_i) - h(wild-type_i) ) )
```

and compare them with a cosine similarity:

```
effect = cos(d_moderately, d_unfit) - cos(d_highly, d_unfit)
```

This statistic is strongly influenced by **which substitutions each class
happens to contain**. Two mutations at the same position share the same
`-h(wild-type)` term, so class-mean directions can differ substantially
even when the classes differ in no functional sense. A 20-dimensional,
**zero-parameter** one-hot encoding — no transformer, no training — can
reproduce the effect.

The result is that a large, highly significant, cross-assay-consistent
statistic can be read as evidence of a model-internal mechanism when it is
largely a property of the input.

## What this package does

Given displacement vectors and their metadata, `confound-audit` runs the
three checks that distinguish these cases:

| Check | Question | Function |
|---|---|---|
| **1. Zero-parameter baseline** | Do untrained encodings reproduce the statistic? | `onehot_delta`, `replacement_pair`, `random_lookup` |
| **2. Stratified permutation** | How much survives within-position label permutation? | `permutation_null`, `stratified_null_report` |
| **3. Substitution-type matching** | What remains once the classes are compositionally identical? | `matched_effect` |

Plus **protein-clustered inference** (`cluster_bootstrap_ci`,
`cluster_paired_test`), because multiple assays frequently come from the same
protein and are not independent replicates.

## Install

```bash
# from GitHub (available now)
pip install git+https://github.com/junmeiW/confound-audit.git

# from a local checkout
pip install -e .

# from PyPI (once the name is registered)
pip install confound-audit
```

Requires Python ≥ 3.8, `numpy`, `pandas`, `scipy`. No GPU, no model weights.

### Developing

```bash
pip install -e ".[dev]"
python -m pytest            # 71 tests; works from a checkout without installing
```

CI runs the test suite on Python 3.8–3.12, against the declared *minimum*
dependency versions, rebuilds the wheel and installs it into a clean
environment to assert the console entry point works, and lints with ruff.

## Quick start

### Command line, no data needed

```bash
confound-audit demo --format md
```

Runs on synthetic data in which fitness class is assigned independently of
the substitution, but the wild-type distribution differs by class — the exact
confound the package detects.

### Python

```python
import numpy as np
from confound_audit import audit, from_arrays, to_text

# assay: (n,) group label   cls: (n,) "highly"/"moderately"/"unfit"
# vecs:  (n, d) h(mutant) - h(wild-type)
bundle = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
report = audit(**bundle)
print(to_text(report))
```

### From a `.npz`

```bash
confound-audit run my_vectors.npz --format md --out report.md --json report.json
```

Recognised keys (aliases in parentheses):

| key | meaning |
|---|---|
| `assay` (`assay_ID`) | group / dataset label |
| `cls` (`class`, `label`) | fitness class |
| `vecs` (`vectors`) | `(n, d)` displacement vectors |
| `wt` (`wt_aa`) | wild-type residue |
| `mut` (`mut_aa`) | mutant residue |
| `pos` (`mut_pos`) | mutated position |
| `groups` (`uniprot`) | cluster label for clustered inference |

Only `assay`, `cls` and `vecs` are required. Supply `wt`/`mut` to enable
checks 1 and 3, and `pos` to enable the within-position null of check 2.

## Example output

```
==========================================================================
 CONFOUND AUDIT
==========================================================================
 mutations          : 2,400
 assays/datasets    : 10
 embedding dims     : 20
 observed effect    : +0.2289
 assay-level        : p2 = 0.0363 (n = 10)
 clustered          : p2 = 0.0363 (10 units / 10 clusters)

-- check 1: zero-parameter baselines -------------------------------------
 one-hot (20-d)     : +0.2289
 replacement-pair   : +0.0108
 random lookup (20) : +0.1464
 one-hot / observed : 100%

-- check 2: stratified permutation nulls ----------------------------------
 stratum      null mean   retained         q95
 assay          -0.0495       -22%     +0.0338
 position       +0.2911       127%     +0.3469
 wt_aa          +0.2908       127%     +0.3481

-- check 3: strict substitution-type matching ------------------------------
 none   : effect = +0.1701
 wt     : effect = +0.0188   (11% of unmatched)
 pair   : effect = +0.0000   (0% of unmatched)   [CONSTRUCTIVE ZERO]

-- interpretation ---------------------------------------------------------
 * Within-position label permutation produces a null as large as or larger than the observed statistic (127% of it). The class labels therefore add nothing beyond the site-level composition: the statistic is a property of the shared per-site component. Note this is a conditional permutation result, NOT a variance decomposition, and a ratio above 100% is meaningful rather than an error.
 * A 20-dimensional zero-parameter one-hot encoding yields 100% of the observed statistic. Untrained encodings reproducing the effect indicates input composition as the dominant driver.
 * The matched effect is exactly zero and was flagged constructive: for a composition-only encoding, exact (wt, mut) matching nullifies the statistic by construction. This must not be extrapolated to a trained model.
==========================================================================
```

## Design commitments

These are deliberate, and they are why the package exists:

1. **A permutation null is not a variance decomposition.** The retained
   fraction is reported as "retains X% of the observed statistic", never as
   "X% of the effect originates from" something.
2. **A constructive zero is not a finding.** Exact `(wt, mut)` matching drives
   a composition-only encoding to *exactly* zero by construction. The code
   raises an error rather than let you report that as evidence — pass
   `expect_degenerate=True` to confirm you understand it.
3. **Both p-values are always available and labelled.** One- and two-sided
   values are both returned; the two-sided value is the reporting default.
4. **Clustering is explicit.** `groups` is a required argument rather than a
   silent assumption, and clustered inference is reported alongside
   assay-level inference.
5. **An association is not a prediction.** The reporting layer does not use
   "predict" for cross-sectional correlations.

## What this package is *not*

- It is **not** a model. It does not embed sequences and does not need to; it
  consumes displacement vectors you already have.
- It is **not** a substitute for a compositional/geometric correction toolkit
  such as [EmmaEmb](https://github.com/broadinstitute/EmmaEmb), which
  addresses anisotropy and hubness. Those are *geometric* corrections; this
  package addresses a *compositional* confound, which re-centring does not
  remove (in the accompanying analysis mean-centring *increased* the effect).
  The two are complementary.
- It does **not** tell you whether a residual is biologically meaningful. It
  tells you how much of a statistic a zero-parameter baseline can reproduce,
  and what survives matching.

## Citation

If you use this, please cite the accompanying paper (currently in preparation)
alongside:

```bibtex
@software{confound_audit,
  author  = {Wang, Junmei},
  title   = {confound-audit: detect input-composition confounds in
             mutation-induced embedding analyses},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.23253177},
  url     = {https://doi.org/10.5281/zenodo.23253176},
}
```

The DOI above is the **concept DOI**: it always resolves to the most recent
version, which is what you normally want to cite. To pin the exact version you
used, cite the version DOI instead — for v0.1.0 that is
`10.5281/zenodo.23253177`.

## License

MIT. See `LICENSE`.
