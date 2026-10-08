# Changelog

## 0.1.0 — 2026-10-08

Initial release, extracted from the accompanying analysis of directional
embedding statistics in protein language models.

### Added
- `core` — the directional effect statistic
  `cos(d_moderately, d_unfit) - cos(d_highly, d_unfit)`, per-assay details,
  paired/one-sample tests (one- and two-sided), and bootstrap CIs.
- `baselines` — zero-parameter encodings: 20-dim one-hot delta, 400-dim
  replacement-pair, and random amino-acid lookup tables.
- `matching` — within-assay substitution-type matching (`pair`, `wt`, `none`)
  with equal-count subsampling, plus a **guard that raises** rather than let a
  constructive zero be reported as a finding.
- `permutation` — stratified permutation nulls (within assay / position /
  wild-type) and retained-fraction reporting.
- `clustering` — protein-clustered bootstrap and tests, for the case where
  several assays derive from one protein.
- `io` — `from_arrays`, `load_npz`, `load_csv`; no imposed directory layout.
- `report` — plain-text and Markdown reports with deliberately conservative
  wording.
- `cli` — `run`, `demo`, `selftest-regression`.
- 71 tests, including regression checks against the published reference
  values of the accompanying analysis.

### Notes
- Metadata lives in `setup.cfg` rather than `pyproject.toml` so that the
  package builds correctly under setuptools < 61 (which lacks PEP 621
  support); this was verified against setuptools 58.0.4.
