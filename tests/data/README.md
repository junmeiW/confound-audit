# Test fixtures

`fig_assay_effects.csv` — per-assay effect values from the reference analysis
(215 assays; columns: `assay_ID`, `onehot20`, `pair400`, `esm`, `onehot_same`,
`uniprot`).

It is bundled (23 KB) so that the regression tests in
`tests/test_regression_reference.py` run for anyone who clones this
repository, rather than silently skipping. The values are **the reference
study's own computed outputs**, not raw ProteinGym measurements; the raw data
are obtained separately from the ProteinGym distribution.

The regression tests assert that the published summary statistics are
reproducible from these per-assay values, and that this package's core
statistic, run on vectors reconstructed to realise them, returns the same
mean (+0.2170).
