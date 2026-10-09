"""
confound-audit — detect input-composition confounds in mutation-induced
embedding analyses.

Typical use::

    import numpy as np
    from confound_audit import audit, from_arrays, to_text

    bundle = from_arrays(assay, cls, vecs, wt=wt, mut=mut, pos=pos)
    report = audit(**bundle)
    print(to_text(report))

Why this exists
---------------
A common way to ask whether a protein language model encodes something
functional is to compare mean displacement directions between fitness
categories with a cosine similarity. That statistic is strongly influenced by
which substitutions each category happens to contain — a property of the
input, not of the model. Untrained encodings can reproduce it.

This package quantifies that influence with three checks and refuses to
overstate what each one shows.
"""

from __future__ import annotations

from .audit import audit
from .baselines import AA_ORDER, N_AA, onehot_delta, random_lookup, replacement_pair
from .clustering import (
                        cluster_bootstrap_ci,
                        cluster_one_sample_test,
                        cluster_paired_test,
                        per_protein,
                        uniprot_of,
)
from .core import (
                        CLASSES,
                        bootstrap_ci,
                        effect_from_vectors,
                        normed,
                        one_sample_test,
                        paired_test,
                        per_assay_effects,
)
from .io import DataBundle, from_arrays, load_csv, load_npz
from .matching import match_indices, matched_effect
from .permutation import permutation_null, stratified_null_report
from .report import to_markdown, to_text

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # orchestrator
    "audit",
    # io
    "DataBundle", "from_arrays", "load_npz", "load_csv",
    # core
    "CLASSES", "effect_from_vectors", "per_assay_effects", "normed",
    "one_sample_test", "paired_test", "bootstrap_ci",
    # baselines
    "AA_ORDER", "N_AA", "onehot_delta", "replacement_pair", "random_lookup",
    # matching
    "matched_effect", "match_indices",
    # permutation
    "permutation_null", "stratified_null_report",
    # clustering
    "uniprot_of", "per_protein", "cluster_bootstrap_ci",
    "cluster_one_sample_test", "cluster_paired_test",
    # reporting
    "to_text", "to_markdown",
]
