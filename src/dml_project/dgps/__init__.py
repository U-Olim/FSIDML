"""Public DGP modules used across the simulation pipeline.

Importing from ``dml_project.dgps`` exposes the canonical scenario generators
used by tasks and tests.
"""

from dml_project.dgps import (
    dense_linear_independent,
    sparse_linear_correlated,
    sparse_linear_independent,
    weak_signal_sparse,
)

__all__ = [
    "dense_linear_independent",
    "sparse_linear_independent",
    "sparse_linear_correlated",
    "weak_signal_sparse",
]
