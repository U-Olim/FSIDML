"""Public DGP modules used across the simulation pipeline.

Importing from ``dml_project.dgps`` exposes the canonical scenario generators
used by tasks and tests.
"""

from dml_project.dgps import (
    linear_baseline,
    linear_sparse_correlated,
)

__all__ = ["linear_baseline", "linear_sparse_correlated"]
