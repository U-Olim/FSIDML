"""Helpers for selecting pipeline mode, replication counts, and file suffixes.

The project supports two execution modes:
- ``full``: production-scale replication count.
- ``fast``: reduced replication count for quicker iteration.
"""

from __future__ import annotations

import os

from dml_project import config

VALID_RUN_MODES = {"full", "fast"}


def get_run_mode() -> str:
    """Read and validate the run mode from ``DML_RUN_MODE``.

    Returns:
        Normalized mode string (``"full"`` or ``"fast"``).

    Raises:
        ValueError: If the environment variable is set to an unsupported value.
    """

    mode = os.getenv("DML_RUN_MODE", "full").lower()
    if mode not in VALID_RUN_MODES:
        raise ValueError(
            f"Unsupported DML_RUN_MODE={mode!r}. Use one of {sorted(VALID_RUN_MODES)}."
        )
    return mode


def get_replication_count(mode: str) -> int:
    """Return run-mode-specific replication count.

    FAST and FULL must use the same scenario grid and differ only in ``n_rep``.

    Args:
        mode: Run mode string (case-insensitive).

    Returns:
        Replication count configured for the selected mode.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """

    normalized_mode = mode.lower()
    if normalized_mode == "full":
        return config.FULL_N_REP
    if normalized_mode == "fast":
        return config.FAST_N_REP
    raise ValueError(f"Unknown mode: {mode}")


def output_suffix(mode: str) -> str:
    """Return output filename suffix for the selected mode.

    Args:
        mode: Run mode string (case-insensitive).

    Returns:
        ``""`` for ``full`` mode and ``"_fast"`` for ``fast`` mode.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """

    normalized_mode = mode.lower()
    if normalized_mode == "full":
        return ""
    if normalized_mode == "fast":
        return "_fast"
    raise ValueError(f"Unknown mode: {mode}")
