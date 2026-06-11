"""Helpers for selecting workflow mode, replication counts, and file suffixes."""

from __future__ import annotations

import os

from dml_project import config

VALID_RUN_MODES = {"smoke", "full"}
ALLOWED_MODES_MESSAGE = "Allowed modes are smoke and full."


def get_run_mode() -> str:
    """Read and validate the run mode from ``DML_RUN_MODE``.

    Returns:
        Normalized mode string.

    Raises:
        ValueError: If the environment variable is set to an unsupported value.
    """

    mode = os.getenv("DML_RUN_MODE", "full").lower()
    if mode not in VALID_RUN_MODES:
        raise ValueError(f"Unsupported DML_RUN_MODE={mode!r}. {ALLOWED_MODES_MESSAGE}")
    return mode


def get_replication_count(mode: str) -> int:
    """Return run-mode-specific replication count.

    Args:
        mode: Run mode string (case-insensitive).

    Returns:
        Replication count configured for the selected mode.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """

    normalized_mode = mode.lower()
    if normalized_mode == "smoke":
        return config.SMOKE_N_REPLICATIONS
    if normalized_mode == "full":
        return config.N_REPLICATIONS
    raise ValueError(f"Unknown mode: {mode}. {ALLOWED_MODES_MESSAGE}")


def output_suffix(mode: str) -> str:
    """Return output filename suffix for the selected mode.

    Args:
        mode: Run mode string (case-insensitive).

    Returns:
        ``""`` for ``full`` mode and ``"_smoke"`` for smoke mode.

    Raises:
        ValueError: If ``mode`` is not recognized.
    """

    normalized_mode = mode.lower()
    if normalized_mode == "full":
        return ""
    if normalized_mode == "smoke":
        return "_smoke"
    raise ValueError(f"Unknown mode: {mode}. {ALLOWED_MODES_MESSAGE}")
