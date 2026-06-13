"""Pytask step for smoke/full result diagnostic ranking reports."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from pytask import Product

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from dml_project import config
    from dml_project.pipeline_mode import get_run_mode, output_suffix
    from dml_project.reporting.result_diagnostics import create_result_diagnostics
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.pipeline_mode import get_run_mode, output_suffix
    from src.dml_project.reporting.result_diagnostics import create_result_diagnostics

MODE = get_run_mode()
SUFFIX = output_suffix(MODE)
RESULT_DIAGNOSTICS_DIR = config.OUTPUT_DIR / "checks" / "result_diagnostics"


def task_result_diagnostics(
    path_to_manifest: Annotated[Path, Product] = (
        RESULT_DIAGNOSTICS_DIR / f"manifest{SUFFIX}.txt"
    ),
) -> None:
    """Create result diagnostic CSVs from existing aggregated outputs."""

    expected_summary = config.AGGREGATED_RESULTS_DIR / f"scenario_summary{SUFFIX}.csv"
    if not expected_summary.exists():
        raise FileNotFoundError(f"Scenario summary not found: {expected_summary}")

    tables = create_result_diagnostics(expected_summary, path_to_manifest.parent)
    output_paths = [path_to_manifest.parent / f"{name}.csv" for name in tables]
    manifest_lines = [
        str(path.relative_to(PROJECT_ROOT).as_posix())
        for path in sorted(output_paths, key=lambda path: str(path))
    ]
    path_to_manifest.parent.mkdir(parents=True, exist_ok=True)
    path_to_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
