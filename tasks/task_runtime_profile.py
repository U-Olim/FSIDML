"""Pytask step for lightweight runtime profiling reports."""

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
    from dml_project.reporting.runtime_profile import create_runtime_profile
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.reporting.runtime_profile import create_runtime_profile

PROFILE_DIR = config.OUTPUT_DIR / "checks" / "runtime_profile"


def _write_profile_tables(raw_results_path: Path, output_dir: Path) -> list[Path]:
    """Write runtime profile tables and return generated paths."""

    profile = create_runtime_profile(raw_results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for name, table in profile.items():
        output_path = output_dir / f"{name}.csv"
        table.to_csv(output_path, index=False, encoding="utf-8")
        output_paths.append(output_path)
    return output_paths


def task_runtime_profile(
    path_to_raw: Path = config.RAW_RESULTS_DIR / "simulations_smoke.csv",
    path_to_manifest: Annotated[Path, Product] = PROFILE_DIR / "manifest.txt",
) -> None:
    """Create runtime profile CSVs from existing smoke outputs."""

    expected_raw = config.RAW_RESULTS_DIR / "simulations_smoke.csv"
    if path_to_raw != expected_raw:
        raise ValueError(f"task_runtime_profile must read {expected_raw}, got {path_to_raw}")
    if not path_to_raw.exists():
        raise FileNotFoundError(f"Smoke raw output not found: {path_to_raw}")

    output_paths = _write_profile_tables(path_to_raw, path_to_manifest.parent)
    manifest_lines = [
        str(path.relative_to(PROJECT_ROOT).as_posix())
        for path in sorted(output_paths, key=lambda path: str(path))
    ]
    path_to_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
