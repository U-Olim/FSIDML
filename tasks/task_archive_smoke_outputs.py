"""Pytask utility for archiving existing smoke outputs."""

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
    from dml_project.utils.output_cleanup import archive_smoke_outputs
except ModuleNotFoundError:
    from src.dml_project import config
    from src.dml_project.utils.output_cleanup import archive_smoke_outputs


def task_archive_smoke_outputs(
    path_to_manifest: Annotated[Path, Product] = (
        config.OUTPUT_ARCHIVE_DIR / "smoke_archive_manifest.txt"
    ),
) -> None:
    """Archive smoke outputs without running simulations."""

    moved_paths = archive_smoke_outputs()
    path_to_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        str(path.relative_to(config.PROJECT_ROOT).as_posix()) for path in moved_paths
    ]
    path_to_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    for path in moved_paths:
        print(path)
