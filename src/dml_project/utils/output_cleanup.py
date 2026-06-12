"""Safe archiving helpers for generated simulation outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dml_project import config

SMOKE_OUTPUT_PATTERNS = [
    "raw/simulations_smoke.csv",
    "aggregated/*smoke*",
    "tables/*smoke*",
    "figures/*smoke*",
    "checks/runtime_profile/*smoke*",
    "checks/runtime_profile/*.csv",
]


def _unique_archive_dir(archive_root: Path, name: str) -> Path:
    """Return a non-existing archive directory path."""

    candidate = archive_root / name
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        numbered = archive_root / f"{name}_{index:03d}"
        if not numbered.exists():
            return numbered
        index += 1


def _smoke_output_files(output_dir: Path) -> list[Path]:
    """Return existing smoke output files matched by cleanup patterns."""

    files: dict[Path, None] = {}
    for pattern in SMOKE_OUTPUT_PATTERNS:
        for path in output_dir.glob(pattern):
            if path.is_file():
                files[path.resolve()] = None
    return sorted(files, key=lambda path: str(path))


def archive_smoke_outputs(
    output_dir: str | Path = config.OUTPUT_DIR,
    archive_root: str | Path | None = None,
    timestamp: str | None = None,
    delete: bool = False,
) -> list[Path]:
    """Archive or delete existing smoke outputs before a new smoke run.

    Args:
        output_dir: Root ``documents/outputs`` directory to scan.
        archive_root: Root archive directory. Defaults to ``output_dir/archive``.
        timestamp: Optional timestamp label for deterministic tests.
        delete: If ``True``, delete matched files instead of archiving. Defaults
            to ``False`` so prior outputs are preserved.

    Returns:
        Paths moved into the archive, or deleted source paths when
        ``delete=True``.
    """

    output_root = Path(output_dir).resolve()
    archive_base = (
        Path(archive_root).resolve()
        if archive_root is not None
        else output_root / "archive"
    )
    archive_base.mkdir(parents=True, exist_ok=True)

    files = _smoke_output_files(output_root)
    if delete:
        deleted: list[Path] = []
        for source in files:
            source.unlink()
            deleted.append(source)
        return deleted

    label = timestamp if timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = _unique_archive_dir(archive_base, f"smoke_{label}")
    moved: list[Path] = []
    for source in files:
        relative_path = source.relative_to(output_root)
        destination = archive_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved.append(destination)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return moved
