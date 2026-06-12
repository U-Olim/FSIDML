"""Tests for safe smoke-output archiving."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dml_project.utils.output_cleanup import archive_smoke_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / ".test_tmp" / "cleanup"


@pytest.fixture()
def cleanup_root() -> Path:
    """Create a project-local temporary output tree."""

    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    output_dir = TEST_ROOT / "documents" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield output_dir
    finally:
        shutil.rmtree(TEST_ROOT, ignore_errors=True)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_archive_moves_smoke_raw_file(cleanup_root: Path) -> None:
    """Smoke raw output should move into a timestamped archive."""

    source = _write(cleanup_root / "raw" / "simulations_smoke.csv")

    moved = archive_smoke_outputs(output_dir=cleanup_root, timestamp="20260612_120000")

    destination = (
        cleanup_root
        / "archive"
        / "smoke_20260612_120000"
        / "raw"
        / "simulations_smoke.csv"
    )
    assert moved == [destination.resolve()]
    assert destination.exists()
    assert not source.exists()


def test_archive_skips_missing_files_without_error(cleanup_root: Path) -> None:
    """Missing smoke outputs should not raise."""

    moved = archive_smoke_outputs(output_dir=cleanup_root, timestamp="20260612_120000")

    assert moved == []
    assert (cleanup_root / "archive").exists()


def test_archive_does_not_move_non_smoke_raw_file(cleanup_root: Path) -> None:
    """Full raw output should not be archived by smoke cleanup."""

    full_output = _write(cleanup_root / "raw" / "simulations.csv", "full")

    moved = archive_smoke_outputs(output_dir=cleanup_root, timestamp="20260612_120000")

    assert moved == []
    assert full_output.exists()
    assert full_output.read_text(encoding="utf-8") == "full"


def test_archive_creates_archive_directory(cleanup_root: Path) -> None:
    """Archive root should be created if missing."""

    _write(cleanup_root / "tables" / "table_main_results_smoke.csv")

    archive_smoke_outputs(output_dir=cleanup_root, timestamp="20260612_120000")

    assert (cleanup_root / "archive").is_dir()
    assert (cleanup_root / "archive" / "smoke_20260612_120000").is_dir()


def test_repeated_archive_calls_do_not_overwrite(cleanup_root: Path) -> None:
    """Repeated archive calls with the same timestamp should use a new folder."""

    first_source = _write(cleanup_root / "raw" / "simulations_smoke.csv", "first")
    first_moved = archive_smoke_outputs(
        output_dir=cleanup_root,
        timestamp="20260612_120000",
    )
    assert not first_source.exists()

    second_source = _write(cleanup_root / "raw" / "simulations_smoke.csv", "second")
    second_moved = archive_smoke_outputs(
        output_dir=cleanup_root,
        timestamp="20260612_120000",
    )
    assert not second_source.exists()

    assert first_moved[0].parent.parent != second_moved[0].parent.parent
    assert "smoke_20260612_120000" in first_moved[0].parts
    assert "smoke_20260612_120000_001" in second_moved[0].parts
    assert first_moved[0].read_text(encoding="utf-8") == "first"
    assert second_moved[0].read_text(encoding="utf-8") == "second"
