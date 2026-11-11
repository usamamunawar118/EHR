"""Shared fixtures."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ehr_media.ingestion.models import IngestResult
from ehr_media.ingestion.pipeline import run_ingest
from ehr_media.synthetic import generate_dataset

# Fixed so that results never depend on the day the suite runs.
REFERENCE_DATE = date(2026, 11, 11)


@pytest.fixture(scope="session")
def reference_date() -> date:
    return REFERENCE_DATE


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The full synthetic dataset generated once per test session."""
    root = tmp_path_factory.mktemp("synthetic")
    generate_dataset(root / "raw")
    return root


@pytest.fixture(scope="session")
def manifest(synthetic_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((synthetic_dir / "DEFECT_MANIFEST.json").read_text("utf-8"))
    return data


@pytest.fixture(scope="session")
def ingest_result(synthetic_dir: Path) -> IngestResult:
    return run_ingest([synthetic_dir / "raw"], reference_date=REFERENCE_DATE)
