"""`ehr build-fhir` / `ehr validate` end to end, including a deliberately corrupted Bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ehr_media.cli import app
from ehr_media.config import get_settings

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at an isolated data dir and produce ingest output in it."""
    monkeypatch.setenv("EHR_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("EHR_RAW_DIR", str(tmp_path / "raw"))
    get_settings.cache_clear()
    assert runner.invoke(app, ["generate-data"]).exit_code == 0
    res = runner.invoke(app, ["ingest", "--reference-date", "2026-11-11"])
    assert res.exit_code == 0, res.output
    yield tmp_path
    get_settings.cache_clear()


def test_build_fhir_then_validate_store(workspace: Path) -> None:
    res = runner.invoke(app, ["build-fhir", "--strict"])
    assert res.exit_code == 0, res.output
    assert "Validation PASS" in res.output
    assert (workspace / "runtime" / "reports" / "validation_report.md").exists()
    assert (workspace / "runtime" / "ehr.db").exists()

    again = runner.invoke(app, ["validate", "--strict"])
    assert again.exit_code == 0, again.output
    assert "60/60 bundles valid" in again.output


def test_validate_flags_a_corrupted_bundle_file(workspace: Path) -> None:
    assert runner.invoke(app, ["build-fhir"]).exit_code == 0
    from ehr_media.storage.bundle_repo import BundleRepository
    from ehr_media.storage.db import Database

    stored = next(BundleRepository(Database(get_settings().db_path)).iter_bundles())
    broken = stored.bundle
    broken["entry"][0]["resource"]["gender"] = "banana"
    broken["entry"][1]["resource"]["subject"]["reference"] = "urn:uuid:ghost"
    path = workspace / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    res = runner.invoke(app, ["validate", "--bundle", str(path), "--strict"])

    assert res.exit_code == 1
    assert "Validation FAIL" in res.output
    report = (workspace / "runtime" / "reports" / "validation_report.md").read_text("utf-8")
    assert "entry[0].resource.gender" in report
    assert "unresolved-reference" in report


def test_build_fhir_requires_ingest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EHR_DATA_DIR", str(tmp_path / "empty"))
    get_settings.cache_clear()
    try:
        res = runner.invoke(app, ["build-fhir"])
        assert res.exit_code != 0
    finally:
        get_settings.cache_clear()
