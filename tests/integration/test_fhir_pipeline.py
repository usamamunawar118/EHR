"""Task 2 end-to-end on the full synthetic dataset."""

from __future__ import annotations

from pathlib import Path

from ehr_media.fhir.report import write_report
from ehr_media.fhir.service import build_and_validate, store_bundles
from ehr_media.ingestion.models import IngestResult
from ehr_media.storage.bundle_repo import BundleRepository
from ehr_media.storage.db import Database


def test_every_clean_patient_gets_a_valid_bundle(ingest_result: IngestResult) -> None:
    result = build_and_validate(ingest_result.patients)

    assert result.report.bundles_total == len(ingest_result.patients) == 60
    assert result.report.total_errors == 0, result.report.issues_by_code
    assert result.report.bundles_invalid == 0
    assert {"Patient", "DocumentReference", "DiagnosticReport", "Encounter", "Observation"} <= set(
        result.report.resource_counts
    )


def test_only_expected_warnings_are_raised(ingest_result: IngestResult) -> None:
    report = build_and_validate(ingest_result.patients).report
    # Patients whose DOB could not be established -> exactly the "missing-birthdate" warnings.
    assert set(report.issues_by_code) <= {
        "warning:missing-birthdate",
        "warning:uncoded-observation",
    }
    assert report.issues_by_code.get("warning:missing-birthdate", 0) == sum(
        1 for r in ingest_result.patients if r.patient.birth_date is None
    )


def test_bundles_are_stored_and_retrievable(ingest_result: IngestResult, tmp_path: Path) -> None:
    result = build_and_validate(ingest_result.patients)
    repo = BundleRepository(Database(tmp_path / "ehr.db"))

    assert store_bundles(result, repo) == 60
    counts = repo.count()
    assert counts["patients"] == counts["bundles"] == 60
    assert counts["resources"] == sum(len(b.resources) for b in result.bundles)

    one = result.bundles[0]
    stored = repo.get_bundle(one.patient_id)
    assert stored is not None
    assert stored.bundle == one.bundle

    # Re-running is idempotent (deterministic ids + upserts).
    store_bundles(result, repo)
    assert repo.count() == counts


def test_report_files_are_written(ingest_result: IngestResult, tmp_path: Path) -> None:
    report = build_and_validate(ingest_result.patients).report
    json_path, md_path = write_report(report, tmp_path)
    assert "**Result: PASS**" in md_path.read_text("utf-8")
    assert json_path.stat().st_size > 0
