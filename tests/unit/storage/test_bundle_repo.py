"""SQLite store: migrations, upsert idempotency, retrieval and cascade behaviour."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ehr_media.fhir.builder import PatientBundle
from ehr_media.fhir.validation import validate_bundle
from ehr_media.storage.bundle_repo import BundleRepository
from ehr_media.storage.db import MIGRATIONS, Database, connect, migrate


@pytest.fixture
def repo(tmp_path: Path) -> BundleRepository:
    return BundleRepository(Database(tmp_path / "t.db"))


def test_migrations_are_idempotent_and_versioned(tmp_path: Path) -> None:
    conn = connect(tmp_path / "m.db")
    assert migrate(conn) == len(MIGRATIONS)
    assert migrate(conn) == len(MIGRATIONS)  # second run is a no-op
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"patients", "fhir_bundles", "fhir_resources"} <= tables
    conn.close()


def test_save_and_get_round_trip(repo: BundleRepository, patient_bundle: PatientBundle) -> None:
    validation = validate_bundle(patient_bundle.bundle)
    repo.save(patient_bundle, validation)

    stored = repo.get_bundle(patient_bundle.patient_id)

    assert stored is not None
    assert stored.bundle == patient_bundle.bundle
    assert stored.content_hash == patient_bundle.content_hash
    assert stored.validation_status == "valid"
    assert stored.patient.mrn == "00000042"
    assert repo.get_patient_by_mrn("00000042") is not None
    assert repo.get_patient("nope") is None


def test_save_is_idempotent_no_duplicate_rows(
    repo: BundleRepository, patient_bundle: PatientBundle
) -> None:
    validation = validate_bundle(patient_bundle.bundle)
    repo.save(patient_bundle, validation)
    first = repo.count()
    repo.save(patient_bundle, validation)
    assert repo.count() == first
    assert first["patients"] == 1
    assert first["resources"] == len(patient_bundle.resources)


def test_invalid_bundles_are_stored_with_their_verdict(
    repo: BundleRepository, patient_bundle: PatientBundle
) -> None:
    broken = {**patient_bundle.bundle, "type": "pile"}
    repo.save(patient_bundle, validate_bundle(broken))
    stored = repo.get_bundle(patient_bundle.patient_id)
    assert stored is not None
    assert stored.validation_status == "invalid"
    assert any(i.code == "invalid-code" for i in stored.validation.issues)


def test_resources_can_be_listed_and_filtered(
    repo: BundleRepository, patient_bundle: PatientBundle
) -> None:
    repo.save(patient_bundle, validate_bundle(patient_bundle.bundle))
    all_rows = repo.list_resources(patient_bundle.patient_id)
    assert {r.resource_type for r in all_rows} >= {
        "Patient",
        "DocumentReference",
        "DiagnosticReport",
    }
    docs = list(repo.iter_resources(("DocumentReference",)))
    assert docs
    assert all(r.resource_type == "DocumentReference" and r.text for r in docs)


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    db = Database(tmp_path / "fk.db")
    with db.connection() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO fhir_resources (resource_id, patient_id, resource_type, display, json) "
            "VALUES ('r', 'missing', 'Patient', 'x', '{}')"
        )


def _insert_patient_then_fail(db: Database) -> None:
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO patients (patient_id, mrn, display_name, gender) "
            "VALUES ('p', '1', 'n', 'male')"
        )
        raise RuntimeError("boom")


def test_failed_transaction_rolls_back(tmp_path: Path) -> None:
    db = Database(tmp_path / "rb.db")
    with pytest.raises(RuntimeError):
        _insert_patient_then_fail(db)
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0
