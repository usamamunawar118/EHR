"""Repository for patients, FHIR bundles and their flattened resources."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ehr_media.fhir.builder import PatientBundle, ResourceRow
from ehr_media.fhir.validation import BundleValidation
from ehr_media.storage.db import Database


@dataclass(frozen=True)
class PatientRow:
    patient_id: str
    mrn: str
    display_name: str
    family_name: str | None
    given_name: str | None
    birth_date: str | None
    gender: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class StoredBundle:
    patient: PatientRow
    bundle: dict[str, Any]
    content_hash: str
    validation_status: str
    validation: BundleValidation


@dataclass(frozen=True)
class StoredResource:
    resource_id: str
    patient_id: str
    resource_type: str
    effective_date: str | None
    doc_key: str | None
    display: str
    text: str | None
    json: str


def _patient_row(r: Any) -> PatientRow:
    return PatientRow(
        r["patient_id"],
        r["mrn"],
        r["display_name"],
        r["family_name"],
        r["given_name"],
        r["birth_date"],
        r["gender"],
        tuple(json.loads(r["flags"])),
    )


class BundleRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def save(self, pb: PatientBundle, validation: BundleValidation) -> None:
        """Upsert a patient, its bundle and flattened resources atomically."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        status = "valid" if validation.valid else "invalid"
        with self._db.connection() as conn:
            conn.execute(
                """INSERT INTO patients (patient_id, mrn, display_name, family_name, given_name,
                                         birth_date, gender, flags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(patient_id) DO UPDATE SET
                     mrn=excluded.mrn, display_name=excluded.display_name,
                     family_name=excluded.family_name, given_name=excluded.given_name,
                     birth_date=excluded.birth_date, gender=excluded.gender,
                     flags=excluded.flags""",
                (
                    pb.patient_id,
                    pb.mrn,
                    pb.display_name,
                    pb.family_name,
                    pb.given_name,
                    pb.birth_date,
                    pb.gender,
                    json.dumps(list(pb.flags)),
                ),
            )
            conn.execute(
                """INSERT INTO fhir_bundles (patient_id, bundle_json, content_hash,
                                             validation_status, validation_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(patient_id) DO UPDATE SET
                     bundle_json=excluded.bundle_json, content_hash=excluded.content_hash,
                     validation_status=excluded.validation_status,
                     validation_json=excluded.validation_json, updated_at=excluded.updated_at""",
                (
                    pb.patient_id,
                    json.dumps(pb.bundle),
                    pb.content_hash,
                    status,
                    validation.model_dump_json(),
                    now,
                ),
            )
            conn.execute("DELETE FROM fhir_resources WHERE patient_id = ?", (pb.patient_id,))
            conn.executemany(
                """INSERT INTO fhir_resources (resource_id, patient_id, resource_type,
                       effective_date, doc_key, display, text, json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [self._resource_params(pb.patient_id, r) for r in pb.resources],
            )

    @staticmethod
    def _resource_params(patient_id: str, r: ResourceRow) -> tuple[Any, ...]:
        return (
            r.resource_id,
            patient_id,
            r.resource_type,
            r.effective_date,
            r.doc_key,
            r.display,
            r.text,
            r.json,
        )

    def get_patient(self, patient_id: str) -> PatientRow | None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return _patient_row(row) if row else None

    def get_patient_by_mrn(self, mrn: str) -> PatientRow | None:
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM patients WHERE mrn = ?", (mrn,)).fetchone()
        return _patient_row(row) if row else None

    def content_hash(self, patient_id: str) -> str | None:
        """Hash of the patient's current bundle (the summary cache's "record hash")."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT content_hash FROM fhir_bundles WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return str(row[0]) if row else None

    def latest_effective_date(self, patient_id: str) -> str | None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT MAX(effective_date) FROM fhir_resources WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
        return str(row[0]) if row and row[0] else None

    def get_resource(self, patient_id: str, resource_id: str) -> StoredResource | None:
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM fhir_resources WHERE patient_id = ? AND resource_id = ?",
                (patient_id, resource_id),
            ).fetchone()
        return self._resource(row) if row else None

    def list_patients(self) -> list[PatientRow]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY family_name, given_name, mrn"
            ).fetchall()
        return [_patient_row(r) for r in rows]

    def get_bundle(self, patient_id: str) -> StoredBundle | None:
        with self._db.connection() as conn:
            row = conn.execute(
                """SELECT b.*, p.mrn, p.display_name, p.family_name, p.given_name, p.birth_date,
                          p.gender, p.flags
                   FROM fhir_bundles b JOIN patients p USING (patient_id)
                   WHERE b.patient_id = ?""",
                (patient_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredBundle(
            patient=_patient_row(row),
            bundle=json.loads(row["bundle_json"]),
            content_hash=row["content_hash"],
            validation_status=row["validation_status"],
            validation=BundleValidation.model_validate_json(row["validation_json"]),
        )

    def iter_bundles(self) -> Iterator[StoredBundle]:
        for patient in self.list_patients():
            stored = self.get_bundle(patient.patient_id)
            if stored is not None:
                yield stored

    def list_resources(self, patient_id: str) -> list[StoredResource]:
        with self._db.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM fhir_resources WHERE patient_id = ?
                   ORDER BY effective_date DESC, resource_type, resource_id""",
                (patient_id,),
            ).fetchall()
        return [self._resource(r) for r in rows]

    def iter_resources(
        self, resource_types: tuple[str, ...] | None = None
    ) -> Iterator[StoredResource]:
        sql = "SELECT * FROM fhir_resources"
        params: tuple[Any, ...] = ()
        if resource_types:
            sql += f" WHERE resource_type IN ({','.join('?' * len(resource_types))})"
            params = resource_types
        sql += " ORDER BY patient_id, effective_date, resource_id"
        with self._db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        for r in rows:
            yield self._resource(r)

    @staticmethod
    def _resource(r: Any) -> StoredResource:
        return StoredResource(
            r["resource_id"],
            r["patient_id"],
            r["resource_type"],
            r["effective_date"],
            r["doc_key"],
            r["display"],
            r["text"],
            r["json"],
        )

    def count(self) -> dict[str, int]:
        with self._db.connection() as conn:
            return {
                "patients": conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
                "bundles": conn.execute("SELECT COUNT(*) FROM fhir_bundles").fetchone()[0],
                "resources": conn.execute("SELECT COUNT(*) FROM fhir_resources").fetchone()[0],
            }
