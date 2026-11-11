"""Reader for the simulated Epic-style JSON export.

The export is intentionally inconsistent: three key-naming styles co-exist (``MRN`` /
``mrn`` / ``MedicalRecordNumber`` ...). Alias tables below translate each to the reader-independent
keys documented on :class:`RawRecord`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from ehr_media.ingestion.models import RecordType, SourceRef
from ehr_media.ingestion.readers.base import RawRecord, UnsupportedInputError, as_text

_PATIENT_ALIASES: dict[str, tuple[str, ...]] = {
    "mrn": ("MRN", "mrn", "MedicalRecordNumber"),
    "family": ("LastName", "family"),
    "given": ("FirstName", "given"),
    "full_name": ("patient_name", "PatientName"),
    "dob": ("dob", "birthDate", "DateOfBirth"),
    "sex": ("sex", "gender", "Gender"),
}
_ENCOUNTER_ALIASES: dict[str, tuple[str, ...]] = {
    "encounter_id": ("encounter_id", "id", "EncounterID"),
    "setting": ("type", "class", "EncounterType"),
    "start": ("start", "period_start", "AdmitDate"),
    "end": ("end", "period_end", "DischargeDate"),
    "reason": ("reason", "ChiefComplaint"),
}
_DOCUMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "doc_id": ("doc_id", "id", "DocumentID"),
    "doc_type": ("type", "documentType", "DocType"),
    "date": ("date", "documentDate", "ServiceDate"),
    "encounter_id": ("encounter_id", "encounterId", "EncounterID"),
    "text": ("text", "body", "Content"),
}


def _pick(data: Mapping[str, Any], aliases: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for canonical, keys in aliases.items():
        out[canonical] = next((as_text(data[k]) for k in keys if k in data), None)
    return out


class JsonExportReader:
    format: ClassVar[str] = "json"
    suffixes: ClassVar[tuple[str, ...]] = (".json",)

    def read(self, path: Path) -> Iterator[RawRecord]:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        patients = data.get("patients") if isinstance(data, dict) else None
        if not isinstance(patients, list):
            raise UnsupportedInputError(f"{path.name}: expected an object with a 'patients' list")

        def ref(locator: str) -> SourceRef:
            return SourceRef(file=path.name, format="json", locator=locator)

        for i, patient in enumerate(patients):
            base = f"patients[{i}]"
            pfields = _pick(patient, _PATIENT_ALIASES)
            name = patient.get("name")
            if isinstance(name, dict):  # style A: {"name": {"family": .., "given": ..}}
                pfields["family"] = as_text(name.get("family"))
                pfields["given"] = as_text(name.get("given"))
            yield RawRecord(RecordType.PATIENT, ref(base), pfields)

            mrn = pfields["mrn"]
            encounters = patient.get("encounters") or patient.get("Encounters") or []
            for j, enc in enumerate(encounters):
                yield RawRecord(
                    RecordType.ENCOUNTER,
                    ref(f"{base}.encounters[{j}]"),
                    {"mrn": mrn, **_pick(enc, _ENCOUNTER_ALIASES)},
                )
            documents = patient.get("documents") or patient.get("Documents") or []
            for k, doc in enumerate(documents):
                yield RawRecord(
                    RecordType.DOCUMENT,
                    ref(f"{base}.documents[{k}]"),
                    {"mrn": mrn, **_pick(doc, _DOCUMENT_ALIASES)},
                )
