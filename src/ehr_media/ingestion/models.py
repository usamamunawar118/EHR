"""Canonical intermediate representation (IR) produced by the ingestion pipeline.

Everything downstream (FHIR mapping, summarisation, search) consumes these models, never the raw
source formats. Models are immutable and strictly validated (``extra="forbid"``).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MRN_WIDTH = 8
Mrn = Annotated[str, StringConstraints(pattern=rf"^\d{{{MRN_WIDTH}}}$")]


class Gender(StrEnum):
    """FHIR ``administrative-gender`` value set."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class DocumentKind(StrEnum):
    PROGRESS_NOTE = "progress_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    IMAGING_REPORT = "imaging_report"


class EncounterClass(StrEnum):
    """FHIR v3-ActCode encounter class codes."""

    INPATIENT = "IMP"
    AMBULATORY = "AMB"
    EMERGENCY = "EMER"


class LabFlag(StrEnum):
    NORMAL = "N"
    HIGH = "H"
    LOW = "L"


class RecordType(StrEnum):
    PATIENT = "patient"
    ENCOUNTER = "encounter"
    DOCUMENT = "document"
    LAB = "lab"


class AuditAction(StrEnum):
    NORMALIZED = "normalized"
    IMPUTED = "imputed"
    FLAGGED = "flagged"
    DEDUPLICATED = "deduplicated"
    MERGED = "merged"
    REJECTED = "rejected"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceRef(_Frozen):
    """Where a record came from (for traceability back to the raw file)."""

    file: str
    format: Literal["json", "csv", "txt"]
    locator: str

    @property
    def key(self) -> str:
        return f"{self.file}:{self.locator}"


class AuditEntry(_Frozen):
    """One change/decision applied to one record."""

    record_id: str
    record_type: RecordType
    mrn: str | None = None
    field: str
    rule: str
    action: AuditAction
    before: str | None = None
    after: str | None = None
    severity: Severity = Severity.INFO
    message: str = ""


class CanonicalPatient(_Frozen):
    mrn: Mrn
    family_name: str | None = None
    given_name: str | None = None
    birth_date: date | None = None
    gender: Gender = Gender.UNKNOWN
    flags: tuple[str, ...] = ()
    sources: tuple[SourceRef, ...] = ()

    @property
    def display_name(self) -> str:
        parts = [p for p in (self.given_name, self.family_name) if p]
        return " ".join(parts) if parts else "Unknown patient"


class CanonicalEncounter(_Frozen):
    record_id: str
    mrn: Mrn
    encounter_id: str
    encounter_class: EncounterClass
    start_date: date
    end_date: date | None = None
    reason: str | None = None
    synthesized: bool = False


class ClinicalDocument(_Frozen):
    record_id: str
    doc_id: str = Field(description="Stable content-derived id, independent of the source file")
    mrn: Mrn
    kind: DocumentKind
    title: str
    document_date: date
    encounter_id: str | None = None
    text: str = Field(min_length=1)
    source: SourceRef


class LabResult(_Frozen):
    record_id: str
    mrn: Mrn
    test_key: str
    test_name: str
    value: float
    unit: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None
    flag: LabFlag | None = None  # None = unknown (no source flag and no reference range)
    collected_date: date
    encounter_id: str | None = None
    source: SourceRef


class PatientRecord(_Frozen):
    """Everything known about one (non-conflicting) patient after cleaning."""

    patient: CanonicalPatient
    encounters: tuple[CanonicalEncounter, ...] = ()
    documents: tuple[ClinicalDocument, ...] = ()
    lab_results: tuple[LabResult, ...] = ()
    audit: tuple[AuditEntry, ...] = ()


class QuarantinedRecord(_Frozen):
    """A record that was *not* admitted to the clean set (never silently dropped)."""

    record_id: str
    record_type: RecordType
    mrn: str | None = None
    reason: str
    message: str = ""
    source: SourceRef
    raw: dict[str, Any] = Field(default_factory=dict)


class IngestResult(_Frozen):
    patients: tuple[PatientRecord, ...] = ()
    quarantine: tuple[QuarantinedRecord, ...] = ()
    audit: tuple[AuditEntry, ...] = ()
    stats: dict[str, int] = Field(default_factory=dict)

    def audit_by_record(self) -> dict[str, list[AuditEntry]]:
        grouped: dict[str, list[AuditEntry]] = {}
        for entry in self.audit:
            grouped.setdefault(entry.record_id, []).append(entry)
        return grouped
