"""Ingestion pipeline orchestration.

    read -> build (normalise + validate) -> reconcile patients -> quarantine conflicts
         -> dedupe -> link encounters -> assemble PatientRecords

Every transformation is recorded in the audit trail; every rejection lands in quarantine.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

from ehr_media.ingestion.audit import AuditTrail
from ehr_media.ingestion.builders import (
    BuildContext,
    EncounterKey,
    PatientClaim,
    RecordRejected,
    build_document,
    build_encounter,
    build_lab,
    build_patient_claim,
)
from ehr_media.ingestion.dedupe import dedupe_documents, dedupe_encounters, dedupe_labs
from ehr_media.ingestion.linking import link_encounters
from ehr_media.ingestion.models import (
    AuditAction,
    CanonicalEncounter,
    ClinicalDocument,
    IngestResult,
    LabResult,
    PatientRecord,
    QuarantinedRecord,
    RecordType,
    Severity,
    SourceRef,
)
from ehr_media.ingestion.readers import RawRecord, discover_inputs, read_all
from ehr_media.ingestion.reconcile import reconcile_patients

logger = logging.getLogger(__name__)

_RAW_PREVIEW_CHARS = 300
T = TypeVar("T")
Built = ClinicalDocument | LabResult | CanonicalEncounter


def _preview(fields: dict[str, Any]) -> dict[str, Any]:
    """Quarantine keeps a bounded preview of the raw record (limits PHI at rest)."""
    return {
        k: (
            v[:_RAW_PREVIEW_CHARS] + "…"
            if isinstance(v, str) and len(v) > _RAW_PREVIEW_CHARS
            else v
        )
        for k, v in fields.items()
    }


def _source(item: Built) -> SourceRef:
    return (
        item.source
        if not isinstance(item, CanonicalEncounter)
        else SourceRef(file="(encounter)", format="json", locator=item.record_id)
    )


def run_ingest(inputs: Sequence[Path], *, reference_date: date | None = None) -> IngestResult:
    """Ingest every supported file under ``inputs`` and return the clean IR + audit + quarantine."""
    reference = reference_date or date.today()
    trail = AuditTrail()
    ctx = BuildContext(reference_date=reference, trail=trail)
    quarantine: list[QuarantinedRecord] = []

    paths = sorted({p for root in inputs for p in discover_inputs(root)})
    raws = list(read_all(paths))
    logger.info("read %d raw records from %d files", len(raws), len(paths))

    def guarded(raw: RawRecord, build: Callable[[RawRecord], T]) -> T | None:
        try:
            return build(raw)
        except RecordRejected as exc:
            trail.add(
                record_id=raw.record_id,
                record_type=raw.record_type,
                mrn=exc.mrn,
                field="record",
                rule=exc.rule,
                action=AuditAction.REJECTED,
                severity=Severity.ERROR,
                message=exc.message,
            )
            quarantine.append(
                QuarantinedRecord(
                    record_id=raw.record_id,
                    record_type=raw.record_type,
                    mrn=exc.mrn,
                    reason=exc.rule,
                    message=exc.message,
                    source=raw.source,
                    raw=_preview(raw.fields),
                )
            )
            return None

    by_type: dict[RecordType, list[RawRecord]] = defaultdict(list)
    for raw in raws:
        by_type[raw.record_type].append(raw)

    def collect(rtype: RecordType, build: Callable[[RawRecord], T]) -> list[T]:
        built = (guarded(r, build) for r in by_type[rtype])
        return [item for item in built if item is not None]

    claims = collect(RecordType.PATIENT, lambda r: build_patient_claim(r, ctx))
    encounters = dedupe_encounters(
        collect(RecordType.ENCOUNTER, lambda r: build_encounter(r, ctx)), trail
    )
    starts: dict[EncounterKey, date] = {(e.mrn, e.encounter_id): e.start_date for e in encounters}
    documents = collect(RecordType.DOCUMENT, lambda r: build_document(r, ctx, starts))
    labs = collect(RecordType.LAB, lambda r: build_lab(r, ctx))

    all_mrns = {
        *(e.mrn for e in encounters),
        *(d.mrn for d in documents),
        *(x.mrn for x in labs),
    }
    outcome = reconcile_patients(claims, all_mrns, trail)

    _quarantine_conflicts(outcome.conflicts, claims, encounters, documents, labs, trail, quarantine)
    clean_docs = dedupe_documents([d for d in documents if d.mrn not in outcome.conflicts], trail)
    clean_labs = dedupe_labs([x for x in labs if x.mrn not in outcome.conflicts], trail)
    clean_encs = [e for e in encounters if e.mrn not in outcome.conflicts]

    enc_by, doc_by, lab_by = (defaultdict(list) for _ in range(3))
    for e in clean_encs:
        enc_by[e.mrn].append(e)
    for d in clean_docs:
        doc_by[d.mrn].append(d)
    for lab in clean_labs:
        lab_by[lab.mrn].append(lab)

    linked: dict[str, tuple[list[CanonicalEncounter], list[ClinicalDocument], list[LabResult]]]
    linked = {
        mrn: link_encounters(mrn, enc_by[mrn], doc_by[mrn], lab_by[mrn], trail)
        for mrn in outcome.patients
    }

    audit_entries = trail.entries
    by_mrn_audit = defaultdict(list)
    for entry in audit_entries:
        if entry.mrn:
            by_mrn_audit[entry.mrn].append(entry)

    records = tuple(
        PatientRecord(
            patient=patient,
            encounters=tuple(linked[mrn][0]),
            documents=tuple(linked[mrn][1]),
            lab_results=tuple(linked[mrn][2]),
            audit=tuple(by_mrn_audit[mrn]),
        )
        for mrn, patient in sorted(outcome.patients.items())
    )
    stats = {
        "files": len(paths),
        "raw_records": len(raws),
        "patients": len(records),
        "documents": sum(len(r.documents) for r in records),
        "lab_results": sum(len(r.lab_results) for r in records),
        "encounters": sum(len(r.encounters) for r in records),
        "quarantined": len(quarantine),
        "conflicting_mrns": len(outcome.conflicts),
        "audit_entries": len(audit_entries),
    }
    logger.info("ingest complete: %s", stats)
    return IngestResult(
        patients=records, quarantine=tuple(quarantine), audit=tuple(audit_entries), stats=stats
    )


def _quarantine_conflicts(
    conflicts: dict[str, str],
    claims: Sequence[PatientClaim],
    encounters: Sequence[CanonicalEncounter],
    documents: Sequence[ClinicalDocument],
    labs: Sequence[LabResult],
    trail: AuditTrail,
    quarantine: list[QuarantinedRecord],
) -> None:
    for mrn, reason in conflicts.items():
        first_claim = next((c for c in claims if c.mrn == mrn), None)
        quarantine.append(
            QuarantinedRecord(
                record_id=f"patient:{mrn}",
                record_type=RecordType.PATIENT,
                mrn=mrn,
                reason="IDENTIFIER_CONFLICT",
                message=reason,
                source=first_claim.source
                if first_claim
                else SourceRef(file="(reconciliation)", format="json", locator=mrn),
            )
        )
    items: list[tuple[Built, RecordType]] = [
        *((e, RecordType.ENCOUNTER) for e in encounters),
        *((d, RecordType.DOCUMENT) for d in documents),
        *((x, RecordType.LAB) for x in labs),
    ]
    for item, rtype in items:
        if item.mrn not in conflicts:
            continue
        trail.add(
            record_id=item.record_id,
            record_type=rtype,
            mrn=item.mrn,
            field="record",
            rule="IDENTIFIER_CONFLICT",
            action=AuditAction.REJECTED,
            severity=Severity.ERROR,
            message="quarantined with its MRN: " + conflicts[item.mrn],
        )
        quarantine.append(
            QuarantinedRecord(
                record_id=item.record_id,
                record_type=rtype,
                mrn=item.mrn,
                reason="IDENTIFIER_CONFLICT",
                message=conflicts[item.mrn],
                source=_source(item),
                raw=_preview(item.model_dump(mode="json", exclude={"source"})),
            )
        )
