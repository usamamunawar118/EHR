"""Duplicate detection for documents, lab results and encounters.

* **exact** - identical after whitespace/case normalisation
* **near**  - same patient/kind/date and text similarity >= ``NEAR_DUPLICATE_THRESHOLD``
  (catches the same note arriving twice, e.g. an export and its OCR'd scan)

The survivor is chosen deterministically: structured sources beat scans, then longer text.
"""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher

from ehr_media.ingestion.audit import AuditTrail
from ehr_media.ingestion.models import (
    AuditAction,
    CanonicalEncounter,
    ClinicalDocument,
    LabResult,
    RecordType,
    Severity,
)
from ehr_media.ingestion.normalizers.documents import text_signature

NEAR_DUPLICATE_THRESHOLD = 0.95
SOURCE_PRIORITY = {"json": 0, "csv": 1, "txt": 2}


def _similarity(a: str, b: str) -> float:
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    if matcher.real_quick_ratio() < NEAR_DUPLICATE_THRESHOLD:
        return 0.0
    if matcher.quick_ratio() < NEAR_DUPLICATE_THRESHOLD:
        return 0.0
    return matcher.ratio()


def dedupe_documents(
    documents: Sequence[ClinicalDocument], trail: AuditTrail
) -> list[ClinicalDocument]:
    ordered = sorted(
        documents,
        key=lambda d: (SOURCE_PRIORITY[d.source.format], -len(d.text), d.record_id),
    )
    kept: list[ClinicalDocument] = []
    signatures: dict[str, str] = {}
    for doc in ordered:
        signature = text_signature(doc.text)
        duplicate_of: tuple[ClinicalDocument, float] | None = None
        for survivor in kept:
            if (survivor.mrn, survivor.kind, survivor.document_date) != (
                doc.mrn,
                doc.kind,
                doc.document_date,
            ):
                continue
            score = (
                1.0
                if signatures[survivor.record_id] == signature
                else _similarity(signatures[survivor.record_id], signature)
            )
            if score >= NEAR_DUPLICATE_THRESHOLD:
                duplicate_of = (survivor, score)
                break
        if duplicate_of is None:
            kept.append(doc)
            signatures[doc.record_id] = signature
            continue
        survivor, score = duplicate_of
        exact = score == 1.0
        trail.add(
            record_id=doc.record_id,
            record_type=RecordType.DOCUMENT,
            mrn=doc.mrn,
            field="document",
            rule="DUPLICATE_EXACT" if exact else "DUPLICATE_NEAR",
            action=AuditAction.DEDUPLICATED,
            before=doc.record_id,
            after=survivor.record_id,
            severity=Severity.INFO,
            message=f"{'identical' if exact else 'near-identical'} to {survivor.record_id} "
            f"(similarity {score:.3f}); kept the {survivor.source.format} copy",
        )
    return sorted(kept, key=lambda d: (d.mrn, d.document_date, d.kind.value, d.doc_id))


def dedupe_labs(labs: Sequence[LabResult], trail: AuditTrail) -> list[LabResult]:
    seen: dict[tuple[object, ...], LabResult] = {}
    kept: list[LabResult] = []
    for lab in sorted(labs, key=lambda x: (SOURCE_PRIORITY[x.source.format], x.record_id)):
        key = (lab.mrn, lab.test_key, lab.collected_date, lab.value, lab.unit)
        if key in seen:
            trail.add(
                record_id=lab.record_id,
                record_type=RecordType.LAB,
                mrn=lab.mrn,
                field="lab",
                rule="DUPLICATE_EXACT",
                action=AuditAction.DEDUPLICATED,
                before=lab.record_id,
                after=seen[key].record_id,
                message=f"same {lab.test_key} result already recorded in {seen[key].record_id}",
            )
            continue
        seen[key] = lab
        kept.append(lab)
    return sorted(kept, key=lambda x: (x.mrn, x.collected_date, x.test_key))


def dedupe_encounters(
    encounters: Sequence[CanonicalEncounter], trail: AuditTrail
) -> list[CanonicalEncounter]:
    seen: dict[tuple[str, str], CanonicalEncounter] = {}
    for enc in sorted(encounters, key=lambda e: e.record_id):
        key = (enc.mrn, enc.encounter_id)
        if key in seen:
            trail.add(
                record_id=enc.record_id,
                record_type=RecordType.ENCOUNTER,
                mrn=enc.mrn,
                field="encounter",
                rule="DUPLICATE_EXACT",
                action=AuditAction.DEDUPLICATED,
                before=enc.record_id,
                after=seen[key].record_id,
                message=f"encounter {enc.encounter_id} already present",
            )
            continue
        seen[key] = enc
    return sorted(seen.values(), key=lambda e: (e.mrn, e.start_date, e.encounter_id))
