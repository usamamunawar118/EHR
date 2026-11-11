"""Guarantee every document and lab result belongs to an encounter.

FHIR ``subject``/``encounter`` references must resolve, so after cleaning:

1. items citing a *known* encounter id keep it;
2. items citing an *unknown* id get a synthesised encounter stub carrying that id;
3. items with no id are linked to the single encounter whose period covers their date, or else
   get a synthesised per-day encounter.

All synthesis/linking is audited.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from ehr_media.ingestion.audit import AuditTrail
from ehr_media.ingestion.models import (
    AuditAction,
    CanonicalEncounter,
    ClinicalDocument,
    EncounterClass,
    LabResult,
    RecordType,
    Severity,
)


def link_encounters(
    mrn: str,
    encounters: Sequence[CanonicalEncounter],
    documents: Sequence[ClinicalDocument],
    labs: Sequence[LabResult],
    trail: AuditTrail,
) -> tuple[list[CanonicalEncounter], list[ClinicalDocument], list[LabResult]]:
    known: dict[str, CanonicalEncounter] = {e.encounter_id: e for e in encounters}
    stubs: dict[str, CanonicalEncounter] = {}

    def stub(eid: str, when: date, why: str) -> str:
        existing = stubs.get(eid)
        if existing is None:
            stubs[eid] = CanonicalEncounter(
                record_id=f"encounter:synthesized:{mrn}:{eid}",
                mrn=mrn,
                encounter_id=eid,
                encounter_class=EncounterClass.AMBULATORY,
                start_date=when,
                end_date=None,
                reason=None,
                synthesized=True,
            )
            trail.add(
                record_id=f"encounter:synthesized:{mrn}:{eid}",
                record_type=RecordType.ENCOUNTER,
                mrn=mrn,
                field="encounter",
                rule="ENCOUNTER_SYNTHESIZED",
                action=AuditAction.IMPUTED,
                after=eid,
                message=why,
            )
        elif when < existing.start_date:
            stubs[eid] = existing.model_copy(update={"start_date": when})
        return eid

    def covering(when: date) -> list[CanonicalEncounter]:
        return [e for e in encounters if e.start_date <= when <= (e.end_date or e.start_date)]

    def resolve(record_id: str, rtype: RecordType, eid: str | None, when: date) -> str:
        if eid and eid in known:
            return eid
        if eid:
            return stub(eid, when, "source cites an encounter that has no encounter record")
        matches = covering(when)
        if len(matches) == 1:
            trail.add(
                record_id=record_id,
                record_type=rtype,
                mrn=mrn,
                field="encounter_id",
                rule="ENCOUNTER_LINKED_BY_DATE",
                action=AuditAction.IMPUTED,
                after=matches[0].encounter_id,
                severity=Severity.INFO,
                message="no encounter id; linked to the only encounter covering the date",
            )
            return matches[0].encounter_id
        return stub(
            f"SYN-{mrn}-{when:%Y%m%d}",
            when,
            "no encounter id and no unique covering encounter; synthesised per-day stub",
        )

    new_docs = [
        d.model_copy(
            update={
                "encounter_id": resolve(
                    d.record_id, RecordType.DOCUMENT, d.encounter_id, d.document_date
                )
            }
        )
        for d in documents
    ]
    new_labs = [
        lab.model_copy(
            update={
                "encounter_id": resolve(
                    lab.record_id, RecordType.LAB, lab.encounter_id, lab.collected_date
                )
            }
        )
        for lab in labs
    ]

    ordered = sorted({**known, **stubs}.values(), key=lambda e: (e.start_date, e.encounter_id))
    return ordered, new_docs, new_labs
