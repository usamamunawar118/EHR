"""Assemble one FHIR ``Bundle`` per patient and the flat rows used for storage/search."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

from fhir.resources.R4B.bundle import Bundle

from ehr_media.fhir import mappers, terminology
from ehr_media.fhir.ids import resource_id, urn
from ehr_media.fhir.text import extract_section, render_lab_panel
from ehr_media.ingestion.models import DocumentKind, LabResult, PatientRecord


@dataclass(frozen=True)
class ResourceRow:
    """Denormalised view of one resource (drives listing, filtering and indexing)."""

    resource_id: str
    resource_type: str
    effective_date: str | None
    doc_key: str | None
    display: str
    text: str | None
    json: str


@dataclass(frozen=True)
class PatientBundle:
    patient_id: str
    mrn: str
    display_name: str
    family_name: str | None
    given_name: str | None
    birth_date: str | None
    gender: str
    flags: tuple[str, ...]
    bundle: dict[str, Any]
    resources: tuple[ResourceRow, ...]
    content_hash: str


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(bundle: dict[str, Any]) -> str:
    """SHA-256 of the canonical bundle JSON: the "record hash" used for cache invalidation."""
    return hashlib.sha256(canonical_json(bundle).encode("utf-8")).hexdigest()


def _row(
    model: Any,
    *,
    rtype: str,
    when: date | None,
    display: str,
    doc_key: str | None = None,
    text: str | None = None,
) -> ResourceRow:
    return ResourceRow(
        resource_id=model.id,
        resource_type=rtype,
        effective_date=when.isoformat() if when else None,
        doc_key=doc_key,
        display=display,
        text=text,
        json=canonical_json(mappers.to_dict(model)),
    )


def build_patient_bundle(record: PatientRecord) -> PatientBundle:
    """Map one cleaned patient record to a FHIR collection Bundle (unvalidated)."""
    p = record.patient
    patient = mappers.map_patient(p)
    pid = patient.id
    models: list[tuple[Any, ResourceRow]] = [
        (patient, _row(patient, rtype="Patient", when=None, display=p.display_name))
    ]

    for enc in sorted(record.encounters, key=lambda e: (e.start_date, e.encounter_id)):
        m = mappers.map_encounter(enc, pid)
        label = terminology.ENCOUNTER_CLASS[enc.encounter_class].display.capitalize()
        models.append(
            (
                m,
                _row(
                    m,
                    rtype="Encounter",
                    when=enc.start_date,
                    display=f"{label}: {enc.reason}" if enc.reason else label,
                ),
            )
        )

    def enc_id(encounter_id: str | None) -> str:
        if encounter_id is None:  # ingestion links/synthesises one for every item
            raise ValueError(f"record for MRN {p.mrn} has an item without an encounter")
        return mappers.encounter_rid(p.mrn, encounter_id)

    for doc in record.documents:
        eid = enc_id(doc.encounter_id)
        dref = mappers.map_document_reference(doc, pid, eid)
        models.append(
            (
                dref,
                _row(
                    dref,
                    rtype="DocumentReference",
                    when=doc.document_date,
                    display=doc.title,
                    doc_key=doc.doc_id,
                    text=doc.text,
                ),
            )
        )
        if doc.kind is DocumentKind.IMAGING_REPORT:
            rep = mappers.map_imaging_report(doc, pid, eid)
            exam = extract_section(doc.text, "EXAM") or doc.title
            models.append(
                (
                    rep,
                    _row(
                        rep,
                        rtype="DiagnosticReport",
                        when=doc.document_date,
                        display=f"Imaging report: {exam}",
                        doc_key=doc.doc_id,
                        text=doc.text,
                    ),
                )
            )

    panels: dict[tuple[str | None, date], list[LabResult]] = defaultdict(list)
    for lab in record.lab_results:
        panels[(lab.encounter_id, lab.collected_date)].append(lab)
    for (encounter_id, collected), labs in sorted(
        panels.items(), key=lambda kv: (kv[0][1], kv[0][0] or "")
    ):
        eid = enc_id(encounter_id)
        obs_models = [(mappers.map_observation(lab, pid, eid), lab) for lab in labs]
        report = mappers.map_lab_report(labs, pid, eid)
        abnormal = sum(1 for _, lab in obs_models if lab.flag and lab.flag.value != "N")
        models.append(
            (
                report,
                _row(
                    report,
                    rtype="DiagnosticReport",
                    when=collected,
                    display=f"Laboratory report: {len(labs)} results, {abnormal} abnormal",
                    doc_key=report.id,
                    text=render_lab_panel(labs),
                ),
            )
        )
        for obs, lab in obs_models:
            models.append(
                (
                    obs,
                    _row(
                        obs,
                        rtype="Observation",
                        when=lab.collected_date,
                        display=f"{lab.test_name} {lab.value:g} {lab.unit or ''}".strip(),
                    ),
                )
            )

    bundle_model = Bundle.model_validate(
        {
            "resourceType": "Bundle",
            "id": resource_id("Bundle", p.mrn),
            "identifier": {"system": terminology.BUNDLE_SYSTEM, "value": p.mrn},
            "type": "collection",
            "entry": [{"fullUrl": urn(m.id), "resource": mappers.to_dict(m)} for m, _ in models],
        }
    )
    bundle = mappers.to_dict(bundle_model)
    return PatientBundle(
        patient_id=pid,
        mrn=p.mrn,
        display_name=p.display_name,
        family_name=p.family_name,
        given_name=p.given_name,
        birth_date=p.birth_date.isoformat() if p.birth_date else None,
        gender=p.gender.value,
        flags=p.flags,
        bundle=bundle,
        resources=tuple(row for _, row in models),
        content_hash=content_hash(bundle),
    )
