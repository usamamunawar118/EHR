"""Pure mapping functions: canonical IR -> validated ``fhir.resources`` (R4B) models.

Each mapper builds a plain dict and passes it through ``Model.model_validate`` so structural errors
surface at mapping time with a JSON path. Value-set bindings and cross-resource references are
checked later by :mod:`ehr_media.fhir.validation`.

R4B vs R4: ``fhir.resources`` 8.x (the Pydantic-v2 line) ships R5, R4B and STU3 but no pure R4.
R4B (4.3.0) is wire-compatible with R4 for every resource type used here.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from datetime import date
from typing import Any

from fhir.resources.R4B.diagnosticreport import DiagnosticReport
from fhir.resources.R4B.documentreference import DocumentReference
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.patient import Patient

from ehr_media.fhir import terminology as term
from ehr_media.fhir.ids import resource_id, urn
from ehr_media.fhir.text import extract_section, lab_conclusion, render_lab_panel
from ehr_media.ingestion.models import (
    CanonicalEncounter,
    CanonicalPatient,
    ClinicalDocument,
    LabResult,
)

JsonDict = dict[str, Any]


def to_dict(model: Any) -> JsonDict:
    """Serialise a FHIR model to a plain JSON-compatible dict (no ``None``s)."""
    data: JsonDict = json.loads(model.model_dump_json())
    return data


def _ref(rid: str) -> JsonDict:
    return {"reference": urn(rid)}


# --- identifiers ---------------------------------------------------------------------------
def patient_rid(mrn: str) -> str:
    return resource_id("Patient", mrn)


def encounter_rid(mrn: str, encounter_id: str) -> str:
    return resource_id("Encounter", mrn, encounter_id)


def document_rid(mrn: str, doc_id: str) -> str:
    return resource_id("DocumentReference", mrn, doc_id)


def imaging_report_rid(mrn: str, doc_id: str) -> str:
    return resource_id("DiagnosticReport", mrn, "imaging", doc_id)


def lab_report_rid(mrn: str, encounter_id: str | None, collected: date) -> str:
    return resource_id("DiagnosticReport", mrn, "lab", encounter_id, collected.isoformat())


def observation_rid(lab: LabResult) -> str:
    return resource_id(
        "Observation", lab.mrn, lab.test_key, lab.collected_date.isoformat(), lab.value, lab.unit
    )


# --- shared builders -----------------------------------------------------------------------
def _attachment(text: str, title: str, created: date) -> JsonDict:
    raw = text.encode("utf-8")
    return {
        "contentType": "text/plain; charset=utf-8",
        "language": "en",
        "data": base64.b64encode(raw).decode("ascii"),
        "size": len(raw),
        # FHIR mandates SHA-1 for Attachment.hash; this is an integrity hint, not security.
        "hash": base64.b64encode(hashlib.sha1(raw, usedforsecurity=False).digest()).decode("ascii"),
        "title": title,
        "creation": created.isoformat(),
    }


def _instant(d: date) -> str:
    # Sources only carry a calendar date; FHIR `instant` needs a time, so use midnight UTC.
    return f"{d.isoformat()}T00:00:00Z"


def _codeable(code: term.Code | None, text: str | None = None) -> JsonDict:
    out: JsonDict = {}
    if code is not None:
        out["coding"] = [code.as_coding()]
    if text:
        out["text"] = text
    return out


# --- mappers -------------------------------------------------------------------------------
def map_patient(p: CanonicalPatient) -> Patient:
    data: JsonDict = {
        "resourceType": "Patient",
        "id": patient_rid(p.mrn),
        "active": True,
        "identifier": [
            {
                "use": "usual",
                "type": {"coding": [term.MRN_TYPE.as_coding()]},
                "system": term.MRN_SYSTEM,
                "value": p.mrn,
            }
        ],
        "gender": term.gender_code(p.gender),
    }
    if p.family_name or p.given_name:
        name: JsonDict = {"use": "official"}
        if p.family_name:
            name["family"] = p.family_name
        if p.given_name:
            name["given"] = p.given_name.split()
        data["name"] = [name]
    if p.birth_date:
        data["birthDate"] = p.birth_date.isoformat()
    if p.flags:
        # Data-quality findings from ingestion travel with the resource.
        data["meta"] = {
            "tag": [{"system": term.DATA_QUALITY_SYSTEM, "code": flag} for flag in p.flags]
        }
    return Patient.model_validate(data)


def map_encounter(e: CanonicalEncounter, patient_id: str) -> Encounter:
    period: JsonDict = {"start": e.start_date.isoformat()}
    if e.end_date:
        period["end"] = e.end_date.isoformat()
    data: JsonDict = {
        "resourceType": "Encounter",
        "id": encounter_rid(e.mrn, e.encounter_id),
        "identifier": [{"system": term.ENCOUNTER_SYSTEM, "value": e.encounter_id}],
        "status": "finished",
        "class": term.ENCOUNTER_CLASS[e.encounter_class].as_coding(),
        "subject": _ref(patient_id),
        "period": period,
    }
    if e.reason:
        data["reasonCode"] = [{"text": e.reason}]
    if e.synthesized:
        data["meta"] = {"tag": [{"system": term.DATA_QUALITY_SYSTEM, "code": "SYNTHESIZED"}]}
    return Encounter.model_validate(data)


def map_document_reference(
    d: ClinicalDocument, patient_id: str, encounter_id: str
) -> DocumentReference:
    data: JsonDict = {
        "resourceType": "DocumentReference",
        "id": document_rid(d.mrn, d.doc_id),
        "identifier": [{"system": term.DOCUMENT_SYSTEM, "value": d.doc_id}],
        "status": "current",
        "docStatus": "final",
        "type": _codeable(term.DOCUMENT_TYPE[d.kind], d.title),
        "subject": _ref(patient_id),
        "date": _instant(d.document_date),
        "description": f"{d.title} dated {d.document_date.isoformat()}",
        "content": [{"attachment": _attachment(d.text, d.title, d.document_date)}],
        "context": {
            "encounter": [_ref(encounter_id)],
            "period": {"start": d.document_date.isoformat()},
        },
    }
    return DocumentReference.model_validate(data)


def map_imaging_report(d: ClinicalDocument, patient_id: str, encounter_id: str) -> DiagnosticReport:
    exam = extract_section(d.text, "EXAM") or d.title
    data: JsonDict = {
        "resourceType": "DiagnosticReport",
        "id": imaging_report_rid(d.mrn, d.doc_id),
        "identifier": [{"system": term.DOCUMENT_SYSTEM, "value": f"imaging-{d.doc_id}"}],
        "status": "final",
        "category": [_codeable(term.CATEGORY_RAD)],
        "code": _codeable(term.IMAGING_REPORT_CODE, exam),
        "subject": _ref(patient_id),
        "encounter": _ref(encounter_id),
        "effectiveDateTime": d.document_date.isoformat(),
        "presentedForm": [_attachment(d.text, exam, d.document_date)],
    }
    if impression := extract_section(d.text, "IMPRESSION"):
        data["conclusion"] = impression
    return DiagnosticReport.model_validate(data)


def map_observation(lab: LabResult, patient_id: str, encounter_id: str) -> Observation:
    loinc = term.LAB_LOINC.get(lab.test_key)
    quantity: JsonDict = {"value": lab.value}
    if lab.unit:
        quantity["unit"] = lab.unit
        if ucum := term.UCUM_UNITS.get(lab.unit):
            quantity.update({"system": term.UCUM, "code": ucum})
    data: JsonDict = {
        "resourceType": "Observation",
        "id": observation_rid(lab),
        "status": "final",
        "category": [_codeable(term.OBS_CATEGORY_LAB)],
        "code": _codeable(loinc, lab.test_name),
        "subject": _ref(patient_id),
        "encounter": _ref(encounter_id),
        "effectiveDateTime": lab.collected_date.isoformat(),
        "valueQuantity": quantity,
    }
    if lab.flag is not None:
        data["interpretation"] = [_codeable(term.INTERPRETATION[lab.flag])]
    if lab.ref_low is not None or lab.ref_high is not None:
        rng: JsonDict = {}
        for key, bound in (("low", lab.ref_low), ("high", lab.ref_high)):
            if bound is not None:
                rng[key] = {"value": bound, **({"unit": lab.unit} if lab.unit else {})}
        data["referenceRange"] = [rng]
    return Observation.model_validate(data)


def map_lab_report(
    labs: Sequence[LabResult], patient_id: str, encounter_id: str
) -> DiagnosticReport:
    """One DiagnosticReport per (encounter, collection date) pointing at its Observations."""
    first = labs[0]
    data: JsonDict = {
        "resourceType": "DiagnosticReport",
        "id": lab_report_rid(first.mrn, first.encounter_id, first.collected_date),
        "status": "final",
        "category": [_codeable(term.CATEGORY_LAB)],
        "code": _codeable(term.LAB_REPORT_CODE),
        "subject": _ref(patient_id),
        "encounter": _ref(encounter_id),
        "effectiveDateTime": first.collected_date.isoformat(),
        "result": [_ref(observation_rid(lab)) for lab in labs],
        "conclusion": lab_conclusion(labs),
        "presentedForm": [
            _attachment(render_lab_panel(labs), "Laboratory report", first.collected_date)
        ],
    }
    return DiagnosticReport.model_validate(data)
