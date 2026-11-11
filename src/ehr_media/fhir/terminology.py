"""Code systems, value sets and local code maps.

LOINC codes below were checked against the NLM LOINC registry. Anything not mapped is emitted as
text-only ``CodeableConcept`` - a code is never invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from ehr_media.ingestion.models import DocumentKind, EncounterClass, Gender, LabFlag

# --- code systems -------------------------------------------------------------------------
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
V2_0203 = "http://terminology.hl7.org/CodeSystem/v2-0203"  # identifier type
V2_0074 = "http://terminology.hl7.org/CodeSystem/v2-0074"  # diagnostic service section
V3_ACT_CODE = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
V3_OBS_INTERP = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"

# Placeholder identifier systems for this synthetic "hospital". Real deployments use the
# organisation's registered OIDs/URIs.
MRN_SYSTEM = "http://hospital.example.org/fhir/sid/mrn"
ENCOUNTER_SYSTEM = "http://hospital.example.org/fhir/sid/encounter"
DOCUMENT_SYSTEM = "http://hospital.example.org/fhir/sid/document"
BUNDLE_SYSTEM = "http://hospital.example.org/fhir/sid/patient-bundle"
DATA_QUALITY_SYSTEM = "http://hospital.example.org/fhir/CodeSystem/data-quality"

ALLOWED_CODE_SYSTEMS = frozenset(
    {LOINC, UCUM, V2_0203, V2_0074, V3_ACT_CODE, V3_OBS_INTERP, OBS_CATEGORY, DATA_QUALITY_SYSTEM}
)


@dataclass(frozen=True)
class Code:
    system: str
    code: str
    display: str

    def as_coding(self) -> dict[str, str]:
        return {"system": self.system, "code": self.code, "display": self.display}


# --- required value sets (fhir.resources checks structure, not these bindings) -----------------
VALUE_SETS: dict[str, frozenset[str]] = {
    "administrative-gender": frozenset({"male", "female", "other", "unknown"}),
    "document-reference-status": frozenset({"current", "superseded", "entered-in-error"}),
    "composition-status": frozenset({"preliminary", "final", "amended", "entered-in-error"}),
    "diagnostic-report-status": frozenset(
        {
            "registered",
            "partial",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "appended",
            "cancelled",
            "entered-in-error",
            "unknown",
        }
    ),
    "observation-status": frozenset(
        {
            "registered",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "cancelled",
            "entered-in-error",
            "unknown",
        }
    ),
    "encounter-status": frozenset(
        {
            "planned",
            "arrived",
            "triaged",
            "in-progress",
            "onleave",
            "finished",
            "cancelled",
            "entered-in-error",
            "unknown",
        }
    ),
    "bundle-type": frozenset(
        {
            "document",
            "message",
            "transaction",
            "transaction-response",
            "batch",
            "batch-response",
            "history",
            "searchset",
            "collection",
        }
    ),
}

# --- mappings ----------------------------------------------------------------------------
MRN_TYPE = Code(V2_0203, "MR", "Medical record number")

DOCUMENT_TYPE: dict[DocumentKind, Code] = {
    DocumentKind.PROGRESS_NOTE: Code(LOINC, "11506-3", "Progress note"),
    DocumentKind.DISCHARGE_SUMMARY: Code(LOINC, "18842-5", "Discharge summary"),
    DocumentKind.IMAGING_REPORT: Code(LOINC, "18748-4", "Diagnostic imaging study"),
}
LAB_REPORT_CODE = Code(LOINC, "11502-2", "Laboratory report")
IMAGING_REPORT_CODE = DOCUMENT_TYPE[DocumentKind.IMAGING_REPORT]
CATEGORY_LAB = Code(V2_0074, "LAB", "Laboratory")
CATEGORY_RAD = Code(V2_0074, "RAD", "Radiology")
OBS_CATEGORY_LAB = Code(OBS_CATEGORY, "laboratory", "Laboratory")

ENCOUNTER_CLASS: dict[EncounterClass, Code] = {
    EncounterClass.INPATIENT: Code(V3_ACT_CODE, "IMP", "inpatient encounter"),
    EncounterClass.AMBULATORY: Code(V3_ACT_CODE, "AMB", "ambulatory"),
    EncounterClass.EMERGENCY: Code(V3_ACT_CODE, "EMER", "emergency"),
}

INTERPRETATION: dict[LabFlag, Code] = {
    LabFlag.HIGH: Code(V3_OBS_INTERP, "H", "High"),
    LabFlag.LOW: Code(V3_OBS_INTERP, "L", "Low"),
    LabFlag.NORMAL: Code(V3_OBS_INTERP, "N", "Normal"),
}

LAB_LOINC: dict[str, Code] = {
    "glucose": Code(LOINC, "2345-7", "Glucose [Mass/volume] in Serum or Plasma"),
    "hba1c": Code(LOINC, "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood"),
    "creatinine": Code(LOINC, "2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
    "egfr": Code(
        LOINC,
        "62238-1",
        "Glomerular filtration rate/1.73 sq M predicted by CKD-EPI creatinine formula",
    ),
    "potassium": Code(LOINC, "2823-3", "Potassium [Moles/volume] in Serum or Plasma"),
    "sodium": Code(LOINC, "2951-2", "Sodium [Moles/volume] in Serum or Plasma"),
    "wbc": Code(LOINC, "6690-2", "Leukocytes [#/volume] in Blood by Automated count"),
    "hemoglobin": Code(LOINC, "718-7", "Hemoglobin [Mass/volume] in Blood"),
    "ferritin": Code(LOINC, "2276-4", "Ferritin [Mass/volume] in Serum or Plasma"),
    "bnp": Code(LOINC, "30934-4", "Natriuretic peptide B [Mass/volume] in Serum or Plasma"),
    "crp": Code(LOINC, "1988-5", "C reactive protein [Mass/volume] in Serum or Plasma"),
    "tsh": Code(LOINC, "3016-3", "Thyrotropin [Units/volume] in Serum or Plasma"),
}

# canonical display unit -> UCUM code
UCUM_UNITS: dict[str, str] = {
    "mg/dL": "mg/dL",
    "%": "%",
    "mmol/L": "mmol/L",
    "10*3/uL": "10*3/uL",
    "g/dL": "g/dL",
    "ng/mL": "ng/mL",
    "pg/mL": "pg/mL",
    "mg/L": "mg/L",
    "mIU/L": "m[IU]/L",
    "mL/min/1.73m2": "mL/min/{1.73_m2}",
}


def gender_code(gender: Gender) -> str:
    return gender.value
