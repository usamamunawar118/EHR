"""Fixtures for FHIR tests: a small, fully-featured patient built through the real pipeline."""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ehr_media.fhir.builder import PatientBundle, build_patient_bundle
from ehr_media.ingestion.models import PatientRecord
from ehr_media.ingestion.pipeline import run_ingest
from tests.helpers import doc_a, enc_a, patient_a, write_export

IMAGING_TEXT = (
    "EXAM: XR CHEST 2 VIEWS\nTECHNIQUE: PA and lateral.\n"
    "FINDINGS: Right lower lobe consolidation.\nIMPRESSION: Right lower lobe pneumonia."
)


@pytest.fixture
def record(tmp_path: Path) -> PatientRecord:
    write_export(
        tmp_path,
        [
            patient_a(
                "00000042",
                family="Fixture",
                given="Fay",
                dob="1960-02-03",
                sex="F",
                encounters=[enc_a("E1", "2025-03-01", "2025-03-04", "Inpatient")],
                documents=[
                    doc_a("d1", date_="2025-03-01", encounter_id="E1", text="PLAN: antibiotics."),
                    doc_a(
                        "d2",
                        date_="2025-03-01",
                        encounter_id="E1",
                        type_="Radiology Report",
                        text=IMAGING_TEXT,
                    ),
                    doc_a(
                        "d3",
                        date_="2025-03-04",
                        encounter_id="E1",
                        type_="Discharge Summary",
                        text="DISCHARGE SUMMARY\nHOSPITAL COURSE: improved.",
                    ),
                ],
            )
        ],
    )
    (tmp_path / "labs.csv").write_text(
        "MRN,PatientName,DOB,CollectedDate,TestName,Result,Units,RefRange,Flag,EncounterID\n"
        '42,"FIXTURE, FAY",,2025-03-01,WBC,15.2,10*3/uL,4.5-11,H,E1\n'
        '42,"FIXTURE, FAY",,2025-03-01,Sodium,140,mmol/L,135-145,N,E1\n'
        '42,"FIXTURE, FAY",,2025-03-01,Mystery Marker,3.3,widgets,,,E1\n',
        encoding="utf-8",
    )
    result = run_ingest([tmp_path], reference_date=date(2026, 11, 11))
    assert len(result.patients) == 1
    return result.patients[0]


@pytest.fixture
def patient_bundle(record: PatientRecord) -> PatientBundle:
    return build_patient_bundle(record)


@pytest.fixture
def bundle(patient_bundle: PatientBundle) -> dict[str, Any]:
    """A fresh deep copy per test so tests can corrupt it freely."""
    return copy.deepcopy(patient_bundle.bundle)


def find(bundle: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    return [
        e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == resource_type
    ]
