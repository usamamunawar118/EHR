"""Builders for tiny hand-made inputs used by edge-case tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_export(
    directory: Path, patients: list[dict[str, Any]], name: str = "export.json"
) -> Path:
    """Write a minimal JSON export (style A keys)."""
    path = directory / name
    path.write_text(json.dumps({"patients": patients}), encoding="utf-8")
    return path


def patient_a(
    mrn: str,
    *,
    family: str = "Tester",
    given: str = "Una",
    dob: str | None = "1980-01-02",
    sex: str = "F",
    encounters: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "MRN": mrn,
        "name": {"family": family, "given": given},
        "dob": dob,
        "sex": sex,
        "encounters": encounters or [],
        "documents": documents or [],
    }


def doc_a(
    doc_id: str,
    *,
    date_: str | None,
    text: str = "ASSESSMENT: stable. PLAN: continue current management.",
    type_: str = "Progress Note",
    encounter_id: str | None = None,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "type": type_,
        "date": date_,
        "encounter_id": encounter_id,
        "text": text,
    }


def enc_a(
    encounter_id: str, start: str, end: str | None = None, kind: str = "Outpatient"
) -> dict[str, Any]:
    return {
        "encounter_id": encounter_id,
        "type": kind,
        "start": start,
        "end": end or start,
        "reason": "visit",
    }


IMAGING_TEXT = (
    "EXAM: XR CHEST 2 VIEWS\nTECHNIQUE: PA and lateral.\n"
    "FINDINGS: Right lower lobe consolidation.\nIMPRESSION: Right lower lobe pneumonia."
)


def find(bundle: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    """All resources of ``resource_type`` inside a Bundle dict."""
    return [
        e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == resource_type
    ]
