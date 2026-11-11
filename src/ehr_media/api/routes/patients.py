"""Patient detail, FHIR Bundle and individual resource endpoints."""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Response

from ehr_media.api.deps import ContainerDep
from ehr_media.api.schemas import ErrorBody, PatientDetail, PatientOut, ResourceOut

router = APIRouter(prefix="/patients", tags=["patients"])

_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorBody, "description": "Unknown patient or resource"}
}
FHIR_JSON = "application/fhir+json"


def _patient_out(row: Any, validation_status: str) -> PatientOut:
    return PatientOut(
        patient_id=row.patient_id,
        mrn=row.mrn,
        display_name=row.display_name,
        birth_date=date.fromisoformat(row.birth_date) if row.birth_date else None,
        gender=row.gender,
        data_quality_flags=list(row.flags),
        validation_status=validation_status,
    )


@router.get("/{patient_id}", response_model=PatientDetail, responses=_NOT_FOUND)
def patient_detail(patient_id: str, container: ContainerDep) -> PatientDetail:
    """Demographics, the full AI summary and the list of linked FHIR resources."""
    stored = container.repo.get_bundle(patient_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="patient not found")
    summary = container.summaries.latest_for(patient_id, stored.content_hash)
    resources = [
        ResourceOut(
            resource_id=r.resource_id,
            resource_type=r.resource_type,
            effective_date=date.fromisoformat(r.effective_date) if r.effective_date else None,
            display=r.display,
        )
        for r in container.repo.list_resources(patient_id)
    ]
    return PatientDetail(
        patient=_patient_out(stored.patient, stored.validation_status),
        summary=summary,
        resources=resources,
    )


@router.get(
    "/{patient_id}/bundle",
    responses={**_NOT_FOUND, 200: {"content": {FHIR_JSON: {}}}},
    response_class=Response,
)
def patient_bundle(
    patient_id: str,
    container: ContainerDep,
    download: Annotated[bool, Query(description="Send as a file attachment.")] = False,
) -> Response:
    """The patient's FHIR R4 Bundle as ``application/fhir+json``."""
    stored = container.repo.get_bundle(patient_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="patient not found")
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="bundle-{stored.patient.mrn}.json"'
    return Response(json.dumps(stored.bundle, indent=2), media_type=FHIR_JSON, headers=headers)


@router.get(
    "/{patient_id}/resources/{resource_id}",
    responses={**_NOT_FOUND, 200: {"content": {FHIR_JSON: {}}}},
    response_class=Response,
)
def patient_resource(patient_id: str, resource_id: str, container: ContainerDep) -> Response:
    """One FHIR resource as JSON (loaded lazily by the UI's "show JSON" toggle)."""
    resource = container.repo.get_resource(patient_id, resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail="resource not found")
    return Response(resource.json, media_type=FHIR_JSON)
