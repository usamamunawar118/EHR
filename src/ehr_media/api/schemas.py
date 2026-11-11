"""Public API contract (request/response models)."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ehr_media.summarization.models import ClinicalSummary


class ResourceType(StrEnum):
    """Filterable resource types. ``Patient`` entries are the AI summaries."""

    DOCUMENT_REFERENCE = "DocumentReference"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    PATIENT = "Patient"


RESOURCE_TYPE_LABELS: dict[ResourceType, str] = {
    ResourceType.DOCUMENT_REFERENCE: "Documents (notes, discharge, imaging)",
    ResourceType.DIAGNOSTIC_REPORT: "Diagnostic reports (labs, imaging)",
    ResourceType.PATIENT: "AI patient summaries",
}


class SearchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"query": "chest x-ray pneumonia"}})

    query: str = Field(min_length=1, max_length=500, description="Free-text clinical query.")


class SearchResultItem(BaseModel):
    patient_id: str
    mrn: str
    patient_name: str
    kind: Literal["record", "summary"]
    resource_type: str
    resource_id: str
    display: str = Field(description="Human label, e.g. 'Imaging report: XR CHEST 2 VIEWS'.")
    record_date: date | None
    score: float = Field(ge=0, le=1, description="Cosine similarity, 1 = identical meaning.")
    matched_snippet: str
    summary_snippet: str | None = None


class SearchResponse(BaseModel):
    query: str
    total: int
    took_ms: float
    results: list[SearchResultItem]


class PatientOut(BaseModel):
    patient_id: str
    mrn: str
    display_name: str
    birth_date: date | None
    gender: str
    data_quality_flags: list[str]
    validation_status: str


class ResourceOut(BaseModel):
    resource_id: str
    resource_type: str
    effective_date: date | None
    display: str


class PatientDetail(BaseModel):
    patient: PatientOut
    summary: ClinicalSummary | None
    resources: list[ResourceOut]


class ResourceTypeOption(BaseModel):
    value: ResourceType
    label: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    patients: int
    indexed_entries: int
    embedding_model: str
    llm_provider: str


class ErrorBody(BaseModel):
    detail: str | list[dict[str, Any]]
