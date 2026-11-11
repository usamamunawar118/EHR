"""POST /search - natural-language search with resource-type and date filters."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from ehr_media.api.deps import ContainerDep
from ehr_media.api.schemas import (
    RESOURCE_TYPE_LABELS,
    ErrorBody,
    ResourceType,
    ResourceTypeOption,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from ehr_media.search.service import SearchFilters

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search across patient records and AI summaries",
    responses={422: {"model": ErrorBody, "description": "Invalid query or filters"}},
)
def search(
    body: SearchRequest,
    container: ContainerDep,
    resource_type: Annotated[
        ResourceType | None, Query(description="Only return this FHIR resource type.")
    ] = None,
    date_from: Annotated[
        date | None, Query(description="Earliest record date (inclusive, YYYY-MM-DD).")
    ] = None,
    date_to: Annotated[
        date | None, Query(description="Latest record date (inclusive, YYYY-MM-DD).")
    ] = None,
) -> SearchResponse:
    """Return the top-ranked record matches (default 5) with relevance scores.

    Filters are query parameters (``?resource_type=...&date_from=...&date_to=...``); the
    free-text query is the JSON body.
    """
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not be after date_to")

    result = container.search.search(
        body.query.strip(),
        SearchFilters(
            resource_type=resource_type.value if resource_type else None,
            date_from=date_from,
            date_to=date_to,
        ),
    )
    items = [
        SearchResultItem(
            patient_id=h.patient_id,
            mrn=h.mrn,
            patient_name=h.patient_name,
            kind=h.kind,
            resource_type=h.resource_type,
            resource_id=h.resource_id,
            display=h.display,
            record_date=date.fromisoformat(h.record_date) if h.record_date else None,
            score=h.score,
            matched_snippet=h.matched_snippet,
            summary_snippet=h.summary_snippet,
        )
        for h in result.hits
    ]
    return SearchResponse(
        query=result.query, total=len(items), took_ms=result.took_ms, results=items
    )


@router.get("/meta/resource-types", response_model=list[ResourceTypeOption])
def resource_types() -> list[ResourceTypeOption]:
    """Values accepted by the ``resource_type`` filter (drives the UI dropdown)."""
    return [ResourceTypeOption(value=v, label=RESOURCE_TYPE_LABELS[v]) for v in ResourceType]
