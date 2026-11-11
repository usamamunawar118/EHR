"""Liveness / readiness."""

from __future__ import annotations

from fastapi import APIRouter

from ehr_media.api.deps import ContainerDep
from ehr_media.api.schemas import Health

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=Health)
def health(container: ContainerDep) -> Health:
    return Health(
        patients=container.repo.count()["patients"],
        indexed_entries=container.store.count(),
        embedding_model=container.embedder.name,
        llm_provider=container.llm_provider,
    )
