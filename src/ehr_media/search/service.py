"""Semantic search over indexed records and AI summaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Literal

from ehr_media.search.embedder import Embedder
from ehr_media.search.vector_store import (
    ChromaVectorStore,
    VectorHit,
    build_where,
    int_to_iso,
)
from ehr_media.storage.bundle_repo import BundleRepository
from ehr_media.storage.summary_cache import SummaryCache

SNIPPET_CHARS = 280


@dataclass(frozen=True)
class SearchFilters:
    resource_type: str | None = None
    date_from: date | None = None
    date_to: date | None = None


@dataclass(frozen=True)
class SearchHit:
    patient_id: str
    mrn: str
    patient_name: str
    kind: Literal["record", "summary"]
    resource_type: str
    resource_id: str
    display: str
    record_date: str | None
    score: float
    matched_snippet: str
    summary_snippet: str | None


@dataclass(frozen=True)
class SearchResult:
    query: str
    hits: tuple[SearchHit, ...]
    took_ms: float


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + "…"


class SearchService:
    def __init__(
        self,
        repo: BundleRepository,
        summaries: SummaryCache,
        embedder: Embedder,
        store: ChromaVectorStore,
        *,
        top_k: int = 5,
        overfetch: int = 6,
    ) -> None:
        self._repo = repo
        self._summaries = summaries
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._overfetch = overfetch

    def search(self, query: str, filters: SearchFilters | None = None) -> SearchResult:
        started = time.perf_counter()
        filters = filters or SearchFilters()
        where = build_where(filters.resource_type, filters.date_from, filters.date_to)
        vector = self._embedder.embed([query])[0]
        # Over-fetch: several chunks of one document (and the DocumentReference/DiagnosticReport
        # pair for imaging) collapse into one result, so ask for more than top_k.
        raw = self._store.query(vector, self._top_k * self._overfetch, where)

        best: dict[tuple[str, str], VectorHit] = {}
        for hit in raw:  # already ordered best-first
            key = (str(hit.metadata["patient_id"]), str(hit.metadata["doc_key"]))
            best.setdefault(key, hit)
        top = list(best.values())[: self._top_k]

        results = tuple(self._hydrate(h) for h in top)
        return SearchResult(
            query=query, hits=results, took_ms=round((time.perf_counter() - started) * 1000, 1)
        )

    def _hydrate(self, hit: VectorHit) -> SearchHit:
        meta = hit.metadata
        patient_id = str(meta["patient_id"])
        patient = self._repo.get_patient(patient_id)
        content_hash = self._repo.content_hash(patient_id)
        summary = self._summaries.latest_for(patient_id, content_hash) if content_hash else None
        date_int = int(meta.get("date_int", 0))
        return SearchHit(
            patient_id=patient_id,
            mrn=patient.mrn if patient else "unknown",
            patient_name=patient.display_name if patient else "Unknown patient",
            kind="summary" if meta["kind"] == "summary" else "record",
            resource_type=str(meta["resource_type"]),
            resource_id=str(meta["resource_id"]),
            display=str(meta.get("display", "")),
            record_date=int_to_iso(date_int) if date_int else None,
            score=hit.score,
            matched_snippet=_snippet(hit.document),
            summary_snippet=_snippet(summary.narrative()) if summary else None,
        )
