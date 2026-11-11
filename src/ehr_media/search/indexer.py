"""Build / refresh the vector index from stored FHIR resources and AI summaries.

Two kinds of entries share one collection:

* ``record``  - a chunk of a DocumentReference / DiagnosticReport text
* ``summary`` - one per patient: the AI summary, tagged ``resource_type="Patient"`` (it summarises
  the Patient compartment) and dated by the patient's most recent record, so the resource-type and
  date filters apply uniformly to both kinds.

Indexing is incremental and idempotent: unchanged entries (same content hash) are not
re-embedded, entries that no longer exist are deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from ehr_media.search.chunking import chunk_words
from ehr_media.search.embedder import Embedder
from ehr_media.search.vector_store import ChromaVectorStore, VectorRecord, date_to_int
from ehr_media.storage.bundle_repo import BundleRepository
from ehr_media.storage.summary_cache import SummaryCache

logger = logging.getLogger(__name__)

INDEXED_RESOURCE_TYPES = ("DocumentReference", "DiagnosticReport")
SUMMARY_RESOURCE_TYPE = "Patient"
MISSING_DATE = 0  # sorts before every real date; only used if a resource has no date


@dataclass(frozen=True)
class IndexStats:
    desired: int
    embedded: int
    unchanged: int
    deleted: int
    total_in_store: int


@dataclass(frozen=True)
class _Pending:
    id: str
    document: str  # what is stored and shown
    embed_text: str  # what is embedded (display label prefix helps type-oriented queries)
    metadata: dict[str, str | int]


def _hash(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


class Indexer:
    def __init__(
        self,
        repo: BundleRepository,
        summaries: SummaryCache,
        embedder: Embedder,
        store: ChromaVectorStore,
        *,
        chunk_size: int = 150,
        chunk_overlap: int = 25,
    ) -> None:
        self._repo = repo
        self._summaries = summaries
        self._embedder = embedder
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _desired(self) -> dict[str, _Pending]:
        desired: dict[str, _Pending] = {}
        for patient in self._repo.list_patients():
            pid = patient.patient_id
            for res in self._repo.list_resources(pid):
                if res.resource_type not in INDEXED_RESOURCE_TYPES or not res.text:
                    continue
                date_int = date_to_int(res.effective_date) if res.effective_date else MISSING_DATE
                chunks = chunk_words(res.text, self._chunk_size, self._chunk_overlap)
                for i, chunk in enumerate(chunks):
                    meta: dict[str, str | int] = {
                        "kind": "record",
                        "patient_id": pid,
                        "resource_id": res.resource_id,
                        "resource_type": res.resource_type,
                        "doc_key": res.doc_key or res.resource_id,
                        "display": res.display,
                        "date_int": date_int,
                        "chunk": i,
                    }
                    embed_text = f"{res.display}: {chunk}"
                    meta["content_hash"] = _hash(embed_text, meta)
                    desired[f"{res.resource_id}:{i}"] = _Pending(
                        f"{res.resource_id}:{i}", chunk, embed_text, meta
                    )

            content_hash = self._repo.content_hash(pid)
            summary = self._summaries.latest_for(pid, content_hash) if content_hash else None
            if summary is not None:
                latest = self._repo.latest_effective_date(pid)
                text = summary.narrative()
                meta = {
                    "kind": "summary",
                    "patient_id": pid,
                    "resource_id": f"summary:{pid}",
                    "resource_type": SUMMARY_RESOURCE_TYPE,
                    "doc_key": f"summary:{pid}",
                    "display": "AI summary",
                    "date_int": date_to_int(latest) if latest else MISSING_DATE,
                    "chunk": 0,
                }
                meta["content_hash"] = _hash(text, meta)
                desired[f"summary:{pid}"] = _Pending(
                    f"summary:{pid}", text, f"AI summary: {text}", meta
                )
        return desired

    def index_all(self) -> IndexStats:
        desired = self._desired()
        existing = self._store.existing_hashes()

        changed = [p for p in desired.values() if existing.get(p.id) != p.metadata["content_hash"]]
        stale = sorted(existing.keys() - desired.keys())

        if changed:
            vectors = self._embedder.embed([p.embed_text for p in changed])
            self._store.upsert(
                [
                    VectorRecord(id=p.id, embedding=v, document=p.document, metadata=p.metadata)
                    for p, v in zip(changed, vectors, strict=True)
                ]
            )
        self._store.delete(stale)

        stats = IndexStats(
            desired=len(desired),
            embedded=len(changed),
            unchanged=len(desired) - len(changed),
            deleted=len(stale),
            total_in_store=self._store.count(),
        )
        logger.info("index: %s", stats)
        return stats
