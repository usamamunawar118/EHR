"""Thin wrapper over a Chroma collection.

Isolating Chroma here keeps the rest of the code independent of its API (and of its metadata
quirks): metadata filters support range operators on **numbers only**, so dates are stored as
``date_int = YYYYMMDD`` integers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import chromadb

COLLECTION = "ehr_records"
_PAGE = 1000


def date_to_int(d: date | str) -> int:
    """``date`` or ISO string -> 20250304."""
    iso = d.isoformat() if isinstance(d, date) else d
    return int(iso[:10].replace("-", ""))


def int_to_iso(value: int) -> str:
    s = f"{value:08d}"
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


@dataclass(frozen=True)
class VectorRecord:
    id: str
    embedding: list[float]
    document: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class VectorHit:
    id: str
    distance: float
    metadata: dict[str, Any]
    document: str

    @property
    def score(self) -> float:
        """Cosine similarity in [0, 1] (Chroma returns cosine *distance*)."""
        return round(max(0.0, min(1.0, 1.0 - self.distance)), 4)


def build_where(
    resource_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> dict[str, Any] | None:
    """Metadata filter for Chroma, or ``None`` when no filter is requested."""
    clauses: list[dict[str, Any]] = []
    if resource_type:
        clauses.append({"resource_type": resource_type})
    if date_from:
        clauses.append({"date_int": {"$gte": date_to_int(date_from)}})
    if date_to:
        clauses.append({"date_int": {"$lte": date_to_int(date_to)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaVectorStore:
    def __init__(self, path: Path) -> None:
        """Persistent store at ``path``.

        (No in-memory mode on purpose: Chroma's ephemeral clients can share process-wide state,
        which makes tests leak into each other. A temp directory is just as fast.)
        """
        path.mkdir(parents=True, exist_ok=True)
        self._client: Any = chromadb.PersistentClient(
            path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._collection: Any = self._open()

    def _open(self) -> Any:
        return self._client.get_or_create_collection(
            COLLECTION, configuration={"hnsw": {"space": "cosine"}}
        )

    def count(self) -> int:
        return int(self._collection.count())

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.embedding for r in records],
            documents=[r.document for r in records],
            metadatas=[dict(r.metadata) for r in records],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._collection.delete(ids=list(ids))

    def existing_hashes(self) -> dict[str, str]:
        """``{id: content_hash}`` of everything indexed (drives incremental re-indexing)."""
        out: dict[str, str] = {}
        offset = 0
        while True:
            page = self._collection.get(include=["metadatas"], limit=_PAGE, offset=offset)
            ids = page["ids"]
            if not ids:
                return out
            for id_, meta in zip(ids, page["metadatas"], strict=True):
                out[id_] = str((meta or {}).get("content_hash", ""))
            offset += len(ids)

    def query(
        self, embedding: Sequence[float], n_results: int, where: Mapping[str, Any] | None = None
    ) -> list[VectorHit]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(n_results, total),
            where=dict(where) if where else None,
            include=["metadatas", "documents", "distances"],
        )
        return [
            VectorHit(id=i, distance=float(d), metadata=dict(m or {}), document=doc or "")
            for i, d, m, doc in zip(
                result["ids"][0],
                result["distances"][0],
                result["metadatas"][0],
                result["documents"][0],
                strict=True,
            )
        ]

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._collection = self._open()
