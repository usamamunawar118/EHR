"""Dependency container: built once at start-up, injected into routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from ehr_media.config import Settings
from ehr_media.search.embedder import Embedder, create_embedder
from ehr_media.search.service import SearchService
from ehr_media.search.vector_store import ChromaVectorStore
from ehr_media.storage.bundle_repo import BundleRepository
from ehr_media.storage.db import Database
from ehr_media.storage.summary_cache import SummaryCache


@dataclass
class Container:
    settings: Settings
    repo: BundleRepository
    summaries: SummaryCache
    embedder: Embedder
    store: ChromaVectorStore
    search: SearchService
    llm_provider: str = "unknown"


def build_container(settings: Settings, embedder: Embedder | None = None) -> Container:
    """Create the long-lived singletons (the embedding model is loaded and warmed here once)."""
    db = Database(settings.db_path)
    repo, summaries = BundleRepository(db), SummaryCache(db)
    embedder = embedder or create_embedder(settings)
    store = ChromaVectorStore(settings.chroma_dir)
    search = SearchService(
        repo,
        summaries,
        embedder,
        store,
        top_k=settings.search_top_k,
        overfetch=settings.search_overfetch,
    )
    return Container(
        settings=settings,
        repo=repo,
        summaries=summaries,
        embedder=embedder,
        store=store,
        search=search,
        llm_provider=settings.resolved_llm_provider,
    )


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
