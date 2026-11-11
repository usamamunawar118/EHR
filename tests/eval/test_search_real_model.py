"""Search quality and latency with the *real* sentence-transformer model.

Slow (first run downloads ~90 MB and loads torch), so excluded from the default run:

    pytest -m "slow or perf" -s

* relevance - paraphrased clinical queries must surface the right records (synonyms are exactly
  what the hashing stand-in cannot do)
* latency   - the brief requires < 2 s per query over >= 50 records
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ehr_media.api.app import create_app
from ehr_media.api.deps import Container, build_container
from ehr_media.config import Settings
from ehr_media.fhir.service import build_and_validate, store_bundles
from ehr_media.ingestion.models import IngestResult
from ehr_media.search.embedder import SentenceTransformerEmbedder
from ehr_media.search.indexer import Indexer
from ehr_media.summarization.llm import FakeSummaryLLM
from ehr_media.summarization.service import SummaryService

pytestmark = pytest.mark.slow

LATENCY_BUDGET_MS = 2000
MIN_RECORDS = 50


@pytest.fixture(scope="module")
def real_env(ingest_result: IngestResult, tmp_path_factory: pytest.TempPathFactory) -> Container:
    root = tmp_path_factory.mktemp("realenv")
    settings = Settings(
        data_dir=root / "data",
        frontend_dir=root / "none",
        embedding_backend="sentence-transformers",
        llm_provider="fake",
    )
    container = build_container(settings)  # loads + warms the real model once
    store_bundles(build_and_validate(ingest_result.patients), container.repo)
    SummaryService(FakeSummaryLLM(), container.summaries).summarize_all(
        list(container.repo.iter_bundles())
    )
    Indexer(
        container.repo,
        container.summaries,
        container.embedder,
        container.store,
        chunk_size=settings.chunk_words,
        chunk_overlap=settings.chunk_overlap_words,
    ).index_all()
    assert isinstance(container.embedder, SentenceTransformerEmbedder)
    return container


@pytest.fixture(scope="module")
def client(real_env: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container=real_env)) as c:
        yield c


RELEVANCE_CASES = [
    # (paraphrased clinician query, any of these words must appear in the top-3 results)
    ("chest x-ray pneumonia", ("pneumonia", "consolidation")),
    ("broken wrist after a fall", ("fracture", "wrist")),
    ("swollen legs and trouble breathing when lying flat", ("heart failure", "orthopnea", "flat")),
    ("high blood sugar that is not controlled", ("diabetes", "glucose", "hyperglycemia")),
    ("kidney problems with high potassium", ("kidney", "potassium", "renal")),
    ("irregular heartbeat, on a blood thinner", ("atrial fibrillation", "apixaban")),
    ("low iron and feeling tired", ("iron", "anemia", "ferritin")),
    ("appendix surgery", ("appendicitis", "appendectomy", "appendix")),
]


@pytest.mark.parametrize(("query", "keywords"), RELEVANCE_CASES)
def test_paraphrased_queries_find_the_right_records(
    real_env: Container, query: str, keywords: tuple[str, ...]
) -> None:
    hits = real_env.search.search(query).hits[:3]
    assert hits, "no results"
    haystack = " ".join(
        f"{h.matched_snippet} {h.summary_snippet or ''} {h.display}" for h in hits
    ).lower()
    assert any(k in haystack for k in keywords), (
        f"{query!r}: none of {keywords} in top-3\n"
        + "\n".join(h.matched_snippet[:90] for h in hits)
    )


def test_unrelated_queries_score_lower_than_relevant_ones(real_env: Container) -> None:
    relevant = real_env.search.search("chest x-ray pneumonia").hits[0].score
    unrelated = real_env.search.search("quarterly municipal budget for road repairs").hits[0].score
    assert relevant > unrelated + 0.2


@pytest.mark.perf
def test_latency_requirement_over_at_least_50_records(
    real_env: Container, client: TestClient
) -> None:
    assert real_env.repo.count()["patients"] >= MIN_RECORDS
    assert real_env.store.count() >= MIN_RECORDS

    queries = [q for q, _ in RELEVANCE_CASES] * 4  # 32 queries
    service_ms: list[float] = []
    http_ms: list[float] = []
    for q in queries:
        t = time.perf_counter()
        real_env.search.search(q)
        service_ms.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        res = client.post("/search", json={"query": q})
        http_ms.append((time.perf_counter() - t) * 1000)
        assert res.status_code == 200

    p95 = lambda xs: sorted(xs)[int(len(xs) * 0.95) - 1]  # noqa: E731
    print(
        f"\nservice: median {statistics.median(service_ms):.0f} ms, p95 {p95(service_ms):.0f} ms | "
        f"http: median {statistics.median(http_ms):.0f} ms, p95 {p95(http_ms):.0f} ms, "
        f"max {max(http_ms):.0f} ms"
    )
    assert max(http_ms) < LATENCY_BUDGET_MS
    assert p95(http_ms) < LATENCY_BUDGET_MS / 4  # comfortable headroom, not just "under 2 s"


@pytest.mark.perf
def test_filtered_queries_stay_fast(client: TestClient) -> None:
    started = time.perf_counter()
    res = client.post(
        "/search",
        params={
            "resource_type": "DiagnosticReport",
            "date_from": "2024-06-01",
            "date_to": "2024-12-31",
        },
        json={"query": "abnormal laboratory results"},
    )
    assert res.status_code == 200
    assert (time.perf_counter() - started) * 1000 < LATENCY_BUDGET_MS
