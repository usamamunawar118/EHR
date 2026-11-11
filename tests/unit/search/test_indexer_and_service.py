"""Indexer (incremental, idempotent) and SearchService (ranking, filters, de-duplication)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from ehr_media.api.deps import Container
from ehr_media.fhir.builder import PatientBundle
from ehr_media.fhir.validation import validate_bundle
from ehr_media.search.embedder import HashingEmbedder
from ehr_media.search.indexer import Indexer
from ehr_media.search.service import SearchFilters, SearchService
from ehr_media.search.vector_store import ChromaVectorStore, VectorRecord
from ehr_media.storage.bundle_repo import BundleRepository
from ehr_media.storage.db import Database
from ehr_media.storage.summary_cache import SummaryCache
from ehr_media.summarization.llm import FakeSummaryLLM
from ehr_media.summarization.service import SummaryService


@dataclass
class Mini:
    db: Database
    repo: BundleRepository
    cache: SummaryCache
    store: ChromaVectorStore
    embedder: HashingEmbedder
    indexer: Indexer
    search: SearchService
    bundle: PatientBundle


@pytest.fixture
def mini(tmp_path: Path, patient_bundle: PatientBundle) -> Mini:
    """One patient (3 documents, 1 imaging report, 3 labs) stored + summarised + indexable."""
    db = Database(tmp_path / "m.db")
    repo, cache = BundleRepository(db), SummaryCache(db)
    repo.save(patient_bundle, validate_bundle(patient_bundle.bundle))
    SummaryService(FakeSummaryLLM(), cache).summarize_all(list(repo.iter_bundles()))
    embedder, store = HashingEmbedder(), ChromaVectorStore(tmp_path / "chroma")
    return Mini(
        db,
        repo,
        cache,
        store,
        embedder,
        Indexer(repo, cache, embedder, store, chunk_size=150, chunk_overlap=25),
        SearchService(repo, cache, embedder, store, top_k=5),
        patient_bundle,
    )


class TestIndexer:
    def test_indexes_documents_reports_and_the_summary(self, mini: Mini) -> None:
        stats = mini.indexer.index_all()
        # 3 DocumentReferences + imaging DiagnosticReport + lab DiagnosticReport + 1 summary
        assert stats.desired == 6
        assert stats.embedded == 6
        assert stats.total_in_store == 6

    def test_rerun_embeds_nothing(self, mini: Mini) -> None:
        mini.indexer.index_all()
        again = mini.indexer.index_all()
        assert (again.embedded, again.unchanged, again.deleted) == (0, 6, 0)

    def test_new_summary_text_reembeds_only_the_summary(self, mini: Mini) -> None:
        mini.indexer.index_all()
        SummaryService(FakeSummaryLLM(), mini.cache).summarize_all(
            list(mini.repo.iter_bundles()), force=True
        )  # same content -> same hash -> still nothing to do
        assert mini.indexer.index_all().embedded == 0

        summary = mini.cache.latest_for(mini.bundle.patient_id, mini.bundle.content_hash)
        assert summary is not None
        summary.chief_concern = "Entirely different concern"
        mini.cache.put(summary)
        stats = mini.indexer.index_all()
        assert stats.embedded == 1

    def test_stale_entries_are_deleted(self, mini: Mini) -> None:
        mini.indexer.index_all()
        mini.store.upsert([VectorRecord("orphan", [0.0] * 512, "gone", {"content_hash": "x"})])
        assert mini.store.count() == 7
        stats = mini.indexer.index_all()
        assert stats.deleted == 1
        assert mini.store.count() == 6

    def test_patient_without_a_summary_still_indexes_its_records(self, mini: Mini) -> None:
        with mini.db.connection() as conn:
            conn.execute("DELETE FROM summary_cache")
        assert mini.indexer.index_all().desired == 5


class TestSearchService:
    @pytest.fixture(autouse=True)
    def _indexed(self, mini: Mini) -> None:
        mini.indexer.index_all()

    def test_returns_ranked_hits_with_scores_and_patient_fields(self, mini: Mini) -> None:
        result = mini.search.search("right lower lobe pneumonia chest x-ray")
        assert 1 <= len(result.hits) <= 5
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)
        top = result.hits[0]
        assert top.patient_name == "Fay Fixture"
        assert top.mrn == "00000042"
        assert 0 < top.score <= 1
        assert top.summary_snippet
        assert result.took_ms >= 0

    def test_imaging_documentreference_and_diagnosticreport_collapse_into_one_hit(
        self, mini: Mini
    ) -> None:
        hits = mini.search.search("XR CHEST 2 VIEWS right lower lobe consolidation").hits
        imaging = [h for h in hits if "XR CHEST" in h.matched_snippet]
        assert len(imaging) == 1

    def test_resource_type_filter(self, mini: Mini) -> None:
        reports = mini.search.search("pneumonia", SearchFilters(resource_type="DiagnosticReport"))
        assert reports.hits
        assert {h.resource_type for h in reports.hits} == {"DiagnosticReport"}

        summaries = mini.search.search("pneumonia", SearchFilters(resource_type="Patient"))
        assert {h.kind for h in summaries.hits} == {"summary"}

    def test_documents_are_not_returned_when_filtering_to_summaries(self, mini: Mini) -> None:
        hits = mini.search.search("pneumonia", SearchFilters(resource_type="Patient")).hits
        assert all(h.resource_type == "Patient" for h in hits)

    def test_date_range_filter_is_inclusive(self, mini: Mini) -> None:
        inside = mini.search.search("antibiotics", SearchFilters(date_from=date(2025, 3, 4)))
        assert {h.record_date for h in inside.hits} >= {"2025-03-04"}
        assert all(h.record_date and h.record_date >= "2025-03-04" for h in inside.hits)

        none = mini.search.search("antibiotics", SearchFilters(date_to=date(2025, 2, 28)))
        assert none.hits == ()

    def test_combined_filters(self, mini: Mini) -> None:
        res = mini.search.search(
            "pneumonia",
            SearchFilters("DocumentReference", date(2025, 3, 1), date(2025, 3, 1)),
        )
        assert res.hits
        assert all(
            h.resource_type == "DocumentReference" and h.record_date == "2025-03-01"
            for h in res.hits
        )

    def test_top_k_is_respected(self, tmp_path: Path, mini: Mini) -> None:
        limited = SearchService(mini.repo, mini.cache, mini.embedder, mini.store, top_k=2)
        assert len(limited.search("pneumonia").hits) == 2

    def test_empty_index_returns_no_results(self, tmp_path: Path, mini: Mini) -> None:
        empty = SearchService(
            mini.repo, mini.cache, mini.embedder, ChromaVectorStore(tmp_path / "none")
        )
        assert empty.search("anything").hits == ()


class TestFullDataset:
    """The hashing embedder is not semantic, but shared vocabulary must still rank correctly."""

    @pytest.mark.parametrize(
        ("query", "expected_word"),
        [
            ("wrist fracture splint", "wrist"),
            ("heart failure ejection fraction", "ejection"),
            ("iron deficiency anemia ferritin", "ferritin"),
        ],
    )
    def test_relevant_documents_rank_first(
        self, search_env: Container, query: str, expected_word: str
    ) -> None:
        result = search_env.search.search(query)
        assert len(result.hits) == 5
        assert expected_word in result.hits[0].matched_snippet.lower()

    def test_each_hit_is_a_distinct_document(self, search_env: Container) -> None:
        hits = search_env.search.search("pneumonia chest x-ray").hits
        keys = [(h.patient_id, h.display, h.record_date, h.kind) for h in hits]
        assert len(keys) == len(set(keys))

    def test_dataset_is_large_enough_for_the_performance_requirement(
        self, search_env: Container
    ) -> None:
        assert search_env.repo.count()["patients"] >= 50
        assert search_env.store.count() >= 50
