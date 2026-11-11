"""Chunking, embedders and the Chroma wrapper."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pytest

from ehr_media.search.chunking import chunk_words
from ehr_media.search.embedder import HashingEmbedder
from ehr_media.search.vector_store import (
    ChromaVectorStore,
    VectorHit,
    VectorRecord,
    build_where,
    date_to_int,
    int_to_iso,
)


class TestChunking:
    def test_short_text_is_a_single_chunk(self) -> None:
        assert chunk_words("one two three", size=10, overlap=2) == ["one two three"]

    def test_empty_text_has_no_chunks(self) -> None:
        assert chunk_words("   \n ", size=10, overlap=2) == []

    def test_windows_overlap_and_cover_everything(self) -> None:
        words = [f"w{i}" for i in range(95)]
        chunks = chunk_words(" ".join(words), size=40, overlap=10)
        assert all(len(c.split()) <= 40 for c in chunks)
        assert chunks[0].split()[-10:] == chunks[1].split()[:10]  # the overlap
        covered = {w for c in chunks for w in c.split()}
        assert covered == set(words)

    def test_last_window_is_not_duplicated(self) -> None:
        chunks = chunk_words(" ".join(f"w{i}" for i in range(40)), size=40, overlap=10)
        assert len(chunks) == 1

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
    def test_invalid_parameters(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError, match=r"positive|overlap"):
            chunk_words("a b c", size=size, overlap=overlap)


class TestHashingEmbedder:
    def test_is_deterministic_and_normalised(self) -> None:
        emb = HashingEmbedder()
        (a,), (b,) = emb.embed(["Chest X-ray: pneumonia"]), emb.embed(["Chest X-ray: pneumonia"])
        assert a == b
        assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
        assert len(a) == emb.dimension

    def test_shared_words_mean_higher_similarity(self) -> None:
        emb = HashingEmbedder()
        q, near, far = emb.embed(
            ["pneumonia chest", "chest pneumonia consolidation", "wrist fracture"]
        )
        dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
        assert dot(q, near) > dot(q, far)

    def test_empty_text_is_a_zero_vector_not_a_crash(self) -> None:
        (vec,) = HashingEmbedder().embed([""])
        assert not any(vec)


class TestWhereBuilder:
    def test_no_filters(self) -> None:
        assert build_where() is None

    def test_single_clause_is_not_wrapped(self) -> None:
        assert build_where(resource_type="Patient") == {"resource_type": "Patient"}
        assert build_where(date_from=date(2025, 1, 2)) == {"date_int": {"$gte": 20250102}}

    def test_combined_clauses(self) -> None:
        where = build_where("DiagnosticReport", date(2025, 1, 1), date(2025, 12, 31))
        assert where == {
            "$and": [
                {"resource_type": "DiagnosticReport"},
                {"date_int": {"$gte": 20250101}},
                {"date_int": {"$lte": 20251231}},
            ]
        }

    def test_date_int_round_trip(self) -> None:
        assert date_to_int(date(2025, 3, 4)) == 20250304
        assert date_to_int("2025-03-04") == 20250304
        assert int_to_iso(20250304) == "2025-03-04"

    def test_score_is_clamped_similarity(self) -> None:
        assert VectorHit("a", 0.25, {}, "").score == 0.75
        assert VectorHit("a", 1.7, {}, "").score == 0.0
        assert VectorHit("a", -0.1, {}, "").score == 1.0


class TestChromaStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> ChromaVectorStore:
        store = ChromaVectorStore(tmp_path / "chroma")
        store.upsert(
            [
                VectorRecord(
                    "a", [1.0, 0.0, 0.0], "alpha", {"resource_type": "X", "date_int": 20250101}
                ),
                VectorRecord(
                    "b", [0.0, 1.0, 0.0], "beta", {"resource_type": "Y", "date_int": 20250301}
                ),
                VectorRecord(
                    "c", [0.7, 0.7, 0.0], "gamma", {"resource_type": "X", "date_int": 20250601}
                ),
            ]
        )
        return store

    def test_nearest_first_with_cosine_scores(self, store: ChromaVectorStore) -> None:
        hits = store.query([1.0, 0.0, 0.0], n_results=3)
        assert [h.id for h in hits] == ["a", "c", "b"]
        assert hits[0].score == 1.0
        assert hits[2].score == 0.0  # orthogonal

    def test_resource_type_and_date_range_filters(self, store: ChromaVectorStore) -> None:
        where = build_where("X", date(2025, 3, 1), date(2025, 12, 31))
        assert [h.id for h in store.query([1.0, 0.0, 0.0], 3, where)] == ["c"]
        assert [
            h.id for h in store.query([1.0, 0.0, 0.0], 3, build_where(date_from=date(2025, 3, 1)))
        ] == ["c", "b"]

    def test_asking_for_more_than_exists_is_fine(self, store: ChromaVectorStore) -> None:
        assert len(store.query([1.0, 0.0, 0.0], n_results=500)) == 3

    def test_upsert_is_idempotent_and_delete_works(self, store: ChromaVectorStore) -> None:
        store.upsert(
            [VectorRecord("a", [1.0, 0.0, 0.0], "alpha", {"resource_type": "X", "date_int": 1})]
        )
        assert store.count() == 3
        store.delete(["a", "missing"])
        assert store.count() == 2

    def test_empty_store_returns_no_hits(self, tmp_path: Path) -> None:
        assert ChromaVectorStore(tmp_path / "empty").query([1.0, 0.0], 5) == []

    def test_existing_hashes(self, tmp_path: Path) -> None:
        store = ChromaVectorStore(tmp_path / "h")
        store.upsert([VectorRecord("a", [1.0, 0.0], "x", {"content_hash": "h1"})])
        assert store.existing_hashes() == {"a": "h1"}

    def test_reset_empties_the_collection(self, store: ChromaVectorStore) -> None:
        store.reset()
        assert store.count() == 0

    def test_data_survives_reopening(self, tmp_path: Path) -> None:
        first = ChromaVectorStore(tmp_path / "persist")
        first.upsert([VectorRecord("a", [1.0, 0.0], "x", {"k": 1})])
        assert ChromaVectorStore(tmp_path / "persist").count() == 1
