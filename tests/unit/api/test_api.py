"""HTTP contract tests (FastAPI TestClient) over the full synthetic dataset."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ehr_media.api.app import create_app
from ehr_media.api.deps import Container


@pytest.fixture(scope="module")
def client(search_env: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container=search_env)) as c:
        yield c


def post_search(client: TestClient, query: str = "pneumonia", **params: str) -> object:
    return client.post("/search", params=params, json={"query": query})


class TestSearch:
    def test_returns_up_to_five_ranked_results_with_the_card_fields(
        self, client: TestClient
    ) -> None:
        res = client.post("/search", json={"query": "chest x-ray pneumonia"})
        assert res.status_code == 200
        body = res.json()
        assert body["query"] == "chest x-ray pneumonia"
        assert body["total"] == len(body["results"]) == 5
        assert body["took_ms"] >= 0
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)
        first = body["results"][0]
        assert set(first) >= {
            "patient_id",
            "mrn",
            "patient_name",
            "kind",
            "resource_type",
            "resource_id",
            "display",
            "record_date",
            "score",
            "matched_snippet",
            "summary_snippet",
        }
        assert 0 <= first["score"] <= 1

    def test_resource_type_filter_is_a_query_parameter(self, client: TestClient) -> None:
        res = client.post(
            "/search", params={"resource_type": "DiagnosticReport"}, json={"query": "glucose"}
        )
        assert res.status_code == 200
        assert {r["resource_type"] for r in res.json()["results"]} == {"DiagnosticReport"}

    def test_summaries_are_searchable_via_the_patient_type(self, client: TestClient) -> None:
        res = client.post("/search", params={"resource_type": "Patient"}, json={"query": "kidney"})
        results = res.json()["results"]
        assert results
        assert {r["kind"] for r in results} == {"summary"}

    def test_date_range_filter(self, client: TestClient) -> None:
        res = client.post(
            "/search",
            params={"date_from": "2024-06-01", "date_to": "2024-06-30"},
            json={"query": "fever cough"},
        )
        assert res.status_code == 200
        dates = [r["record_date"] for r in res.json()["results"]]
        assert dates
        assert all("2024-06-01" <= d <= "2024-06-30" for d in dates)

    def test_no_matches_is_an_empty_list_not_an_error(self, client: TestClient) -> None:
        res = client.post(
            "/search",
            params={"date_from": "1990-01-01", "date_to": "1990-01-02"},
            json={"query": "pneumonia"},
        )
        assert res.status_code == 200
        assert res.json() == {
            "query": "pneumonia",
            "total": 0,
            "took_ms": res.json()["took_ms"],
            "results": [],
        }

    @pytest.mark.parametrize(
        ("params", "body"),
        [
            ({}, {"query": ""}),
            ({}, {"query": "x" * 501}),
            ({}, {}),
            ({}, {"nope": "x"}),
            ({"resource_type": "Banana"}, {"query": "ok"}),
            ({"date_from": "not-a-date"}, {"query": "ok"}),
            ({"date_from": "2025-02-01", "date_to": "2025-01-01"}, {"query": "ok"}),
        ],
    )
    def test_invalid_requests_are_rejected_with_422(
        self, client: TestClient, params: dict[str, str], body: dict[str, str]
    ) -> None:
        res = client.post("/search", params=params, json=body)
        assert res.status_code == 422
        assert "detail" in res.json()

    def test_whitespace_only_query_does_not_crash(self, client: TestClient) -> None:
        assert client.post("/search", json={"query": "   "}).status_code in (200, 422)

    def test_get_is_not_allowed(self, client: TestClient) -> None:
        assert client.get("/search").status_code == 405

    def test_unicode_and_injection_like_queries_are_just_text(self, client: TestClient) -> None:
        for q in ["'; DROP TABLE patients; --", "<script>alert(1)</script>", "néphrologie 腎臓"]:
            assert client.post("/search", json={"query": q}).status_code == 200
        assert client.get("/health").json()["patients"] == 60  # nothing was harmed


class TestPatients:
    @pytest.fixture(scope="class")
    def patient_id(self, client: TestClient) -> str:
        hit = client.post("/search", json={"query": "pneumonia"}).json()["results"][0]
        return str(hit["patient_id"])

    def test_detail_has_summary_with_disclaimer_and_linked_resources(
        self, client: TestClient, patient_id: str
    ) -> None:
        res = client.get(f"/patients/{patient_id}")
        assert res.status_code == 200
        body = res.json()
        assert body["patient"]["patient_id"] == patient_id
        assert body["patient"]["validation_status"] == "valid"
        summary = body["summary"]
        assert "not a clinical decision" in summary["disclaimer"]
        assert summary["confidence"] in {"low", "medium", "high"}
        assert summary["word_count"] <= 200
        types = {r["resource_type"] for r in body["resources"]}
        assert {"Patient", "Encounter", "DocumentReference"} <= types

    def test_unknown_patient_is_404(self, client: TestClient) -> None:
        assert client.get("/patients/does-not-exist").status_code == 404
        assert client.get("/patients/does-not-exist/bundle").status_code == 404

    def test_bundle_is_served_as_fhir_json(self, client: TestClient, patient_id: str) -> None:
        res = client.get(f"/patients/{patient_id}/bundle")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("application/fhir+json")
        assert res.json()["resourceType"] == "Bundle"
        assert "attachment" not in res.headers.get("content-disposition", "")

    def test_bundle_download_sets_a_filename(self, client: TestClient, patient_id: str) -> None:
        res = client.get(f"/patients/{patient_id}/bundle", params={"download": "true"})
        assert 'filename="bundle-' in res.headers["content-disposition"]

    def test_single_resource_by_id(self, client: TestClient, patient_id: str) -> None:
        resources = client.get(f"/patients/{patient_id}").json()["resources"]
        rid = next(r["resource_id"] for r in resources if r["resource_type"] == "DocumentReference")
        res = client.get(f"/patients/{patient_id}/resources/{rid}")
        assert res.status_code == 200
        assert res.json()["resourceType"] == "DocumentReference"
        assert client.get(f"/patients/{patient_id}/resources/nope").status_code == 404

    def test_resource_of_another_patient_is_not_reachable(
        self, client: TestClient, patient_id: str
    ) -> None:
        other = next(
            r["patient_id"]
            for r in client.post("/search", json={"query": "fracture"}).json()["results"]
            if r["patient_id"] != patient_id
        )
        rid = client.get(f"/patients/{other}").json()["resources"][0]["resource_id"]
        assert client.get(f"/patients/{patient_id}/resources/{rid}").status_code == 404


class TestMeta:
    def test_health(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["patients"] == 60
        assert body["indexed_entries"] > 60
        assert body["embedding_model"]

    def test_resource_type_options(self, client: TestClient) -> None:
        options = client.get("/meta/resource-types").json()
        assert {o["value"] for o in options} == {"DocumentReference", "DiagnosticReport", "Patient"}
        assert all(o["label"] for o in options)

    def test_openapi_documents_the_search_contract(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        params = {p["name"] for p in spec["paths"]["/search"]["post"]["parameters"]}
        assert params == {"resource_type", "date_from", "date_to"}


class TestStaticUiAndErrors:
    def test_ui_is_served_from_the_same_origin(self, search_env: Container, tmp_path: Path) -> None:
        (tmp_path / "index.html").write_text("<!doctype html><title>ui</title>", encoding="utf-8")
        container = Container(
            **{
                **search_env.__dict__,
                "settings": search_env.settings.model_copy(update={"frontend_dir": tmp_path}),
            }
        )
        with TestClient(create_app(container=container)) as c:
            page = c.get("/")
            assert page.status_code == 200
            assert "<title>ui</title>" in page.text
            assert c.get("/health").status_code == 200  # API routes still win over the mount

    def test_static_assets_must_be_revalidated_so_updates_are_never_stale(
        self, search_env: Container, tmp_path: Path
    ) -> None:
        (tmp_path / "index.html").write_text("<!doctype html>", encoding="utf-8")
        (tmp_path / "app.js").write_text("export {}", encoding="utf-8")
        container = Container(
            **{
                **search_env.__dict__,
                "settings": search_env.settings.model_copy(update={"frontend_dir": tmp_path}),
            }
        )
        with TestClient(create_app(container=container)) as c:
            for path in ("/", "/app.js"):
                res = c.get(path)
                assert res.headers["cache-control"] == "no-cache"
                assert res.headers["etag"]  # validators present, so revalidation is a cheap 304
            etag = c.get("/app.js").headers["etag"]
            assert c.get("/app.js", headers={"If-None-Match": etag}).status_code == 304

    def test_unhandled_errors_do_not_leak_internals(self, search_env: Container) -> None:
        class Boom:
            def search(self, *_a: object, **_k: object) -> None:
                raise RuntimeError("secret internal detail: /etc/passwd")

        container = Container(**{**search_env.__dict__, "search": Boom()})
        with TestClient(create_app(container=container), raise_server_exceptions=False) as c:
            res = c.post("/search", json={"query": "x"})
        assert res.status_code == 500
        assert res.json() == {"detail": "Internal server error"}
        assert "secret" not in res.text
