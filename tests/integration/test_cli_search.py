"""`ehr run-all` / `ehr index` and the app factory's real start-up path (settings -> container)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ehr_media.api.app import create_app
from ehr_media.cli import app
from ehr_media.config import get_settings

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("EHR_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("EHR_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("EHR_FRONTEND_DIR", str(tmp_path / "ui"))
    monkeypatch.setenv("EHR_LLM_PROVIDER", "fake")
    monkeypatch.setenv("EHR_EMBEDDING_BACKEND", "hashing")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_run_all_builds_every_stage_and_the_app_serves_it(workspace: Path) -> None:
    res = runner.invoke(app, ["run-all", "--reference-date", "2026-11-11"])
    assert res.exit_code == 0, res.output
    assert "All stages complete" in res.output
    assert "generated=60" in res.output

    with TestClient(create_app()) as client:  # real lifespan: settings -> container
        health = client.get("/health").json()
        assert health["patients"] == 60
        assert health["indexed_entries"] > 60
        hit = client.post("/search", json={"query": "wrist fracture"}).json()["results"][0]
        assert client.get(f"/patients/{hit['patient_id']}").status_code == 200


def test_reindexing_unchanged_data_embeds_nothing(workspace: Path) -> None:
    assert runner.invoke(app, ["run-all", "--reference-date", "2026-11-11"]).exit_code == 0
    again = runner.invoke(app, ["index"])
    assert again.exit_code == 0, again.output
    assert "embedded=0" in again.output
    assert "deleted=0" in again.output


def test_index_requires_stored_bundles(workspace: Path) -> None:
    assert runner.invoke(app, ["index"]).exit_code != 0


def test_index_rejects_unknown_backend(workspace: Path) -> None:
    assert runner.invoke(app, ["index", "--backend", "bogus"]).exit_code != 0
