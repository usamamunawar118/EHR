"""Task 3 on the whole synthetic dataset, through the CLI (offline provider: no network, no key)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ehr_media.cli import app
from ehr_media.config import get_settings

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("EHR_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("EHR_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("EHR_LLM_PROVIDER", "fake")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    for cmd in (["generate-data"], ["ingest", "--reference-date", "2026-11-11"], ["build-fhir"]):
        res = runner.invoke(app, cmd)
        assert res.exit_code == 0, res.output
    yield tmp_path
    get_settings.cache_clear()


def test_summarize_all_then_rerun_is_fully_cached(workspace: Path) -> None:
    first = runner.invoke(app, ["summarize"])
    assert first.exit_code == 0, first.output
    assert "generated=60 cache_hits=0 failed=0" in first.output

    second = runner.invoke(app, ["summarize"])
    assert "generated=0 cache_hits=60 failed=0" in second.output


def test_force_regenerates(workspace: Path) -> None:
    runner.invoke(app, ["summarize", "--limit", "5"])
    again = runner.invoke(app, ["summarize", "--limit", "5", "--force"])
    assert "generated=5 cache_hits=0" in again.output


def test_every_summary_meets_the_hard_requirements(workspace: Path) -> None:
    assert runner.invoke(app, ["summarize"]).exit_code == 0
    res = runner.invoke(app, ["eval-summaries"])
    assert res.exit_code == 0, res.output

    report = json.loads(
        (workspace / "runtime" / "reports" / "summary_eval.json").read_text("utf-8")
    )
    assert report["n"] == 60
    assert report["within_word_limit_rate"] == 1.0
    assert report["disclaimer_rate"] == 1.0
    assert report["grounded_rate"] == 1.0
    assert report["diagnosis_recall"] >= 0.95


def test_eval_requires_summaries(workspace: Path) -> None:
    assert runner.invoke(app, ["eval-summaries"]).exit_code != 0


def test_summarize_rejects_unknown_provider(workspace: Path) -> None:
    assert runner.invoke(app, ["summarize", "--provider", "bogus"]).exit_code != 0
