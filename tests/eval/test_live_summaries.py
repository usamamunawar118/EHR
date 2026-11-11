"""Opt-in evaluation against the real Claude API.

Run with a key:   ANTHROPIC_API_KEY=... pytest -m live -s

Summarises a sample of the synthetic patients with the configured Claude model and scores them
with the same harness used for the offline provider. The thresholds are deliberately modest: the
hard requirements (length, disclaimer) must be perfect, the soft ones guard against regressions.
Costs a few cents with a Haiku-class model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ehr_media.config import Settings
from ehr_media.fhir.service import build_and_validate, store_bundles
from ehr_media.ingestion.models import IngestResult
from ehr_media.storage.bundle_repo import BundleRepository
from ehr_media.storage.db import Database
from ehr_media.storage.summary_cache import SummaryCache
from ehr_media.summarization.evaluation import evaluate
from ehr_media.summarization.llm import AnthropicSummaryLLM
from ehr_media.summarization.service import SummaryService

SAMPLE_SIZE = 12

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


def test_claude_summaries_meet_the_requirements(
    ingest_result: IngestResult, tmp_path: Path
) -> None:
    settings = Settings()
    db = Database(tmp_path / "live.db")
    repo = BundleRepository(db)
    store_bundles(build_and_validate(ingest_result.patients), repo)

    llm = AnthropicSummaryLLM(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
    )
    bundles = list(repo.iter_bundles())[:SAMPLE_SIZE]
    result = SummaryService(llm, SummaryCache(db)).summarize_all(bundles, concurrency=4)

    assert not result.failures, result.failures
    report = evaluate((s, b) for s, b in zip(result.summaries, bundles, strict=True))
    print(report.model_dump_json(indent=2))

    assert report.passes_hard_requirements()
    assert report.grounded_rate >= 0.9
    assert report.model_flag_recall is None or report.model_flag_recall >= 0.8
    assert (report.diagnosis_recall or 0) >= 0.85
