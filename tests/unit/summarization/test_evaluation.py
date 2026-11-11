"""The evaluation harness must catch bad summaries, not just score good ones."""

from __future__ import annotations

from ehr_media.storage.bundle_repo import StoredBundle
from ehr_media.storage.summary_cache import SummaryCache
from ehr_media.summarization.evaluation import evaluate
from ehr_media.summarization.models import MediaItem
from ehr_media.summarization.service import SummaryService
from tests.unit.summarization.conftest import ScriptedLLM, good_summary


def run(stored: StoredBundle, cache: SummaryCache, llm_summary):  # type: ignore[no-untyped-def]
    summary = SummaryService(ScriptedLLM([llm_summary]), cache).summarize(stored).summary
    return summary


def test_good_summary_scores_cleanly(stored: StoredBundle, cache: SummaryCache) -> None:
    summary = run(stored, cache, good_summary(key_diagnoses=["Pneumonia (J18.9)"]))
    report = evaluate([(summary, stored)])

    assert report.n == 1
    assert report.within_word_limit_rate == 1.0
    assert report.disclaimer_rate == 1.0
    assert report.grounded_rate == 1.0
    assert report.passes_hard_requirements()
    assert report.confidence_counts == {"high": 1}


def test_model_flag_recall_exposes_what_the_guardrail_had_to_add(
    stored: StoredBundle, cache: SummaryCache
) -> None:
    summary = run(stored, cache, good_summary(flagged_anomalies=[]))
    report = evaluate([(summary, stored)])
    assert report.model_flag_recall == 0.0  # the model itself missed the one abnormal lab


def test_ungrounded_summary_is_reported_as_a_failure(
    stored: StoredBundle, cache: SummaryCache
) -> None:
    bad = good_summary(
        recent_media=[MediaItem(type="lab", date="2025-03-01", finding="Potassium 7.9")],
    )
    summary = run(stored, cache, bad)
    report = evaluate([(summary, stored)])
    assert report.grounded_rate == 0.0
    assert any("7.9" in f for f in report.failures)


def test_empty_input_is_handled(stored: StoredBundle) -> None:
    report = evaluate([])
    assert report.n == 0
    assert report.model_flag_recall is None
    assert not report.passes_hard_requirements()
