"""Automated quality checks for AI summaries, scored against the structured source data.

These are cheap, objective *proxies* and are reported alongside a manual review rubric
(``docs/summary_review_rubric.md``) - they do not replace clinician review:

* **within_word_limit**   - hard requirement: under 200 words
* **disclaimer**          - AI-generated / not-a-clinical-decision notice present
* **grounded**            - every date, ICD-10 code and multi-digit number is traceable to the
                            source records (a hallucination tripwire, not a proof of accuracy)
* **flag_recall (model)** - how often the *model itself* surfaced each deterministically-detected
                            anomaly, i.e. before the guardrail appended the missing ones
* **diagnosis_recall**    - ICD-10-coded diagnoses documented in the records that reached the
                            summary
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel

from ehr_media.storage.bundle_repo import StoredBundle
from ehr_media.summarization.context import build_context
from ehr_media.summarization.guardrails import find_ungrounded
from ehr_media.summarization.models import MAX_WORDS, ClinicalSummary, LlmSummary

_ICD_IN_PARENS = re.compile(r"\(([A-Z]\d{2}(?:\.\w{1,4})?)\)")


class EvalReport(BaseModel):
    n: int
    within_word_limit_rate: float
    disclaimer_rate: float
    grounded_rate: float
    model_flag_recall: float | None
    diagnosis_recall: float | None
    retry_rate: float
    trimmed_rate: float
    confidence_counts: dict[str, int]
    failures: list[str] = []

    def passes_hard_requirements(self) -> bool:
        return self.within_word_limit_rate == 1.0 and self.disclaimer_rate == 1.0


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def _expected_codes(texts: Iterable[str]) -> set[str]:
    return {code for text in texts for code in _ICD_IN_PARENS.findall(text)}


def evaluate(pairs: Iterable[tuple[ClinicalSummary, StoredBundle]]) -> EvalReport:
    items = list(pairs)
    n = len(items)
    within = disclaimer = grounded = retries = trims = 0
    flag_total = flag_missed = 0
    dx_total = dx_hit = 0
    confidence: Counter[str] = Counter()
    failures: list[str] = []

    for summary, stored in items:
        ctx = build_context(stored.bundle)
        as_llm = LlmSummary(
            chief_concern=summary.chief_concern,
            key_diagnoses=summary.key_diagnoses,
            recent_media=summary.recent_media,
            flagged_anomalies=summary.flagged_anomalies,
            confidence=summary.confidence,
        )
        ok_words = summary.word_count <= MAX_WORDS
        ok_disclaimer = "not a clinical decision" in summary.disclaimer.lower()
        ungrounded = find_ungrounded(as_llm, ctx.rendered)
        within += ok_words
        disclaimer += ok_disclaimer
        grounded += not ungrounded
        retries += summary.guardrails.shortened_by_retry
        trims += summary.guardrails.trimmed_to_limit
        confidence[summary.confidence] += 1

        flag_total += len(ctx.flags)
        flag_missed += len(summary.guardrails.anomalies_appended)

        expected = _expected_codes(r.text for r in ctx.records)
        if expected:
            dx_text = " ".join(summary.key_diagnoses)
            dx_total += len(expected)
            dx_hit += sum(code in dx_text for code in expected)

        if not (ok_words and ok_disclaimer):
            failures.append(f"{summary.patient_id}: words={summary.word_count}")
        if ungrounded:
            failures.append(f"{summary.patient_id}: ungrounded {ungrounded}")

    return EvalReport(
        n=n,
        within_word_limit_rate=_rate(within, n),
        disclaimer_rate=_rate(disclaimer, n),
        grounded_rate=_rate(grounded, n),
        model_flag_recall=round(1 - flag_missed / flag_total, 4) if flag_total else None,
        diagnosis_recall=_rate(dx_hit, dx_total) if dx_total else None,
        retry_rate=_rate(retries, n),
        trimmed_rate=_rate(trims, n),
        confidence_counts=dict(confidence),
        failures=failures,
    )
