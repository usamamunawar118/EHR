"""Summarisation workflow: cache -> context -> LLM -> guardrails -> cache."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ehr_media.storage.bundle_repo import StoredBundle
from ehr_media.storage.summary_cache import SummaryCache
from ehr_media.summarization.context import build_context
from ehr_media.summarization.guardrails import (
    count_words,
    ensure_flags,
    find_ungrounded,
    lower_confidence,
    trim_to_limit,
)
from ehr_media.summarization.llm import SummaryLLM
from ehr_media.summarization.models import (
    DISCLAIMER,
    MAX_WORDS,
    ClinicalSummary,
    GeneratedBy,
    GuardrailReport,
)
from ehr_media.summarization.prompts import PROMPT_VERSION, build_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummaryOutcome:
    summary: ClinicalSummary
    cache_hit: bool


@dataclass
class BatchResult:
    generated: int = 0
    cache_hits: int = 0
    failures: dict[str, str] = field(default_factory=dict)  # patient_id -> reason
    summaries: list[ClinicalSummary] = field(default_factory=list)


class SummaryService:
    def __init__(self, llm: SummaryLLM, cache: SummaryCache) -> None:
        self._llm = llm
        self._cache = cache

    def summarize(self, stored: StoredBundle, *, force: bool = False) -> SummaryOutcome:
        patient_id = stored.patient.patient_id
        if not force:
            cached = self._cache.get(
                patient_id, stored.content_hash, PROMPT_VERSION, self._llm.model
            )
            if cached is not None:
                return SummaryOutcome(cached, cache_hit=True)

        ctx = build_context(stored.bundle)
        raw = self._llm.summarize(build_prompt(ctx))

        shortened = False
        if count_words(raw) > MAX_WORDS:  # one explicit "shorten" retry before trimming in code
            raw = self._llm.summarize(build_prompt(ctx, shorten=True))
            shortened = True

        raw, appended = ensure_flags(raw, ctx.flags)
        raw, trimmed = trim_to_limit(raw)
        raw, appended_again = ensure_flags(raw, ctx.flags)  # trimming must not drop a flag
        appended += appended_again

        ungrounded = find_ungrounded(raw, ctx.rendered)
        confidence = lower_confidence(raw.confidence) if ungrounded else raw.confidence
        if ungrounded:
            logger.warning(
                "patient %s: %d ungrounded value(s) in summary", patient_id, len(ungrounded)
            )

        summary = ClinicalSummary(
            patient_id=patient_id,
            record_hash=stored.content_hash,
            chief_concern=raw.chief_concern,
            key_diagnoses=raw.key_diagnoses,
            recent_media=raw.recent_media,
            flagged_anomalies=raw.flagged_anomalies,
            confidence=confidence,
            disclaimer=DISCLAIMER,  # never taken from the model
            word_count=count_words(raw),
            guardrails=GuardrailReport(
                shortened_by_retry=shortened,
                trimmed_to_limit=trimmed,
                anomalies_appended=appended,
                ungrounded_values=ungrounded,
                confidence_lowered=confidence != raw.confidence,
            ),
            generated_by=GeneratedBy(
                provider=self._llm.provider,
                model=self._llm.model,
                prompt_version=PROMPT_VERSION,
                generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        self._cache.put(summary)
        return SummaryOutcome(summary, cache_hit=False)

    def summarize_all(
        self,
        bundles: Iterable[StoredBundle],
        *,
        concurrency: int = 4,
        limit: int | None = None,
        force: bool = False,
    ) -> BatchResult:
        """Summarise many patients; one patient's failure never stops the batch."""
        items = list(bundles)[:limit]
        result = BatchResult()

        def work(stored: StoredBundle) -> tuple[str, SummaryOutcome | Exception]:
            try:
                return stored.patient.patient_id, self.summarize(stored, force=force)
            except Exception as exc:  # batch boundary: isolate per-patient failures
                logger.error(
                    "summary failed for patient %s: %s",
                    stored.patient.patient_id,
                    type(exc).__name__,
                )
                return stored.patient.patient_id, exc

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            for patient_id, outcome in pool.map(work, items):
                if isinstance(outcome, Exception):
                    result.failures[patient_id] = f"{type(outcome).__name__}: {outcome}"
                    continue
                result.summaries.append(outcome.summary)
                if outcome.cache_hit:
                    result.cache_hits += 1
                else:
                    result.generated += 1
        return result
