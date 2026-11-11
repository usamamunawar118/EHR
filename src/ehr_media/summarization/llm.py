"""LLM providers behind one small protocol.

* :class:`AnthropicSummaryLLM` - Claude via the official SDK, JSON-schema structured output
* :class:`FakeSummaryLLM` - deterministic *extractive* baseline: no network, no key, no cost.
  Used for tests and for running the whole app offline. Its output is labelled provider
  ``fake`` everywhere it is stored or shown.

Swapping in another provider means implementing :class:`SummaryLLM`; nothing else changes.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Protocol

import anthropic

from ehr_media.fhir.text import extract_section
from ehr_media.summarization.models import LlmSummary, MediaItem
from ehr_media.summarization.prompts import Prompt

if TYPE_CHECKING:
    from ehr_media.config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The provider could not produce a usable summary."""


class LLMRefusalError(LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""


class LLMTruncatedError(LLMError):
    """The response hit ``max_tokens`` before finishing."""


class SummaryLLM(Protocol):
    provider: str
    model: str

    def summarize(self, prompt: Prompt) -> LlmSummary: ...


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicSummaryLLM:
    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout_s: float = 60.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        # The SDK already retries 408/409/429/5xx with backoff; we only raise the count a little.
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=3
        )

    def summarize(self, prompt: Prompt) -> LlmSummary:
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self._max_tokens,  # thinking tokens count toward this
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                output_format=LlmSummary,  # JSON-schema structured output
                output_config={"effort": "low"},  # extraction/summarisation needs little thinking
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited by the Anthropic API after retries") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

        usage = response.usage
        logger.info(
            "llm call ok: model=%s in=%s out=%s request_id=%s",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            response._request_id,
        )
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMRefusalError(f"model declined the request (category={category})")
        if response.stop_reason == "max_tokens":
            raise LLMTruncatedError("response truncated at max_tokens; raise EHR_LLM_MAX_TOKENS")
        if response.parsed_output is None:
            raise LLMError("model returned no parseable structured output")
        return response.parsed_output


# --------------------------------------------------------------------------- #
# Offline extractive baseline
# --------------------------------------------------------------------------- #
_DX = re.compile(r"([A-Za-z][A-Za-z0-9 ,'/-]{2,70}?)\s*\(([A-Z]\d{2}(?:\.\w{1,4})?)\)")
_LEADING_FILLER = re.compile(r"^(?:with|and|plus|of)\s+", re.IGNORECASE)


class FakeSummaryLLM:
    """Rule-based summariser. Deterministic, so it doubles as a test double."""

    provider = "fake"
    model = "extractive-baseline-v2"  # bump on any behaviour change: it is part of the cache key

    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, prompt: Prompt) -> LlmSummary:
        self.calls += 1
        ctx = prompt.context

        concern = "not documented"
        for rec in ctx.records:
            if rec.label == "Progress note" and (
                cc := extract_section(rec.text, "CHIEF COMPLAINT")
            ):
                concern = cc
                break
        else:
            if ctx.encounters:
                concern = ctx.encounters[0].split(": ", 1)[-1]

        diagnoses: dict[str, str] = {}
        for rec in ctx.records:
            for label, code in _DX.findall(rec.text):
                clean = _LEADING_FILLER.sub("", label.strip(" ,"))
                diagnoses.setdefault(code, f"{clean} ({code})")

        media: list[MediaItem] = []
        for rec in ctx.records:
            if rec.label == "Imaging report" and len(media) < 2:
                impression = extract_section(rec.text, "IMPRESSION") or "no impression recorded"
                media.append(MediaItem(type="imaging", date=rec.date, finding=impression))
        if ctx.recent_labs:
            latest = ctx.recent_labs[0].date
            same_day = [lab for lab in ctx.recent_labs if lab.date == latest]
            abnormal = [lab for lab in same_day if lab.flag in ("H", "L")]
            finding = f"{len(same_day)} results, {len(abnormal)} abnormal" + (
                ": " + ", ".join(lab.name for lab in abnormal[:4]) if abnormal else ""
            )
            media.append(MediaItem(type="lab", date=latest, finding=finding))

        anomalies = [flag.text for flag in ctx.flags]
        if not ctx.records:
            confidence = "low"
        elif ctx.data_quality_tags or not diagnoses:
            confidence = "medium"
        else:
            confidence = "high"
        return LlmSummary(
            chief_concern=concern,
            key_diagnoses=list(diagnoses.values())[:5],
            recent_media=media[:4],
            flagged_anomalies=anomalies,
            confidence=confidence,
        )


def create_llm(settings: Settings) -> SummaryLLM:
    """Pick the provider from settings (``auto`` = Anthropic only when an API key is present)."""
    if settings.resolved_llm_provider == "anthropic":
        return AnthropicSummaryLLM(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    logger.warning("using the offline extractive baseline (no ANTHROPIC_API_KEY configured)")
    return FakeSummaryLLM()
