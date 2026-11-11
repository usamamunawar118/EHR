"""Deterministic guardrails applied to every model output, whatever the provider.

The prompt *asks* for short, grounded, complete summaries; these functions *guarantee* it:

* :func:`trim_to_limit`      - hard word limit
* :func:`ensure_flags`       - system-detected anomalies are never dropped
* :func:`find_ungrounded`    - numbers/dates/codes must exist in the source context
* :func:`lower_confidence`   - ungrounded content is never reported as high confidence
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from ehr_media.summarization.anomalies import SystemFlag
from ehr_media.summarization.models import MAX_WORDS, Confidence, LlmSummary

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ICD = re.compile(r"\b[A-Z]\d{2}(?:\.\w{1,4})?\b")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def _words(text: str) -> int:
    return len(text.split())


def count_words(summary: LlmSummary) -> int:
    """Words a reader must read: every field except confidence and the fixed disclaimer."""
    total = _words(summary.chief_concern)
    total += sum(_words(d) for d in summary.key_diagnoses)
    total += sum(_words(f"{m.type} {m.date} {m.finding}") for m in summary.recent_media)
    total += sum(_words(a) for a in summary.flagged_anomalies)
    return total


def _clip(text: str, max_words: int) -> str:
    parts = text.split()
    return text if len(parts) <= max_words else " ".join(parts[:max_words]).rstrip(",;:") + "…"


def _shorten_findings(s: LlmSummary) -> None:
    for item in s.recent_media:
        item.finding = _clip(item.finding, 15)


def _shorten_concern(s: LlmSummary) -> None:
    s.chief_concern = _clip(s.chief_concern, 25)


def _keep_media(n: int) -> Callable[[LlmSummary], None]:
    def step(s: LlmSummary) -> None:
        del s.recent_media[n:]

    return step


def _keep_diagnoses(n: int) -> Callable[[LlmSummary], None]:
    def step(s: LlmSummary) -> None:
        del s.key_diagnoses[n:]

    return step


def _shorten_anomalies(s: LlmSummary) -> None:
    s.flagged_anomalies = [_clip(a, 12) for a in s.flagged_anomalies]


# Ordered from least to most important content; anomalies are only touched as a last resort.
_TRIM_STEPS: tuple[Callable[[LlmSummary], None], ...] = (
    _shorten_findings,
    _shorten_concern,
    _keep_media(2),
    _keep_diagnoses(3),
    _keep_media(1),
    _keep_diagnoses(1),
    _shorten_anomalies,
)


def trim_to_limit(summary: LlmSummary, limit: int = MAX_WORDS) -> tuple[LlmSummary, bool]:
    """Deterministically shorten to ``limit`` words, dropping least important content first."""
    if count_words(summary) <= limit:
        return summary, False
    s = summary.model_copy(deep=True)
    for step in _TRIM_STEPS:
        step(s)
        if count_words(s) <= limit:
            break
    return s, True


def ensure_flags(summary: LlmSummary, flags: Sequence[SystemFlag]) -> tuple[LlmSummary, list[str]]:
    """Append the text of any system flag the model failed to surface."""
    missing = [f for f in flags if not f.is_covered_by(summary.flagged_anomalies)]
    if not missing:
        return summary, []
    s = summary.model_copy(deep=True)
    s.flagged_anomalies.extend(f.text for f in missing)
    return s, [f.text for f in missing]


def find_ungrounded(summary: LlmSummary, corpus: str) -> list[str]:
    """Dates, ICD-10 codes and multi-digit numbers in the summary that the source never contains.

    Single digits are ignored (counts like "2 abnormal results" are derived, not copied).
    """
    text = " ".join(
        [
            summary.chief_concern,
            *summary.key_diagnoses,
            *(f"{m.date} {m.finding}" for m in summary.recent_media),
            *summary.flagged_anomalies,
        ]
    )
    found: list[str] = []

    def take(pattern: re.Pattern[str], haystack: str) -> str:
        for match in pattern.finditer(haystack):
            token = match.group(0)
            if token not in found:
                found.append(token)
        return pattern.sub(" ", haystack)

    text = take(_DATE, text)
    text = take(_ICD, text)
    candidates = [t for t in _NUMBER.findall(text) if len(t.replace(".", "")) > 1 or "." in t]
    found.extend(t for t in dict.fromkeys(candidates) if t not in found)

    def present(token: str) -> bool:
        # Not part of a longer token: "10" must not match inside "102", "1.10" or "10.2".
        return re.search(rf"(?<![\w.]){re.escape(token)}(?!\w|\.\d)", corpus) is not None

    return [t for t in found if not present(t)]


_ONE_STEP_DOWN: dict[Confidence, Confidence] = {"high": "medium", "medium": "low", "low": "low"}


def lower_confidence(confidence: Confidence) -> Confidence:
    return _ONE_STEP_DOWN[confidence]
