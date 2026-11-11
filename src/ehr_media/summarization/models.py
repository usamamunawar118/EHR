"""Summary schemas.

``LlmSummary`` is the *only* shape the model is asked to produce (it becomes the JSON schema sent
to the API, so it stays free of unsupported constraints such as min/max lengths - limits are
enforced in code by :mod:`guardrails`). ``ClinicalSummary`` is what the rest of the system stores
and serves; it adds provenance, the disclaimer and guardrail findings that must not come from the
model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = (
    "AI-generated summary of synthetic records, for demonstration only. It is not a clinical "
    "decision and must not be used for diagnosis or treatment. Verify against the source documents."
)
MAX_WORDS = 200

Confidence = Literal["low", "medium", "high"]


class MediaItem(BaseModel):
    type: Literal["imaging", "lab"]
    date: str
    finding: str


class LlmSummary(BaseModel):
    """Structured output requested from the model."""

    chief_concern: str
    key_diagnoses: list[str]
    recent_media: list[MediaItem]
    flagged_anomalies: list[str]
    confidence: Confidence


class GeneratedBy(BaseModel):
    provider: str
    model: str
    prompt_version: str
    generated_at: str


class GuardrailReport(BaseModel):
    """What the deterministic checks changed or found (kept for auditability)."""

    shortened_by_retry: bool = False
    trimmed_to_limit: bool = False
    anomalies_appended: list[str] = Field(default_factory=list)
    ungrounded_values: list[str] = Field(default_factory=list)
    confidence_lowered: bool = False


class ClinicalSummary(BaseModel):
    model_config = ConfigDict(frozen=False)

    patient_id: str
    record_hash: str
    chief_concern: str
    key_diagnoses: list[str]
    recent_media: list[MediaItem]
    flagged_anomalies: list[str]
    confidence: Confidence
    disclaimer: str = DISCLAIMER
    word_count: int
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    generated_by: GeneratedBy

    def narrative(self) -> str:
        """Plain-text rendering used for embedding/search and display snippets."""

        def sentence(text: str) -> str:
            text = text.strip()
            return text if text.endswith((".", "!", "?")) else text + "."

        parts = [sentence(self.chief_concern)]
        if self.key_diagnoses:
            parts.append("Diagnoses: " + "; ".join(self.key_diagnoses) + ".")
        parts += [
            sentence(f"{m.type.capitalize()} {m.date}: {m.finding}") for m in self.recent_media
        ]
        if self.flagged_anomalies:
            parts.append("Flags: " + "; ".join(self.flagged_anomalies) + ".")
        return " ".join(parts)
