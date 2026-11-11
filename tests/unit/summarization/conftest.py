"""Fixtures for summarisation tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ehr_media.fhir.builder import PatientBundle
from ehr_media.fhir.validation import validate_bundle
from ehr_media.storage.bundle_repo import BundleRepository, StoredBundle
from ehr_media.storage.db import Database
from ehr_media.storage.summary_cache import SummaryCache
from ehr_media.summarization.models import LlmSummary, MediaItem
from ehr_media.summarization.prompts import Prompt


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "s.db")


@pytest.fixture
def cache(db: Database) -> SummaryCache:
    return SummaryCache(db)


@pytest.fixture
def stored(db: Database, patient_bundle: PatientBundle) -> StoredBundle:
    repo = BundleRepository(db)
    repo.save(patient_bundle, validate_bundle(patient_bundle.bundle))
    result = repo.get_bundle(patient_bundle.patient_id)
    assert result is not None
    return result


def good_summary(**overrides: object) -> LlmSummary:
    base: dict[str, object] = {
        "chief_concern": "Cough and fever",
        "key_diagnoses": ["Pneumonia (J18.9)"],
        "recent_media": [
            MediaItem(type="imaging", date="2025-03-01", finding="Right lower lobe pneumonia."),
            MediaItem(type="lab", date="2025-03-01", finding="White Blood Cell Count high"),
        ],
        "flagged_anomalies": ["White Blood Cell Count 15.2 10*3/uL (high) on 2025-03-01"],
        "confidence": "high",
    }
    base.update(overrides)
    return LlmSummary(**base)  # type: ignore[arg-type]


class ScriptedLLM:
    """Test double: returns queued summaries (or raises queued exceptions), records prompts."""

    provider = "scripted"

    def __init__(
        self,
        responses: list[LlmSummary | Exception] | Callable[[Prompt], LlmSummary],
        model: str = "scripted-1",
    ) -> None:
        self.model = model
        self._responses = responses
        self.prompts: list[Prompt] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def summarize(self, prompt: Prompt) -> LlmSummary:
        self.prompts.append(prompt)
        if callable(self._responses):
            return self._responses(prompt)
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item
