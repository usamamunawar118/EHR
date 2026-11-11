"""Providers: the offline baseline and the Anthropic adapter (with the SDK stubbed)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from ehr_media.config import Settings
from ehr_media.storage.bundle_repo import StoredBundle
from ehr_media.summarization.context import build_context
from ehr_media.summarization.llm import (
    AnthropicSummaryLLM,
    FakeSummaryLLM,
    LLMError,
    LLMRefusalError,
    LLMTruncatedError,
    create_llm,
)
from ehr_media.summarization.models import LlmSummary
from ehr_media.summarization.prompts import SYSTEM_PROMPT, build_prompt
from tests.unit.summarization.conftest import good_summary


class TestFakeProvider:
    def test_builds_an_extractive_summary_from_the_records(self, stored: StoredBundle) -> None:
        llm = FakeSummaryLLM()
        out = llm.summarize(build_prompt(build_context(stored.bundle)))

        # The fixture's progress note has no CHIEF COMPLAINT section, so the baseline falls back
        # to the encounter's reason.
        assert out.chief_concern == "visit"
        assert llm.calls == 1
        assert [m.type for m in out.recent_media] == ["imaging", "lab"]
        assert out.recent_media[0].finding == "Right lower lobe pneumonia."
        assert any("White Blood Cell Count" in a for a in out.flagged_anomalies)

    def test_is_deterministic(self, stored: StoredBundle) -> None:
        prompt = build_prompt(build_context(stored.bundle))
        assert FakeSummaryLLM().summarize(prompt) == FakeSummaryLLM().summarize(prompt)

    def test_extracts_secondary_diagnoses_that_start_lowercase(self, stored: StoredBundle) -> None:
        import base64

        from tests.helpers import find

        text = "ASSESSMENT: Chronic kidney disease stage 3b (N18.32) with hyperkalemia (E87.5)."
        find(stored.bundle, "DocumentReference")[0]["content"][0]["attachment"]["data"] = (
            base64.b64encode(text.encode()).decode()
        )
        out = FakeSummaryLLM().summarize(build_prompt(build_context(stored.bundle)))
        joined = " ".join(out.key_diagnoses)
        assert "N18.32" in joined
        assert "hyperkalemia (E87.5)" in joined


def stub_client(**response: Any) -> SimpleNamespace:
    """A stand-in for ``anthropic.Anthropic`` exposing only ``messages.parse``."""
    defaults: dict[str, Any] = {
        "stop_reason": "end_turn",
        "stop_details": None,
        "parsed_output": good_summary(),
        "usage": SimpleNamespace(input_tokens=900, output_tokens=120),
        "_request_id": "req_test",
    }
    defaults.update(response)
    calls: list[dict[str, Any]] = []

    def parse(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if isinstance(defaults.get("raises"), Exception):
            raise defaults["raises"]
        return SimpleNamespace(**{k: v for k, v in defaults.items() if k != "raises"})

    return SimpleNamespace(messages=SimpleNamespace(parse=parse), calls=calls)


class TestAnthropicProvider:
    def prompt(self, stored: StoredBundle) -> Any:
        return build_prompt(build_context(stored.bundle))

    def test_request_shape(self, stored: StoredBundle) -> None:
        client = stub_client()
        llm = AnthropicSummaryLLM(model="claude-haiku-5-5", client=client)  # type: ignore[arg-type]

        result = llm.summarize(self.prompt(stored))

        assert isinstance(result, LlmSummary)
        (call,) = client.calls
        assert call["model"] == "claude-haiku-5-5"
        assert call["system"] == SYSTEM_PROMPT
        assert call["output_format"] is LlmSummary
        assert call["messages"][0]["role"] == "user"
        assert "<system_flags>" in call["messages"][0]["content"]
        # Newer models reject non-default sampling parameters, so we must not send any.
        assert not {"temperature", "top_p", "top_k"} & call.keys()
        assert call["max_tokens"] >= 1024  # thinking tokens count toward the cap

    def test_refusal_is_a_typed_error(self, stored: StoredBundle) -> None:
        client = stub_client(
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="general_harms"),
            parsed_output=None,
        )
        llm = AnthropicSummaryLLM(model="m", client=client)  # type: ignore[arg-type]
        with pytest.raises(LLMRefusalError, match="general_harms"):
            llm.summarize(self.prompt(stored))

    def test_truncation_is_a_typed_error(self, stored: StoredBundle) -> None:
        llm = AnthropicSummaryLLM(
            model="m", client=stub_client(stop_reason="max_tokens", parsed_output=None)
        )  # type: ignore[arg-type]
        with pytest.raises(LLMTruncatedError):
            llm.summarize(self.prompt(stored))

    def test_missing_parsed_output_is_an_error(self, stored: StoredBundle) -> None:
        llm = AnthropicSummaryLLM(model="m", client=stub_client(parsed_output=None))  # type: ignore[arg-type]
        with pytest.raises(LLMError, match="no parseable"):
            llm.summarize(self.prompt(stored))

    def test_connection_errors_are_wrapped(self, stored: StoredBundle) -> None:
        err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
        llm = AnthropicSummaryLLM(model="m", client=stub_client(raises=err))  # type: ignore[arg-type]
        with pytest.raises(LLMError, match="could not reach"):
            llm.summarize(self.prompt(stored))


class TestFactory:
    def test_auto_without_key_uses_the_offline_baseline(self) -> None:
        settings = Settings(llm_provider="auto", anthropic_api_key=None)
        assert isinstance(create_llm(settings), FakeSummaryLLM)

    def test_auto_with_key_uses_anthropic(self) -> None:
        settings = Settings(
            llm_provider="auto", anthropic_api_key="sk-test", llm_model="claude-haiku-5-5"
        )
        llm = create_llm(settings)
        assert isinstance(llm, AnthropicSummaryLLM)
        assert llm.model == "claude-haiku-5-5"

    def test_explicit_fake_wins_even_with_a_key(self) -> None:
        settings = Settings(llm_provider="fake", anthropic_api_key="sk-test")
        assert isinstance(create_llm(settings), FakeSummaryLLM)
