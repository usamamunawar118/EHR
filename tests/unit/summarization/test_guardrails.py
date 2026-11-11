"""Deterministic guardrails: word limit, anomaly coverage, grounding, confidence."""

from __future__ import annotations

import pytest

from ehr_media.summarization.anomalies import SystemFlag
from ehr_media.summarization.guardrails import (
    count_words,
    ensure_flags,
    find_ungrounded,
    lower_confidence,
    trim_to_limit,
)
from ehr_media.summarization.models import MAX_WORDS, LlmSummary, MediaItem
from tests.unit.summarization.conftest import good_summary


def verbose(n_words: int = 400) -> LlmSummary:
    sentence = " ".join(f"word{i}" for i in range(n_words // 4))
    return LlmSummary(
        chief_concern=sentence,
        key_diagnoses=[sentence, sentence, sentence, sentence],
        recent_media=[
            MediaItem(type="imaging", date="2025-03-01", finding=sentence),
            MediaItem(type="lab", date="2025-03-01", finding=sentence),
            MediaItem(type="lab", date="2025-02-01", finding=sentence),
        ],
        flagged_anomalies=["Glucose 182 mg/dL (high) on 2025-03-01"],
        confidence="medium",
    )


class TestWordLimit:
    def test_counts_every_reader_visible_field(self) -> None:
        s = good_summary()
        concern = 3  # "Cough and fever"
        diagnoses = 2  # "Pneumonia (J18.9)"
        imaging = 1 + 1 + 4  # type + date + "Right lower lobe pneumonia."
        lab = 1 + 1 + 5  # type + date + "White Blood Cell Count high"
        anomaly = 9  # "White Blood Cell Count 15.2 10*3/uL (high) on 2025-03-01"
        assert count_words(s) == concern + diagnoses + imaging + lab + anomaly == 27

    def test_short_summary_is_untouched(self) -> None:
        s = good_summary()
        out, trimmed = trim_to_limit(s)
        assert not trimmed
        assert out is s

    def test_over_long_summary_is_trimmed_below_the_limit(self) -> None:
        s = verbose()
        assert count_words(s) > MAX_WORDS
        out, trimmed = trim_to_limit(s)
        assert trimmed
        assert count_words(out) <= MAX_WORDS
        assert out.key_diagnoses  # never trimmed to nothing
        assert out.chief_concern

    def test_input_is_not_mutated(self) -> None:
        s = verbose()
        before = s.model_dump()
        trim_to_limit(s)
        assert s.model_dump() == before

    def test_trimming_keeps_mandatory_anomalies_when_possible(self) -> None:
        s = verbose()
        out, _ = trim_to_limit(s)
        assert out.flagged_anomalies == s.flagged_anomalies


class TestFlags:
    flag = SystemFlag("ABNORMAL_LAB", "Glucose 182 mg/dL (high) on 2025-03-01", ("Glucose",))

    def test_missing_flag_is_appended_and_reported(self) -> None:
        s = good_summary(flagged_anomalies=[])
        out, appended = ensure_flags(s, [self.flag])
        assert appended == [self.flag.text]
        assert out.flagged_anomalies == [self.flag.text]
        assert s.flagged_anomalies == []  # original untouched

    def test_covered_flag_is_not_duplicated(self) -> None:
        s = good_summary(flagged_anomalies=["Raised glucose of 182"])
        out, appended = ensure_flags(s, [self.flag])
        assert appended == []
        assert out is s


class TestGrounding:
    corpus = (
        'Hemoglobin: 10.2 g/dL (L)\n<labs date="2025-03-04">\nASSESSMENT: Pneumonia (J18.9). '
        "48-year-old patient"
    )

    def test_values_present_in_source_are_grounded(self) -> None:
        s = good_summary(
            chief_concern="48-year-old with pneumonia",
            key_diagnoses=["Pneumonia (J18.9)"],
            recent_media=[MediaItem(type="lab", date="2025-03-04", finding="Hemoglobin 10.2 low")],
            flagged_anomalies=[],
        )
        assert find_ungrounded(s, self.corpus) == []

    def test_invented_number_date_and_code_are_all_caught(self) -> None:
        s = good_summary(
            key_diagnoses=["Sepsis (A41.9)"],
            recent_media=[MediaItem(type="lab", date="2025-04-01", finding="Hemoglobin 9.8 low")],
            flagged_anomalies=[],
        )
        assert set(find_ungrounded(s, self.corpus)) == {"A41.9", "2025-04-01", "9.8"}

    def test_rounded_value_is_not_treated_as_grounded(self) -> None:
        s = good_summary(
            recent_media=[MediaItem(type="lab", date="2025-03-04", finding="Hemoglobin 10 low")],
            flagged_anomalies=[],
            key_diagnoses=[],
        )
        assert find_ungrounded(s, self.corpus) == ["10"]

    def test_single_digit_counts_are_ignored(self) -> None:
        s = good_summary(
            flagged_anomalies=["2 abnormal results"], key_diagnoses=[], recent_media=[]
        )
        assert find_ungrounded(s, self.corpus) == []

    def test_a_number_inside_a_longer_one_does_not_count_as_present(self) -> None:
        s = good_summary(flagged_anomalies=["value 102"], key_diagnoses=[], recent_media=[])
        assert find_ungrounded(s, "reading 1020 recorded") == ["102"]


@pytest.mark.parametrize(
    ("before", "after"), [("high", "medium"), ("medium", "low"), ("low", "low")]
)
def test_lower_confidence(before: str, after: str) -> None:
    assert lower_confidence(before) == after  # type: ignore[arg-type]
