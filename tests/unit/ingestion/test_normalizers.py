"""Table-driven tests for the field normalisers."""

from __future__ import annotations

from datetime import date

import pytest

from ehr_media.ingestion.models import DocumentKind, EncounterClass, Gender, LabFlag
from ehr_media.ingestion.normalizers.dates import parse_date
from ehr_media.ingestion.normalizers.documents import (
    classify_document,
    classify_encounter,
    clean_text,
    text_signature,
)
from ehr_media.ingestion.normalizers.gender import normalize_gender
from ehr_media.ingestion.normalizers.labs import (
    derive_flag,
    lookup_test,
    normalize_unit,
    parse_flag,
    parse_reference_range,
)
from ehr_media.ingestion.normalizers.mrn import normalize_mrn
from ehr_media.ingestion.normalizers.names import name_similarity, parse_name

REF = date(2026, 11, 11)


class TestDates:
    @pytest.mark.parametrize(
        ("raw", "expected", "fmt"),
        [
            ("2025-03-02", date(2025, 3, 2), "iso"),
            ("2025-03-02T10:15:00", date(2025, 3, 2), "iso"),
            ("20250302", date(2025, 3, 2), "compact"),
            ("03/02/2025", date(2025, 3, 2), "us_numeric"),
            ("3/2/25", date(2025, 3, 2), "us_numeric"),
            ("25/03/2025", date(2025, 3, 25), "dmy_numeric"),
            ("02-Mar-2025", date(2025, 3, 2), "dmy_month_name"),
            ("2 March 2025", date(2025, 3, 2), "dmy_month_name"),
            ("March 2, 2025", date(2025, 3, 2), "month_name_long"),
            ("Mar 2nd, 2025", date(2025, 3, 2), "month_name_long"),
        ],
    )
    def test_supported_formats(self, raw: str, expected: date, fmt: str) -> None:
        parsed = parse_date(raw, reference=REF)
        assert parsed.value == expected
        assert parsed.fmt == fmt

    @pytest.mark.parametrize(
        ("raw", "ambiguous"),
        [
            ("03/04/2025", True),  # Mar 4 or Apr 3?
            ("03/03/2025", False),  # same either way
            ("13/04/2025", False),  # 13 can only be a day
            ("03/25/2025", False),  # 25 can only be a day
            ("2025-03-04", False),
        ],
    )
    def test_ambiguity_is_flagged_not_hidden(self, raw: str, ambiguous: bool) -> None:
        assert parse_date(raw, reference=REF).ambiguous is ambiguous

    def test_ambiguous_dates_are_read_month_first(self) -> None:
        assert parse_date("03/04/2025", reference=REF).value == date(2025, 3, 4)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("3/14/56", date(1956, 3, 14)),
            ("3/14/07", date(2007, 3, 14)),
            ("3/14/26", date(2026, 3, 14)),
            ("3/14/27", date(1927, 3, 14)),
        ],
    )
    def test_two_digit_year_never_lands_in_the_future(self, raw: str, expected: date) -> None:
        parsed = parse_date(raw, reference=REF)
        assert parsed.value == expected
        assert parsed.two_digit_year

    @pytest.mark.parametrize(
        "raw", ["02/30/2025", "2025-13-01", "not a date", "32-Jan-2025", "1/2"]
    )
    def test_invalid_input_returns_error_instead_of_raising(self, raw: str) -> None:
        parsed = parse_date(raw, reference=REF)
        assert parsed.value is None
        assert parsed.error

    @pytest.mark.parametrize("raw", [None, "", "  ", "UNKNOWN", "n/a", "--"])
    def test_null_tokens(self, raw: str | None) -> None:
        assert parse_date(raw, reference=REF).error == "empty"


class TestMrn:
    @pytest.mark.parametrize(
        ("raw", "expected", "rules"),
        [
            ("00012345", "00012345", ()),
            (" 00012345 ", "00012345", ()),
            ("MRN-00012345", "00012345", ("MRN_FORMAT",)),
            ("mrn 12345", "00012345", ("MRN_FORMAT",)),
            ("#00012345", "00012345", ("MRN_FORMAT",)),
            ("12345", "00012345", ("MRN_FORMAT",)),
            ("0O012345", "00012345", ("MRN_OCR_CORRECTION",)),
            ("OO070905", "00070905", ("MRN_OCR_CORRECTION",)),
            ("MRN: 1l345", "00011345", ("MRN_OCR_CORRECTION",)),
        ],
    )
    def test_canonicalisation(self, raw: str, expected: str, rules: tuple[str, ...]) -> None:
        parsed = normalize_mrn(raw)
        assert parsed.value == expected
        assert parsed.rules == rules

    @pytest.mark.parametrize("raw", [None, "", "   ", "ABC123", "123456789", "MRN-"])
    def test_invalid_mrns_are_rejected(self, raw: str | None) -> None:
        parsed = normalize_mrn(raw)
        assert parsed.value is None
        assert parsed.error


class TestGender:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("M", Gender.MALE),
            ("male", Gender.MALE),
            ("1", Gender.MALE),
            ("m.", Gender.MALE),
            ("F", Gender.FEMALE),
            ("FEMALE", Gender.FEMALE),
            ("2", Gender.FEMALE),
            ("other", Gender.OTHER),
            ("U", Gender.UNKNOWN),
            ("Unknown", Gender.UNKNOWN),
        ],
    )
    def test_known_codes(self, raw: str, expected: Gender) -> None:
        parsed = normalize_gender(raw)
        assert parsed.value is expected
        assert parsed.recognised

    @pytest.mark.parametrize("raw", [None, "", "  "])
    def test_missing_is_unknown_but_not_an_error(self, raw: str | None) -> None:
        parsed = normalize_gender(raw)
        assert parsed.value is Gender.UNKNOWN
        assert parsed.recognised

    @pytest.mark.parametrize("raw", ["N/A", "banana", "9"])
    def test_unmapped_codes_are_flagged_not_guessed(self, raw: str) -> None:
        parsed = normalize_gender(raw)
        assert parsed.value is Gender.UNKNOWN
        assert not parsed.recognised


class TestNames:
    @pytest.mark.parametrize(
        ("kwargs", "family", "given"),
        [
            ({"full": "SMITH, JOHN A"}, "Smith", "John A"),
            ({"full": "john a smith"}, "Smith", "John A"),
            ({"family": " o'brien ", "given": "MARY"}, "O'Brien", "Mary"),
            ({"full": "  "}, None, None),
        ],
    )
    def test_parse(self, kwargs: dict[str, str], family: str | None, given: str | None) -> None:
        parsed = parse_name(**kwargs)
        assert (parsed.family, parsed.given) == (family, given)

    def test_similarity_distinguishes_typos_from_different_people(self) -> None:
        assert name_similarity(parse_name(full="Smith, Jon"), parse_name(full="Smith, John")) > 0.8
        assert name_similarity(parse_name(full="Smith, Jon"), parse_name(full="Okafor, Hana")) < 0.6


class TestLabs:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("70-99", (70.0, 99.0)),
            ("70 - 99", (70.0, 99.0)),
            ("<5.7", (None, 5.7)),
            (">10", (10.0, None)),
            ("", (None, None)),
            ("see note", (None, None)),
        ],
    )
    def test_reference_range(self, raw: str, expected: tuple[float | None, float | None]) -> None:
        assert parse_reference_range(raw) == expected

    @pytest.mark.parametrize(
        ("value", "low", "high", "expected"),
        [
            (120, 70, 99, LabFlag.HIGH),
            (50, 70, 99, LabFlag.LOW),
            (80, 70, 99, LabFlag.NORMAL),
            (99, 70, 99, LabFlag.NORMAL),
            (4, None, 5.7, LabFlag.NORMAL),
            (5, None, None, None),
        ],
    )
    def test_derive_flag(
        self, value: float, low: float | None, high: float | None, expected: LabFlag | None
    ) -> None:
        assert derive_flag(value, low, high) == expected

    def test_flag_aliases(self) -> None:
        assert parse_flag("HIGH") is LabFlag.HIGH
        assert parse_flag("l") is LabFlag.LOW
        assert parse_flag("") is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Glucose", "glucose"),
            ("GLU", "glucose"),
            ("Creat", "creatinine"),
            ("HbA1c", "hba1c"),
            ("  wbc ", "wbc"),
            ("Leukocytes", "wbc"),
        ],
    )
    def test_test_name_aliases(self, raw: str, expected: str) -> None:
        test = lookup_test(raw)
        assert test is not None
        assert test.key == expected

    def test_unknown_test_name(self) -> None:
        assert lookup_test("Mystery Marker") is None

    @pytest.mark.parametrize(
        ("raw", "expected", "recognised"),
        [
            ("mg/dl", "mg/dL", True),
            ("MG/DL", "mg/dL", True),
            ("mEq/L", "mmol/L", True),
            ("percent", "%", True),
            (None, None, True),
            ("furlongs", "furlongs", False),
        ],
    )
    def test_units(self, raw: str | None, expected: str | None, recognised: bool) -> None:
        assert normalize_unit(raw) == (expected, recognised)


class TestDocuments:
    def test_clean_text_rejoins_hyphenated_words_and_collapses_whitespace(self) -> None:
        raw = "HPI: pneu-\nmonia  with   fever.\r\n\r\n\r\n\r\nPLAN:  rest  \t"
        assert clean_text(raw) == "HPI: pneumonia with fever.\n\nPLAN: rest"

    def test_signature_ignores_wrapping_and_case(self) -> None:
        assert text_signature("Right  lower\nLOBE") == text_signature("right lower lobe")

    @pytest.mark.parametrize(
        ("raw_type", "kind"),
        [
            ("Progress Note", DocumentKind.PROGRESS_NOTE),
            ("PN", DocumentKind.PROGRESS_NOTE),
            ("progress_note", DocumentKind.PROGRESS_NOTE),
            ("DC SUMM", DocumentKind.DISCHARGE_SUMMARY),
            ("Radiology Report", DocumentKind.IMAGING_REPORT),
            ("RAD", DocumentKind.IMAGING_REPORT),
        ],
    )
    def test_known_labels_are_not_marked_inferred(self, raw_type: str, kind: DocumentKind) -> None:
        parsed = classify_document(raw_type, "irrelevant")
        assert parsed.kind is kind
        assert not parsed.inferred

    def test_unknown_label_is_inferred_from_content_and_says_so(self) -> None:
        text = "EXAM: CT HEAD\nTECHNIQUE: axial\nFINDINGS: none\nIMPRESSION: negative"
        parsed = classify_document("Misc Document", text)
        assert parsed.kind is DocumentKind.IMAGING_REPORT
        assert parsed.inferred

    @pytest.mark.parametrize(
        ("raw", "cls", "recognised"),
        [
            ("Inpatient", EncounterClass.INPATIENT, True),
            ("ED", EncounterClass.EMERGENCY, True),
            ("amb", EncounterClass.AMBULATORY, True),
            ("???", EncounterClass.AMBULATORY, False),
            (None, EncounterClass.AMBULATORY, False),
        ],
    )
    def test_encounter_class(self, raw: str | None, cls: EncounterClass, recognised: bool) -> None:
        assert classify_encounter(raw) == (cls, recognised)
