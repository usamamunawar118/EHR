"""Section extraction and lab rendering helpers."""

from __future__ import annotations

import pytest

from ehr_media.fhir.text import extract_section

IMAGING = (
    "EXAM: XR CHEST 2 VIEWS\n"
    "DATE OF EXAM: 2025-03-01\n"
    "FINDINGS: Right lower lobe consolidation.\n"
    "IMPRESSION: Right lower lobe pneumonia with small effusion.\n"
    "Radiologist: Dr. Lindgren"
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("EXAM", "XR CHEST 2 VIEWS"),
        ("FINDINGS", "Right lower lobe consolidation."),
        # Must stop at the title-case sign-off line, not swallow "Radiologist: Dr. Lindgren".
        ("IMPRESSION", "Right lower lobe pneumonia with small effusion."),
        ("Radiologist", "Dr. Lindgren"),
    ],
)
def test_sections_stop_at_the_next_label_whatever_its_case(label: str, expected: str) -> None:
    assert extract_section(IMAGING, label) == expected


def test_label_match_is_case_insensitive_and_missing_label_is_none() -> None:
    assert extract_section(IMAGING, "impression") == (
        "Right lower lobe pneumonia with small effusion."
    )
    assert extract_section(IMAGING, "COMPARISON") is None


def test_multi_line_sections_are_joined() -> None:
    text = "FINDINGS: first line\nsecond line continues\nIMPRESSION: done"
    assert extract_section(text, "FINDINGS") == "first line second line continues"


def test_last_section_runs_to_the_end() -> None:
    assert extract_section("PLAN: rest\nand fluids", "PLAN") == "rest and fluids"


def test_empty_section_is_none() -> None:
    assert extract_section("PLAN:\nFOLLOW-UP: soon", "PLAN") is None
