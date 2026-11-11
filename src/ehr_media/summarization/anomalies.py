"""Deterministic anomaly detection.

"Flagged anomalies" must not depend on the model noticing them. They are computed from the
structured data (abnormal results, ingestion data-quality tags) and *given* to the model, and a
guardrail verifies each one survives into the final summary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

MAX_LAB_FLAGS = 6


@dataclass(frozen=True)
class LabLine:
    date: str  # ISO date
    name: str
    value: float
    unit: str | None
    flag: str | None  # "H" | "L" | "N" | None
    ref_low: float | None = None
    ref_high: float | None = None

    def render(self) -> str:
        text = f"{self.name}: {self.value:g}"
        if self.unit:
            text += f" {self.unit}"
        if self.flag in ("H", "L"):
            text += f" ({self.flag})"
        if self.ref_low is not None and self.ref_high is not None:
            text += f" [ref {self.ref_low:g}-{self.ref_high:g}]"
        elif self.ref_high is not None:
            text += f" [ref <{self.ref_high:g}]"
        elif self.ref_low is not None:
            text += f" [ref >{self.ref_low:g}]"
        return text


@dataclass(frozen=True)
class SystemFlag:
    """A fact the summary must surface. ``match_terms`` decide whether it was."""

    code: str
    text: str
    match_terms: tuple[str, ...]

    def is_covered_by(self, anomalies: Iterable[str]) -> bool:
        haystack = " ".join(anomalies).casefold()
        return any(term.casefold() in haystack for term in self.match_terms)


_TAG_FLAGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "DOB_MISSING": ("Date of birth missing from source records", ("birth", "dob")),
    "POSSIBLE_DUPLICATE_PATIENT": (
        "Possible duplicate patient record (same name and date of birth under another MRN)",
        ("duplicate",),
    ),
    "GENDER_CONFLICT": ("Source records disagree on patient sex", ("sex", "gender")),
    "NO_DEMOGRAPHICS": ("No patient demographics supplied with the records", ("demographic",)),
}


def detect_flags(tags: Sequence[str], labs: Sequence[LabLine]) -> tuple[SystemFlag, ...]:
    """Flags from ingestion data-quality tags plus the latest abnormal result per test."""
    flags: list[SystemFlag] = []

    latest: dict[str, LabLine] = {}
    for lab in sorted(labs, key=lambda x: (x.date, x.name)):
        latest[lab.name] = lab  # later dates overwrite earlier ones
    abnormal = sorted(
        (lab for lab in latest.values() if lab.flag in ("H", "L")),
        key=lambda x: (x.date, x.name),
        reverse=True,
    )
    for lab in abnormal[:MAX_LAB_FLAGS]:
        direction = "high" if lab.flag == "H" else "low"
        unit = f" {lab.unit}" if lab.unit else ""
        flags.append(
            SystemFlag(
                code="ABNORMAL_LAB",
                text=f"{lab.name} {lab.value:g}{unit} ({direction}) on {lab.date}",
                match_terms=(lab.name,),
            )
        )

    for tag in tags:
        if tag in _TAG_FLAGS:
            text, terms = _TAG_FLAGS[tag]
            flags.append(SystemFlag(code=tag, text=text, match_terms=terms))
    return tuple(flags)
