"""Small text helpers shared by FHIR mapping, summarisation and search."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ehr_media.ingestion.models import LabFlag, LabResult

# A new section starts at any line beginning with a capitalised "Label:" - upper-case headings
# ("FINDINGS:") and title-case sign-off lines ("Radiologist: Dr. X", "Attending: Dr. Y") alike.
_SECTION_LABEL = re.compile(r"^[A-Z][A-Za-z0-9 /&-]{2,40}:", re.MULTILINE)


def extract_section(text: str, label: str) -> str | None:
    """Body of ``LABEL: ...`` up to the next ``Label:`` line (or the end of the text)."""
    match = re.search(rf"^{re.escape(label)}:[ \t]*", text, re.MULTILINE | re.IGNORECASE)
    if match is None:
        return None
    start = match.end()
    nxt = _SECTION_LABEL.search(text, start)
    body = text[start : nxt.start() if nxt else len(text)]
    return " ".join(body.split()) or None


def format_number(value: float) -> str:
    return f"{value:g}"


def render_lab(lab: LabResult) -> str:
    """One human-readable line, e.g. ``Glucose: 182 mg/dL (H) [ref 70-99]``."""
    parts = [f"{lab.test_name}: {format_number(lab.value)}"]
    if lab.unit:
        parts.append(lab.unit)
    if lab.flag in (LabFlag.HIGH, LabFlag.LOW):
        parts.append(f"({lab.flag.value})")
    line = " ".join(parts)
    if lab.ref_low is not None and lab.ref_high is not None:
        line += f" [ref {format_number(lab.ref_low)}-{format_number(lab.ref_high)}]"
    elif lab.ref_high is not None:
        line += f" [ref <{format_number(lab.ref_high)}]"
    elif lab.ref_low is not None:
        line += f" [ref >{format_number(lab.ref_low)}]"
    return line


def render_lab_panel(labs: Sequence[LabResult]) -> str:
    """Narrative for a set of results collected together (used as the report's text)."""
    when = labs[0].collected_date.isoformat() if labs else ""
    return f"LABORATORY REPORT {when}\n" + "\n".join(render_lab(lab) for lab in labs)


def lab_conclusion(labs: Sequence[LabResult]) -> str:
    abnormal = [lab for lab in labs if lab.flag in (LabFlag.HIGH, LabFlag.LOW)]
    if not abnormal:
        return "All reported results are within reference ranges."
    return "Abnormal results: " + "; ".join(render_lab(lab) for lab in abnormal) + "."
