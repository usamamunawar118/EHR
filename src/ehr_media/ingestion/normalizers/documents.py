"""Clinical document normalisation: text clean-up, kind classification, encounter class."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ehr_media.ingestion.models import DocumentKind, EncounterClass

_DEHYPHENATE = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
_WS = re.compile(r"\s+")

_KIND_ALIASES: dict[str, DocumentKind] = {
    "progress note": DocumentKind.PROGRESS_NOTE,
    "pn": DocumentKind.PROGRESS_NOTE,
    "office visit note": DocumentKind.PROGRESS_NOTE,
    "clinic note": DocumentKind.PROGRESS_NOTE,
    "discharge summary": DocumentKind.DISCHARGE_SUMMARY,
    "dc summ": DocumentKind.DISCHARGE_SUMMARY,
    "dc summary": DocumentKind.DISCHARGE_SUMMARY,
    "radiology report": DocumentKind.IMAGING_REPORT,
    "imaging report": DocumentKind.IMAGING_REPORT,
    "imaging": DocumentKind.IMAGING_REPORT,
    "xr report": DocumentKind.IMAGING_REPORT,
    "ct report": DocumentKind.IMAGING_REPORT,
    "rad": DocumentKind.IMAGING_REPORT,
}

KIND_TITLES: dict[DocumentKind, str] = {
    DocumentKind.PROGRESS_NOTE: "Progress Note",
    DocumentKind.DISCHARGE_SUMMARY: "Discharge Summary",
    DocumentKind.IMAGING_REPORT: "Imaging Report",
}

_ENCOUNTER_CLASS_ALIASES: dict[str, EncounterClass] = {
    "inpatient": EncounterClass.INPATIENT,
    "ip": EncounterClass.INPATIENT,
    "imp": EncounterClass.INPATIENT,
    "inp": EncounterClass.INPATIENT,
    "outpatient": EncounterClass.AMBULATORY,
    "op": EncounterClass.AMBULATORY,
    "ambulatory": EncounterClass.AMBULATORY,
    "amb": EncounterClass.AMBULATORY,
    "emergency": EncounterClass.EMERGENCY,
    "ed": EncounterClass.EMERGENCY,
    "er": EncounterClass.EMERGENCY,
    "emer": EncounterClass.EMERGENCY,
}


def clean_text(raw: str) -> str:
    """Normalise scanner/export artefacts without changing meaning.

    * unify line endings, re-join words hyphenated across hard line breaks
    * collapse runs of spaces/tabs and trailing whitespace, cap blank-line runs
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _DEHYPHENATE.sub(r"\1\2", text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()


def text_signature(text: str) -> str:
    """Whitespace/case-insensitive form used for duplicate detection."""
    return _WS.sub(" ", text).strip().casefold()


def content_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KindParse:
    kind: DocumentKind
    inferred: bool


def classify_document(raw_type: str | None, text: str) -> KindParse:
    """Map a raw document-type label to :class:`DocumentKind`, inferring it from content if the
    label is unknown (and saying so, so the caller can audit the guess)."""
    if raw_type:
        key = re.sub(r"[\s_]+", " ", raw_type).strip().lower()
        if key in _KIND_ALIASES:
            return KindParse(_KIND_ALIASES[key], False)
    upper = text.upper()
    if "IMPRESSION:" in upper and ("FINDINGS:" in upper or "TECHNIQUE:" in upper):
        return KindParse(DocumentKind.IMAGING_REPORT, True)
    if upper.lstrip().startswith("DISCHARGE SUMMARY") or "DISCHARGE DIAGNOSES" in upper:
        return KindParse(DocumentKind.DISCHARGE_SUMMARY, True)
    return KindParse(DocumentKind.PROGRESS_NOTE, True)


def classify_encounter(raw: str | None) -> tuple[EncounterClass, bool]:
    """Return ``(class, recognised)``; unknown labels default to ambulatory."""
    if raw:
        key = raw.strip().lower()
        if key in _ENCOUNTER_CLASS_ALIASES:
            return _ENCOUNTER_CLASS_ALIASES[key], True
    return EncounterClass.AMBULATORY, False
