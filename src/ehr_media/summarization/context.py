"""Turn a FHIR Bundle into the compact, PHI-minimised context shown to the model.

Privacy by design: **no name, MRN or exact date of birth** leaves the system - only age and sex.
Record text is kept (it is the clinical content), capped per document so the prompt stays small.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from ehr_media.fhir import terminology as terms
from ehr_media.ingestion.models import DocumentKind
from ehr_media.summarization.anomalies import LabLine, SystemFlag, detect_flags

MAX_RECORDS = 6
MAX_CHARS_PER_RECORD = 1500
MAX_ENCOUNTERS = 4
RECENT_LAB_DATES = 2

_DOC_LABELS = {
    terms.DOCUMENT_TYPE[DocumentKind.PROGRESS_NOTE].code: "Progress note",
    terms.DOCUMENT_TYPE[DocumentKind.DISCHARGE_SUMMARY].code: "Discharge summary",
    terms.DOCUMENT_TYPE[DocumentKind.IMAGING_REPORT].code: "Imaging report",
}


@dataclass(frozen=True)
class RecordText:
    ref: str
    label: str  # "Progress note" | "Discharge summary" | "Imaging report"
    date: str
    text: str


@dataclass(frozen=True)
class SummaryContext:
    age: int | None
    sex: str
    data_quality_tags: tuple[str, ...]
    encounters: tuple[str, ...]
    records: tuple[RecordText, ...]
    recent_labs: tuple[LabLine, ...]
    flags: tuple[SystemFlag, ...]
    rendered: str  # exactly what the model sees; also the grounding corpus

    @property
    def latest_date(self) -> str | None:
        dates = [r.date for r in self.records] + [lab.date for lab in self.recent_labs]
        return max(dates) if dates else None


# --------------------------------------------------------------------------- #
# bundle access helpers
# --------------------------------------------------------------------------- #
def _resources(bundle: Mapping[str, Any], rtype: str) -> Iterator[Mapping[str, Any]]:
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        if res.get("resourceType") == rtype:
            yield res


def _decode(attachment: Mapping[str, Any]) -> str:
    data = attachment.get("data")
    return base64.b64decode(data).decode("utf-8") if data else ""


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + " …"


def _first_coding(concept: Mapping[str, Any] | None) -> Mapping[str, Any]:
    codings: list[Mapping[str, Any]] = (concept or {}).get("coding") or [{}]
    return codings[0]


def _age(birth: str | None, on: str | None) -> int | None:
    if not birth or not on:
        return None
    b, ref = date.fromisoformat(birth), date.fromisoformat(on)
    return ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def _record_texts(bundle: Mapping[str, Any]) -> list[RecordText]:
    docs: list[tuple[str, str, str]] = []
    for res in _resources(bundle, "DocumentReference"):
        code = _first_coding(res.get("type")).get("code", "")
        label = _DOC_LABELS.get(code, "Clinical document")
        when = (res.get("context", {}).get("period", {}) or {}).get("start") or str(
            res.get("date", "")
        )[:10]
        att = (res.get("content") or [{}])[0].get("attachment", {})
        docs.append((when, label, _decode(att)))
    docs.sort(key=lambda d: d[0], reverse=True)
    recent = docs[:MAX_RECORDS]
    return [
        RecordText(ref=f"R{i}", label=label, date=when, text=_truncate(text, MAX_CHARS_PER_RECORD))
        for i, (when, label, text) in enumerate(recent, 1)
    ]


def _labs(bundle: Mapping[str, Any]) -> list[LabLine]:
    out: list[LabLine] = []
    for res in _resources(bundle, "Observation"):
        q = res.get("valueQuantity") or {}
        if "value" not in q:
            continue
        interp = _first_coding((res.get("interpretation") or [None])[0]).get("code")
        rng = (res.get("referenceRange") or [{}])[0]
        out.append(
            LabLine(
                date=str(res.get("effectiveDateTime", ""))[:10],
                name=(res.get("code") or {}).get("text")
                or _first_coding(res.get("code")).get("display", "Unnamed test"),
                value=float(q["value"]),
                unit=q.get("unit"),
                flag=interp,
                ref_low=(rng.get("low") or {}).get("value"),
                ref_high=(rng.get("high") or {}).get("value"),
            )
        )
    return out


def _encounter_lines(bundle: Mapping[str, Any]) -> list[str]:
    rows: list[tuple[str, str]] = []
    for res in _resources(bundle, "Encounter"):
        period = res.get("period") or {}
        start, end = period.get("start", ""), period.get("end")
        span = start if not end or end == start else f"{start} to {end}"
        cls = (res.get("class") or {}).get("display", "encounter")
        reason = ((res.get("reasonCode") or [{}])[0]).get("text")
        rows.append((start, f"{span} - {cls}" + (f": {reason}" if reason else "")))
    rows.sort(reverse=True)
    return [line for _, line in rows[:MAX_ENCOUNTERS]]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_context(bundle: Mapping[str, Any]) -> SummaryContext:
    patient = next(_resources(bundle, "Patient"), {})
    tags = tuple(t["code"] for t in (patient.get("meta") or {}).get("tag", []) if "code" in t)
    all_labs = _labs(bundle)
    records = _record_texts(bundle)
    lab_dates = sorted({lab.date for lab in all_labs}, reverse=True)[:RECENT_LAB_DATES]
    recent_labs = sorted(
        (lab for lab in all_labs if lab.date in lab_dates),
        key=lambda x: (x.date, x.flag not in ("H", "L"), x.name),
        reverse=True,
    )
    latest = max([r.date for r in records] + [lab.date for lab in all_labs], default=None)
    age = _age(patient.get("birthDate"), latest)
    sex = patient.get("gender", "unknown")
    flags = detect_flags(tags, all_labs)
    encounters = tuple(_encounter_lines(bundle))

    lines = [f'<patient age="{age if age is not None else "unknown"}" sex="{sex}"/>']
    if encounters:
        lines += ["<encounters>", *(f"- {e}" for e in encounters), "</encounters>"]
    for r in records:
        lines += [f'<record id="{r.ref}" type="{r.label}" date="{r.date}">', r.text, "</record>"]
    for d in sorted({lab.date for lab in recent_labs}, reverse=True):
        lines.append(f'<labs date="{d}">')
        lines += [lab.render() for lab in recent_labs if lab.date == d]
        lines.append("</labs>")
    lines.append("<system_flags>")
    lines += [f"- {f.text}" for f in flags] or ["- none"]
    lines.append("</system_flags>")

    return SummaryContext(
        age=age,
        sex=sex,
        data_quality_tags=tags,
        encounters=encounters,
        records=tuple(records),
        recent_labs=tuple(recent_labs),
        flags=flags,
        rendered="\n".join(lines),
    )


def source_corpus(contexts: Sequence[SummaryContext]) -> str:
    """Concatenated rendered context(s): the text a summary's facts must be traceable to."""
    return "\n".join(c.rendered for c in contexts)
