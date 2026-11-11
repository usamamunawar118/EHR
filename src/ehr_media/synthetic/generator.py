"""Deterministic synthetic EHR data generator.

Produces three deliberately *messy* source files that mimic an EHR export:

* ``ehr_export.json``       - nested Epic-style export with inconsistent key names/date formats
* ``lab_results.csv``       - flat lab rows (UTF-8 BOM, name aliases, duplicate rows, blanks)
* ``scanned_notes/*.txt``   - OCR-style notes (header block, hard-wrapped text, OCR confusions)

and a ``DEFECT_MANIFEST.json`` describing the defects that were seeded on purpose, so tests can
assert the ingestion pipeline detects each one (ground truth rather than eyeballing).

All people, MRNs and clinical narratives are fictional. No real patient data is used.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from ehr_media.synthetic.archetypes import (
    ARCHETYPES,
    LAB_CATALOG,
    Archetype,
    Direction,
    LabSpec,
)

DEFAULT_SEED = 20250416
N_PATIENTS = 60

DocKind = Literal["progress", "discharge", "imaging"]
Channel = Literal["json", "txt"]

_FIRST_F = (
    "Alma",
    "Brenda",
    "Carla",
    "Dana",
    "Elena",
    "Fiona",
    "Gwen",
    "Hana",
    "Irene",
    "Jade",
    "Kira",
    "Lena",
    "Mara",
    "Nora",
    "Opal",
    "Priya",
    "Quinn",
    "Rosa",
    "Sana",
    "Tess",
)
_FIRST_M = (
    "Adam",
    "Boris",
    "Carlos",
    "Dev",
    "Elias",
    "Felix",
    "Gavin",
    "Hugo",
    "Ivan",
    "Jonas",
    "Kofi",
    "Luis",
    "Marek",
    "Nico",
    "Omar",
    "Pavel",
    "Rami",
    "Sven",
    "Tariq",
    "Victor",
)
_LAST = (
    "Abbott",
    "Banerjee",
    "Castillo",
    "Dvorak",
    "Eriksen",
    "Fontaine",
    "Gallagher",
    "Haddad",
    "Ibarra",
    "Jankowski",
    "Kowalczyk",
    "Lindqvist",
    "Moreau",
    "Nakamura",
    "Okafor",
    "Petrov",
    "Quintero",
    "Rasmussen",
    "Sandoval",
    "Tanaka",
    "Underhill",
    "Vasquez",
    "Whitfield",
    "Yilmaz",
    "Zielinski",
)
_CLINICIANS = ("Rivera", "Chen", "Patel", "Novak", "Okonkwo", "Lindgren", "Haddad", "Moreno")
_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTH_FULL = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# Persons given a specific seeded defect (indices into the person list).
CONFLICT_IDX = (0, 1)
POSSIBLE_DUP_IDX = (2, 3)
MISSING_DOB_IDX = (4, 5, 6)
FUTURE_DOB_IDX = (7,)
IMPUTED_DATE_IDX = (8, 9, 10)
UNDATED_DOC_IDX = (11, 12)
EMPTY_TEXT_IDX = (13,)
EXACT_DUP_IDX = (14, 15, 16, 17)
NEAR_DUP_IDX = (18, 19, 20, 21, 22)
CSV_DUP_IDX = (23, 24, 25)
OCR_MRN_IDX = (27, 28, 29)
UNMAPPED_GENDER_IDX = (30, 31)
UNKNOWN_GENDER_IDX = (32,)
KIND_INFERRED_IDX = (33, 35)
NOT_IN_JSON_IDX = tuple(range(46, 55))


# --------------------------------------------------------------------------- #
# Truth model (what really happened) - rendered into messy sources below
# --------------------------------------------------------------------------- #
@dataclass
class Encounter:
    eid: str
    setting: str
    start: date
    end: date
    reason: str


@dataclass
class Document:
    did: str
    kind: DocKind
    when: date | None
    eid: str | None
    text: str
    channel: Channel = "json"
    raw_type: str | None = None  # forced raw type label (e.g. an unrecognised "Misc")
    emit_date: bool = True  # False = the source system lost the date (seeded defect)


@dataclass
class LabRow:
    when: date
    spec: LabSpec
    value: float
    eid: str | None


@dataclass
class Person:
    idx: int
    mrn: int
    given: str
    family: str
    sex: Literal["male", "female"]
    dob_true: date
    archetype: Archetype
    dob_emit: date | None = None
    sex_raw: str | None = None
    in_json: bool = True
    encounters: list[Encounter] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    labs: list[LabRow] = field(default_factory=list)

    @property
    def mrn_canonical(self) -> str:
        return f"{self.mrn:08d}"


@dataclass
class Manifest:
    seed: int
    expected: list[dict[str, str]] = field(default_factory=list)
    expected_min_counts: dict[str, int] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)

    def expect(self, rule: str, mrn: str, record_type: str, detail: str = "") -> None:
        self.expected.append(
            {"rule": rule, "mrn": mrn, "record_type": record_type, "detail": detail}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthetic": True,
            "seed": self.seed,
            "totals": self.totals,
            "expected": self.expected,
            "expected_min_counts": self.expected_min_counts,
        }


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_date(rng: random.Random, d: date, force: str | None = None) -> str:
    """Render a date in one of several real-world styles."""
    style = (
        force
        or rng.choices(
            ("iso", "us", "us_short", "dmy_mon", "long", "compact"),
            weights=(30, 30, 10, 10, 10, 10),
        )[0]
    )
    if style == "iso":
        return d.isoformat()
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "us_short":
        return f"{d.month}/{d.day}/{d.year % 100:02d}"
    if style == "dmy_mon":
        return f"{d.day:02d}-{_MONTH_ABBR[d.month - 1]}-{d.year}"
    if style == "long":
        return f"{_MONTH_FULL[d.month - 1]} {d.day}, {d.year}"
    return f"{d:%Y%m%d}"


def _mrn_variants(mrn: int) -> list[str]:
    return [f"MRN-{mrn:08d}", f"{mrn:08d}", str(mrn), f"mrn {mrn}", f"#{mrn:08d}", f" {mrn:08d} "]


def _wrap(rng: random.Random, text: str, width: int = 72) -> str:
    """Hard-wrap like a scanner/OCR dump, occasionally hyphenating long words."""
    out: list[str] = []
    for line in text.split("\n"):
        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= width:
                current = candidate
                continue
            room = width - len(current) - 2
            if len(word) >= 9 and room >= 3 and rng.random() < 0.25:
                out.append(f"{current} {word[:room]}-".strip())
                current = word[room:]
            else:
                out.append(current)
                current = word
        out.append(current)
    return "\n".join(out)


def _ocr_noise(rng: random.Random, text: str, *, substitutions: int) -> str:
    """Whitespace noise plus ``substitutions`` single-character OCR misreads (l -> 1)."""
    positions = [m.start() for m in re.finditer("l", text)]
    chars = list(text)
    for pos in rng.sample(positions, k=min(substitutions, len(positions))):
        chars[pos] = "1"
    noisy = "".join(chars)
    noisy = re.sub(r"(?<=\w) (?=\w)", lambda _m: " " * rng.choice((1, 1, 1, 2)), noisy)
    return noisy


def _age_on(person: Person, when: date) -> int:
    return (when - person.dob_true).days // 365


# --------------------------------------------------------------------------- #
# Clinical content
# --------------------------------------------------------------------------- #
def _vitals(rng: random.Random, bias: str) -> str:
    sbp, dbp = rng.randint(108, 164), rng.randint(62, 98)
    hr = rng.randint(58, 112)
    temp = round(rng.uniform(98.0, 99.4), 1)
    spo2 = rng.randint(96, 99)
    if bias == "febrile":
        temp = round(rng.uniform(100.4, 103.1), 1)
        hr = rng.randint(96, 124)
    elif bias == "hypoxic":
        spo2 = rng.randint(87, 93)
        temp = round(rng.uniform(98.6, 101.8), 1)
    return f"BP {sbp}/{dbp} mmHg, HR {hr} bpm, Temp {temp} F, SpO2 {spo2}%"


def _progress_text(rng: random.Random, p: Person, when: date, *, followup: bool) -> str:
    a = p.archetype
    age = _age_on(p, when)
    if followup:
        hpi = (
            f"{age}-year-old {p.sex} returning for follow-up of {a.chief.lower()}. "
            f"{rng.choice(('Symptoms improving.', 'Symptoms stable.', 'Mild residual symptoms.'))}"
        )
    else:
        hpi = rng.choice(a.hpi).format(age=age, sex=p.sex)
    return (
        f"CHIEF COMPLAINT: {a.chief}\n"
        f"HPI: {hpi}\n"
        f"VITALS: {_vitals(rng, a.vitals_bias)}\n"
        f"ASSESSMENT: {a.assessment}\n"
        f"PLAN: {a.plan}\n"
        f"MEDICATIONS: {a.meds}\n"
        f"Attending: Dr. {rng.choice(_CLINICIANS)}, note dated {when.isoformat()}."
    )


def _imaging_text(rng: random.Random, tpl_idx: int, p: Person, when: date) -> str:
    tpl = p.archetype.imaging[tpl_idx]
    return (
        f"EXAM: {tpl.title}\n"
        f"DATE OF EXAM: {when.isoformat()}\n"
        f"TECHNIQUE: {tpl.technique}\n"
        f"COMPARISON: {rng.choice(('None available.', 'No prior studies for comparison.'))}\n"
        f"FINDINGS: {tpl.findings}\n"
        f"IMPRESSION: {tpl.impression}\n"
        f"Radiologist: Dr. {rng.choice(_CLINICIANS)}"
    )


def _discharge_text(p: Person, enc: Encounter) -> str:
    a = p.archetype
    dx = "; ".join(f"{i}. {label} ({code})" for i, (code, label) in enumerate(a.diagnoses, 1))
    return (
        "DISCHARGE SUMMARY\n"
        f"ADMISSION DATE: {enc.start.isoformat()}\n"
        f"DISCHARGE DATE: {enc.end.isoformat()}\n"
        f"ADMITTING COMPLAINT: {a.chief}\n"
        f"DISCHARGE DIAGNOSES: {dx}\n"
        f"HOSPITAL COURSE: {a.course}\n"
        f"DISCHARGE MEDICATIONS: {a.meds}\n"
        "FOLLOW-UP: Primary care in 1-2 weeks. Return precautions reviewed with patient."
    )


def _lab_value(rng: random.Random, spec: LabSpec, direction: Direction, *, recheck: bool) -> float:
    if direction == "high":
        factor = rng.uniform(1.12, 1.7) if not recheck else rng.uniform(1.0, 1.25)
        value = spec.high * factor
    elif direction == "low":
        factor = rng.uniform(0.45, 0.88) if not recheck else rng.uniform(0.85, 1.0)
        value = spec.low * factor
    else:
        value = rng.uniform(spec.low, spec.high)
    return round(value, spec.decimals)


def _build_clinical(rng: random.Random, p: Person) -> None:
    a = p.archetype
    start = date(2024, 3, 1) + timedelta(days=rng.randrange(0, 300))
    docs: list[Document] = []
    lab_dates: list[tuple[date, str | None]] = []

    def new_enc(setting: str, s: date, e: date, reason: str) -> Encounter:
        enc = Encounter(f"E{rng.randint(100000, 999999)}", setting, s, e, reason)
        p.encounters.append(enc)
        return enc

    def add_doc(kind: DocKind, when: date, enc: Encounter, text: str) -> None:
        docs.append(Document(f"D{p.idx:03d}{len(docs) + 1:02d}", kind, when, enc.eid, text))

    if a.setting == "inpatient":
        enc = new_enc("inpatient", start, start + timedelta(days=rng.randint(3, 6)), a.chief)
        add_doc("progress", start, enc, _progress_text(rng, p, start, followup=False))
        for i in range(len(a.imaging)):
            add_doc(
                "imaging",
                start + timedelta(days=min(i, 1)),
                enc,
                _imaging_text(rng, i, p, start + timedelta(days=min(i, 1))),
            )
        add_doc("discharge", enc.end, enc, _discharge_text(p, enc))
        lab_dates.append((start, enc.eid))
        if (enc.end - enc.start).days >= 4:
            lab_dates.append((start + timedelta(days=3), enc.eid))
        if rng.random() < 0.5:
            fu = new_enc(
                "outpatient",
                enc.end + timedelta(days=rng.randint(14, 28)),
                enc.end + timedelta(days=28),
                f"Post-discharge follow-up: {a.chief}",
            )
            fu.end = fu.start
            add_doc("progress", fu.start, fu, _progress_text(rng, p, fu.start, followup=True))
    elif a.setting == "emergency":
        enc = new_enc("emergency", start, start, a.chief)
        add_doc("progress", start, enc, _progress_text(rng, p, start, followup=False))
        for i in range(len(a.imaging)):
            add_doc("imaging", start, enc, _imaging_text(rng, i, p, start))
        lab_dates.append((start, enc.eid))
        if rng.random() < 0.6:
            fu = new_enc(
                "outpatient",
                start + timedelta(days=14),
                start + timedelta(days=14),
                f"Follow-up: {a.chief}",
            )
            add_doc("progress", fu.start, fu, _progress_text(rng, p, fu.start, followup=True))
    else:
        enc = new_enc("outpatient", start, start, a.chief)
        add_doc("progress", start, enc, _progress_text(rng, p, start, followup=False))
        for i in range(len(a.imaging)):
            add_doc("imaging", start, enc, _imaging_text(rng, i, p, start))
        lab_dates.append((start, enc.eid))
        second = start + timedelta(days=rng.randint(60, 90))
        enc2 = new_enc("outpatient", second, second, f"Follow-up: {a.chief}")
        add_doc("progress", second, enc2, _progress_text(rng, p, second, followup=True))
        lab_dates.append((second, enc2.eid))

    for draw_no, (when, eid) in enumerate(lab_dates):
        for key, direction in a.labs:
            spec = LAB_CATALOG[key]
            p.labs.append(
                LabRow(when, spec, _lab_value(rng, spec, direction, recheck=draw_no > 0), eid)
            )
    p.documents = docs


# --------------------------------------------------------------------------- #
# Source renderers
# --------------------------------------------------------------------------- #
_RAW_TYPES: dict[DocKind, tuple[str, ...]] = {
    "progress": ("Progress Note", "progress_note", "PN", "Office Visit Note"),
    "discharge": ("Discharge Summary", "DC SUMM", "discharge_summary"),
    "imaging": ("Radiology Report", "Imaging Report", "XR Report", "RAD"),
}
_SEX_RAW = {
    "male": ("M", "Male", "male", "m", "1", "MALE"),
    "female": ("F", "Female", "female", "f", "2", "FEMALE"),
}
_SETTING_RAW = {
    "inpatient": ("Inpatient", "IP", "inpatient", "IMP"),
    "outpatient": ("Outpatient", "OP", "ambulatory", "AMB"),
    "emergency": ("Emergency", "ED", "emergency", "EMER"),
}


def _raw_sex(rng: random.Random, p: Person) -> str:
    return p.sex_raw if p.sex_raw is not None else rng.choice(_SEX_RAW[p.sex])


def _render_json(rng: random.Random, people: list[Person]) -> dict[str, Any]:
    patients: list[dict[str, Any]] = []
    for p in people:
        if not p.in_json:
            continue
        style = rng.choice(("A", "B", "C"))
        mrn_raw = rng.choice(_mrn_variants(p.mrn))
        # A 2-digit year would silently turn the seeded future DOB (2031) into 1931, so the
        # implausible DOB is always rendered unambiguously.
        dob_style = "iso" if p.idx in FUTURE_DOB_IDX else None
        dob_raw = (
            render_date(rng, p.dob_emit, dob_style)
            if p.dob_emit
            else rng.choice((None, "", "UNKNOWN"))
        )
        sex_raw = _raw_sex(rng, p)
        encs: list[dict[str, Any]] = []
        for e in p.encounters:
            kind_raw = rng.choice(_SETTING_RAW[e.setting])
            s_raw, e_raw = render_date(rng, e.start), render_date(rng, e.end)
            if style == "A":
                encs.append(
                    {
                        "encounter_id": e.eid,
                        "type": kind_raw,
                        "start": s_raw,
                        "end": e_raw,
                        "reason": e.reason,
                    }
                )
            elif style == "B":
                encs.append(
                    {
                        "id": e.eid,
                        "class": kind_raw,
                        "period_start": s_raw,
                        "period_end": e_raw,
                        "reason": e.reason,
                    }
                )
            else:
                encs.append(
                    {
                        "EncounterID": e.eid,
                        "EncounterType": kind_raw,
                        "AdmitDate": s_raw,
                        "DischargeDate": e_raw,
                        "ChiefComplaint": e.reason,
                    }
                )
        docs: list[dict[str, Any]] = []
        for d in p.documents:
            if d.channel != "json":
                continue
            docs.append(_json_doc(rng, style, d))
        record: dict[str, Any]
        if style == "A":
            record = {
                "MRN": mrn_raw,
                "name": {"family": p.family, "given": p.given},
                "dob": dob_raw,
                "sex": sex_raw,
                "encounters": encs,
                "documents": docs,
            }
        elif style == "B":
            record = {
                "mrn": mrn_raw,
                "patient_name": f"{p.family}, {p.given}",
                "birthDate": dob_raw,
                "gender": sex_raw,
                "encounters": encs,
                "documents": docs,
            }
        else:
            record = {
                "MedicalRecordNumber": mrn_raw,
                "FirstName": p.given,
                "LastName": p.family,
                "DateOfBirth": dob_raw,
                "Gender": sex_raw,
                "Encounters": encs,
                "Documents": docs,
            }
        patients.append(record)
    return {
        "_synthetic_notice": "SYNTHETIC DATA - fictional people, not real patient records.",
        "export_meta": {"system": "SYNTHETIC-EHR-EXPORT", "schema_version": "1"},
        "patients": patients,
    }


def _json_doc(rng: random.Random, style: str, d: Document) -> dict[str, Any]:
    raw_type = d.raw_type or rng.choice(_RAW_TYPES[d.kind])
    date_raw = render_date(rng, d.when) if d.when and d.emit_date else None
    if style == "A":
        out = {
            "doc_id": d.did,
            "type": raw_type,
            "date": date_raw,
            "encounter_id": d.eid,
            "text": d.text,
        }
    elif style == "B":
        out = {
            "id": d.did,
            "documentType": raw_type,
            "documentDate": date_raw,
            "encounterId": d.eid,
            "body": d.text,
        }
    else:
        out = {
            "DocumentID": d.did,
            "DocType": raw_type,
            "ServiceDate": date_raw,
            "EncounterID": d.eid,
            "Content": d.text,
        }
    return out


_CSV_HEADER = [
    "MRN",
    "PatientName",
    "DOB",
    "CollectedDate",
    "TestName",
    "Result",
    "Units",
    "RefRange",
    "Flag",
    "EncounterID",
]


def _csv_rows(rng: random.Random, people: list[Person]) -> list[list[str]]:
    rows: list[list[str]] = []
    for p in people:
        for lab in p.labs:
            spec = lab.spec
            abnormal_high, abnormal_low = lab.value > spec.high, lab.value < spec.low
            flag = "H" if abnormal_high else "L" if abnormal_low else rng.choice(("", "N"))
            if (abnormal_high or abnormal_low) and rng.random() < 0.2:
                flag = ""  # missing flag: cleaner must derive it from the reference range
            ref = rng.choice((f"{spec.low:g}-{spec.high:g}", f"{spec.low:g} - {spec.high:g}"))
            rows.append(
                [
                    rng.choice((f"{p.mrn:08d}", str(p.mrn))),
                    f"{p.family.upper()}, {p.given.upper()}",
                    (
                        render_date(rng, p.dob_emit, "us")
                        if p.dob_emit and rng.random() < 0.5
                        else ""
                    ),
                    render_date(rng, lab.when),
                    rng.choice((spec.name, *spec.aliases)),
                    f"{lab.value:.{spec.decimals}f}",
                    rng.choice(spec.unit_variants),
                    ref,
                    flag,
                    lab.eid or "" if rng.random() < 0.7 else "",
                ]
            )
    return rows


def _render_txt(
    rng: random.Random, p: Person, d: Document, *, ocr_mrn: bool, substitutions: int = 0
) -> str:
    mrn = f"{p.mrn:08d}"
    if ocr_mrn:
        # '0' misread as the letter 'O' in two positions, a classic OCR failure.
        mrn = mrn.replace("0", "O", 2)
    when = render_date(rng, d.when) if d.when and d.emit_date else "UNKNOWN"
    dob = render_date(rng, p.dob_emit, "us") if p.dob_emit else "UNKNOWN"
    sex = _raw_sex(rng, p)
    raw_type = (d.raw_type or rng.choice(_RAW_TYPES[d.kind])).upper()
    body = _ocr_noise(rng, _wrap(rng, d.text), substitutions=substitutions)
    header = [
        f"*** SCANNED DOCUMENT - OCR CONFIDENCE {rng.uniform(0.82, 0.97):.2f} ***",
        f"PATIENT NAME: {p.family.upper()}, {p.given.upper()}",
        f"MRN: {mrn}",
        f"DOB: {dob}    SEX: {sex}",
        f"DATE OF SERVICE: {when}",
        f"DOCUMENT TYPE: {raw_type}",
    ]
    if d.eid:
        header.append(f"ENCOUNTER: {d.eid}")
    return "\n".join(header) + "\n" + "-" * 60 + "\n" + body + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _make_people(rng: random.Random, n: int) -> list[Person]:
    used_mrns: set[int] = set()
    people: list[Person] = []
    for idx in range(n):
        sex: Literal["male", "female"] = "female" if rng.random() < 0.5 else "male"
        mrn = rng.randint(10000, 9999999)
        while mrn in used_mrns:
            mrn = rng.randint(10000, 9999999)
        used_mrns.add(mrn)
        dob = date(1935, 1, 1) + timedelta(days=rng.randrange(0, 365 * 72))
        person = Person(
            idx=idx,
            mrn=mrn,
            given=rng.choice(_FIRST_F if sex == "female" else _FIRST_M),
            family=rng.choice(_LAST),
            sex=sex,
            dob_true=dob,
            archetype=ARCHETYPES[idx % len(ARCHETYPES)],
            dob_emit=dob,
            in_json=idx not in NOT_IN_JSON_IDX,
        )
        _build_clinical(rng, person)
        people.append(person)
    return people


def _apply_roles(rng: random.Random, people: list[Person], man: Manifest) -> list[Person]:
    """Seed the targeted defects and return extra (imposter / alias) people."""
    extras: list[Person] = []
    by_idx = {p.idx: p for p in people}

    def next_idx() -> int:
        return len(people) + len(extras)

    # Default channel split: ~25 % of documents arrive as scanned text.
    for p in people:
        for d in p.documents:
            d.channel = "txt" if (not p.in_json or rng.random() < 0.25) else "json"

    for i in MISSING_DOB_IDX:
        by_idx[i].dob_emit = None
        man.expect("DOB_MISSING", by_idx[i].mrn_canonical, "patient")
    for i in FUTURE_DOB_IDX:
        p = by_idx[i]
        p.dob_emit = date(2031, 5, 17)
        for d in p.documents:
            d.channel = "json"
        man.expect("DOB_IMPLAUSIBLE", p.mrn_canonical, "patient")
    for i in UNMAPPED_GENDER_IDX:
        by_idx[i].sex_raw = "N/A"
        man.expect("GENDER_UNMAPPED", by_idx[i].mrn_canonical, "patient")
    for i in UNKNOWN_GENDER_IDX:
        by_idx[i].sex_raw = "Unknown"

    for i in IMPUTED_DATE_IDX:
        p = by_idx[i]
        first = next(d for d in p.documents if d.kind == "progress")
        first.channel = "json"
        first.emit_date = False
        man.expect("DATE_IMPUTED", p.mrn_canonical, "document")

    for i in UNDATED_DOC_IDX:
        p = by_idx[i]
        p.documents.append(
            Document(
                f"D{p.idx:03d}99",
                "progress",
                None,
                None,
                "Telephone note: patient called with a question about "
                "medication timing. Advised to continue current regimen.",
                "json",
            )
        )
        man.expect("MISSING_DOCUMENT_DATE", p.mrn_canonical, "document")
    for i in EMPTY_TEXT_IDX:
        p = by_idx[i]
        p.documents.append(
            Document(
                f"D{p.idx:03d}98",
                "progress",
                p.encounters[0].start,
                p.encounters[0].eid,
                "   \n  ",
                "json",
            )
        )
        man.expect("EMPTY_TEXT", p.mrn_canonical, "document")

    for i in EXACT_DUP_IDX:
        p = by_idx[i]
        first = p.documents[0]
        first.channel = "json"
        p.documents.append(
            Document(
                f"D{p.idx:03d}97",
                first.kind,
                first.when,
                first.eid,
                first.text,
                "json",
                first.raw_type,
            )
        )
        man.expect("DUPLICATE_EXACT", p.mrn_canonical, "document")
    for i in NEAR_DUP_IDX:
        p = by_idx[i]
        p.documents[0].channel = "json"
        man.expect("DUPLICATE_NEAR", p.mrn_canonical, "document")
    for i in CSV_DUP_IDX:
        p = by_idx[i]
        p.labs.append(p.labs[0])
        man.expect("DUPLICATE_EXACT", p.mrn_canonical, "lab")
    for i in OCR_MRN_IDX:
        p = by_idx[i]
        p.documents[0].channel = "txt"
        man.expect("MRN_OCR_CORRECTION", p.mrn_canonical, "document")
    for i in KIND_INFERRED_IDX:
        p = by_idx[i]
        img = next(d for d in p.documents if d.kind == "imaging")
        img.channel = "json"
        img.raw_type = "Misc Document"
        man.expect("DOC_KIND_INFERRED", p.mrn_canonical, "document")

    # Identifier conflicts: a *different* person is filed under the same MRN.
    for i in CONFLICT_IDX:
        victim = by_idx[i]
        imposter = _clone_identity(rng, next_idx(), people, victim.mrn)
        _build_clinical(rng, imposter)
        for d in imposter.documents:
            d.channel = "txt"
        extras.append(imposter)
        man.expect("IDENTIFIER_CONFLICT", victim.mrn_canonical, "patient")

    # Possible duplicate patients: same name + DOB filed under a second MRN.
    for i in POSSIBLE_DUP_IDX:
        original = by_idx[i]
        alias = _clone_identity(rng, next_idx(), people + extras, None)
        alias.given, alias.family = original.given, original.family
        alias.sex, alias.dob_true, alias.dob_emit = (
            original.sex,
            original.dob_true,
            original.dob_emit,
        )
        _build_clinical(rng, alias)
        for d in alias.documents:
            d.channel = "txt"
        extras.append(alias)
        man.expect("POSSIBLE_DUPLICATE_PATIENT", original.mrn_canonical, "patient")
        man.expect("POSSIBLE_DUPLICATE_PATIENT", alias.mrn_canonical, "patient")
    return extras


def _clone_identity(
    rng: random.Random, idx: int, existing: list[Person], mrn: int | None
) -> Person:
    used = {p.mrn for p in existing}
    new_mrn = mrn
    while new_mrn is None or (mrn is None and new_mrn in used):
        new_mrn = rng.randint(10000, 9999999)
    sex: Literal["male", "female"] = "female" if rng.random() < 0.5 else "male"
    dob = date(1935, 1, 1) + timedelta(days=rng.randrange(0, 365 * 72))
    return Person(
        idx=idx,
        mrn=new_mrn,
        given=rng.choice(_FIRST_F if sex == "female" else _FIRST_M),
        family=rng.choice(_LAST),
        sex=sex,
        dob_true=dob,
        dob_emit=dob,
        archetype=ARCHETYPES[idx % len(ARCHETYPES)],
        in_json=False,
    )


def generate_dataset(
    out_dir: Path, *, seed: int = DEFAULT_SEED, n_patients: int = N_PATIENTS
) -> Manifest:
    """Write the synthetic sources to ``out_dir`` and ``DEFECT_MANIFEST.json`` to its parent."""
    rng = random.Random(seed)
    man = Manifest(seed=seed)
    people = _make_people(rng, n_patients)
    extras = _apply_roles(rng, people, man)
    everyone = people + extras

    out_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = out_dir / "scanned_notes"
    notes_dir.mkdir(exist_ok=True)
    for old in notes_dir.glob("*.txt"):
        old.unlink()

    export = _render_json(rng, everyone)
    (out_dir / "ehr_export.json").write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (out_dir / "lab_results.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        writer.writerows(_csv_rows(rng, everyone))

    n_txt = 0
    for p in everyone:
        for n, d in enumerate(p.documents, 1):
            if d.channel != "txt":
                continue
            ocr_mrn = p.idx in OCR_MRN_IDX and d is p.documents[0]
            (notes_dir / f"note_{p.mrn:08d}_{n:02d}.txt").write_text(
                _render_txt(rng, p, d, ocr_mrn=ocr_mrn), encoding="utf-8"
            )
            n_txt += 1
    # Near-duplicates: the same document re-scanned with OCR misreads.
    for i in NEAR_DUP_IDX:
        p, d = people[i], people[i].documents[0]
        (notes_dir / f"note_{p.mrn:08d}_scan_copy.txt").write_text(
            _render_txt(rng, p, d, ocr_mrn=False, substitutions=2), encoding="utf-8"
        )
        n_txt += 1

    n_docs = sum(1 for p in everyone for d in p.documents if d.channel == "json")
    man.totals = {
        "patients": len(everyone),
        "json_documents": n_docs,
        "txt_documents": n_txt,
        "lab_rows": sum(len(p.labs) for p in everyone),
    }
    # Generic noise that must be visible in the audit log.
    man.expected_min_counts = {
        "DATE_FORMAT": 20,
        "AMBIGUOUS_DATE": 5,
        "MRN_FORMAT": 10,
        "GENDER_CODE": 20,
        "LAB_FLAG_DERIVED": 1,
        "TEXT_NORMALIZED": 5,
    }
    rules = Counter(e["rule"] for e in man.expected)
    man.totals["seeded_defects"] = sum(rules.values())
    # The manifest is ground truth for tests, so it lives beside (not inside) the raw inputs.
    (out_dir.parent / "DEFECT_MANIFEST.json").write_text(
        json.dumps(man.to_dict(), indent=2), encoding="utf-8"
    )
    return man
