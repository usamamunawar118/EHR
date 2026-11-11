"""Turn raw records into validated canonical records, auditing every change.

Each ``build_*`` function either returns a canonical model or raises :class:`RecordRejected`
(the pipeline quarantines the record - nothing is ever silently dropped).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from ehr_media.ingestion.audit import AuditTrail
from ehr_media.ingestion.models import (
    AuditAction,
    CanonicalEncounter,
    ClinicalDocument,
    Gender,
    LabResult,
    RecordType,
    Severity,
    SourceRef,
)
from ehr_media.ingestion.normalizers.dates import DateParse, is_null_token, parse_date
from ehr_media.ingestion.normalizers.documents import (
    KIND_TITLES,
    classify_document,
    classify_encounter,
    clean_text,
    content_hash,
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
from ehr_media.ingestion.normalizers.names import parse_name
from ehr_media.ingestion.readers.base import RawRecord

MAX_PLAUSIBLE_AGE_YEARS = 120

EncounterKey = tuple[str, str]  # (mrn, encounter_id)


class RecordRejected(Exception):  # noqa: N818 - domain exception, reads better without "Error"
    """Raised by a builder when a record cannot be admitted to the clean set."""

    def __init__(self, rule: str, message: str, mrn: str | None = None) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule
        self.message = message
        self.mrn = mrn


@dataclass(frozen=True)
class BuildContext:
    reference_date: date
    trail: AuditTrail


@dataclass(frozen=True)
class PatientClaim:
    """One source's assertion about a patient's demographics (reconciled later)."""

    record_id: str
    mrn: str
    family: str | None
    given: str | None
    dob: date | None
    gender: Gender
    source: SourceRef


# --------------------------------------------------------------------------- #
# shared field helpers
# --------------------------------------------------------------------------- #
def _audit(
    ctx: BuildContext,
    raw: RawRecord,
    mrn: str | None,
    *,
    field: str,
    rule: str,
    action: AuditAction,
    before: object = None,
    after: object = None,
    severity: Severity = Severity.INFO,
    message: str = "",
) -> None:
    ctx.trail.add(
        record_id=raw.record_id,
        record_type=raw.record_type,
        mrn=mrn,
        field=field,
        rule=rule,
        action=action,
        before=before,
        after=after,
        severity=severity,
        message=message,
    )


def resolve_mrn(raw: RawRecord, ctx: BuildContext) -> str:
    parsed = normalize_mrn(raw.fields.get("mrn"))
    if parsed.value is None:
        raise RecordRejected("INVALID_MRN", parsed.error or "invalid MRN")
    for rule in parsed.rules:
        _audit(
            ctx,
            raw,
            parsed.value,
            field="mrn",
            rule=rule,
            action=AuditAction.NORMALIZED,
            before=raw.fields.get("mrn"),
            after=parsed.value,
            severity=Severity.WARNING if rule == "MRN_OCR_CORRECTION" else Severity.INFO,
        )
    return parsed.value


def resolve_date(raw: RawRecord, key: str, mrn: str, ctx: BuildContext) -> DateParse:
    """Parse ``raw.fields[key]`` and audit format changes / ambiguity / failures."""
    value = raw.fields.get(key)
    if is_null_token(value):
        if value and value.strip():
            _audit(
                ctx,
                raw,
                mrn,
                field=key,
                rule="NULL_TOKEN",
                action=AuditAction.NORMALIZED,
                before=value,
                after=None,
            )
        return DateParse(None, error="empty")
    parsed = parse_date(value, reference=ctx.reference_date)
    if parsed.value is None:
        _audit(
            ctx,
            raw,
            mrn,
            field=key,
            rule="DATE_UNPARSEABLE",
            action=AuditAction.FLAGGED,
            before=value,
            severity=Severity.WARNING,
            message=parsed.error or "",
        )
        return parsed
    if parsed.fmt != "iso":
        _audit(
            ctx,
            raw,
            mrn,
            field=key,
            rule="DATE_FORMAT",
            action=AuditAction.NORMALIZED,
            before=value,
            after=parsed.value.isoformat(),
            message=f"format={parsed.fmt}",
        )
    if parsed.ambiguous:
        _audit(
            ctx,
            raw,
            mrn,
            field=key,
            rule="AMBIGUOUS_DATE",
            action=AuditAction.FLAGGED,
            before=value,
            after=parsed.value.isoformat(),
            severity=Severity.WARNING,
            message="could be day-first or month-first; read as month-first (US)",
        )
    if parsed.two_digit_year:
        _audit(
            ctx,
            raw,
            mrn,
            field=key,
            rule="TWO_DIGIT_YEAR",
            action=AuditAction.NORMALIZED,
            before=value,
            after=parsed.value.isoformat(),
            message="century chosen so the date is not in the future",
        )
    if parsed.value > ctx.reference_date and key != "dob":
        _audit(
            ctx,
            raw,
            mrn,
            field=key,
            rule="DATE_IN_FUTURE",
            action=AuditAction.FLAGGED,
            before=value,
            severity=Severity.WARNING,
        )
    return parsed


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_patient_claim(raw: RawRecord, ctx: BuildContext) -> PatientClaim:
    mrn = resolve_mrn(raw, ctx)
    f = raw.fields
    name = parse_name(family=f.get("family"), given=f.get("given"), full=f.get("full_name"))

    raw_name = f.get("full_name") or " ".join(p for p in (f.get("given"), f.get("family")) if p)
    raw_name = raw_name.strip()
    clean_name = " ".join(p for p in (name.given, name.family) if p)
    # "Given Family" and "Family, Given" are both acceptable presentations of an already-clean
    # name; only audit when casing/spacing actually had to change.
    accepted = {clean_name, f"{name.family}, {name.given}"}
    if raw_name and raw_name not in accepted:
        _audit(
            ctx,
            raw,
            mrn,
            field="name",
            rule="NAME_FORMAT",
            action=AuditAction.NORMALIZED,
            before=raw_name,
            after=clean_name,
        )

    dob_parse = resolve_date(raw, "dob", mrn, ctx)
    dob = dob_parse.value
    if dob is not None:
        age_days = (ctx.reference_date - dob).days
        if age_days < 0 or age_days > MAX_PLAUSIBLE_AGE_YEARS * 365.25:
            _audit(
                ctx,
                raw,
                mrn,
                field="dob",
                rule="DOB_IMPLAUSIBLE",
                action=AuditAction.REJECTED,
                before=f.get("dob"),
                severity=Severity.ERROR,
                message="date of birth is in the future or implies age > 120; field discarded",
            )
            dob = None

    gender_raw = f.get("sex")
    gender = normalize_gender(gender_raw)
    if not gender.recognised:
        _audit(
            ctx,
            raw,
            mrn,
            field="sex",
            rule="GENDER_UNMAPPED",
            action=AuditAction.FLAGGED,
            before=gender_raw,
            after=gender.value.value,
            severity=Severity.WARNING,
            message="unrecognised gender code mapped to 'unknown'",
        )
    elif gender_raw and gender_raw.strip() != gender.value.value:
        _audit(
            ctx,
            raw,
            mrn,
            field="sex",
            rule="GENDER_CODE",
            action=AuditAction.NORMALIZED,
            before=gender_raw,
            after=gender.value.value,
        )

    return PatientClaim(raw.record_id, mrn, name.family, name.given, dob, gender.value, raw.source)


def build_encounter(raw: RawRecord, ctx: BuildContext) -> CanonicalEncounter:
    mrn = resolve_mrn(raw, ctx)
    f = raw.fields
    eid = _optional_str(f.get("encounter_id"))
    if eid is None:
        raise RecordRejected("MISSING_ENCOUNTER_ID", "encounter has no identifier", mrn)

    enc_class, recognised = classify_encounter(f.get("setting"))
    if not recognised:
        _audit(
            ctx,
            raw,
            mrn,
            field="class",
            rule="ENCOUNTER_CLASS_DEFAULTED",
            action=AuditAction.FLAGGED,
            before=f.get("setting"),
            after=enc_class.value,
            severity=Severity.WARNING,
        )

    start = resolve_date(raw, "start", mrn, ctx).value
    if start is None:
        raise RecordRejected("MISSING_ENCOUNTER_DATE", "encounter has no usable start date", mrn)
    end = resolve_date(raw, "end", mrn, ctx).value
    if end is not None and end < start:
        _audit(
            ctx,
            raw,
            mrn,
            field="end",
            rule="DATE_ORDER",
            action=AuditAction.FLAGGED,
            before=end.isoformat(),
            severity=Severity.WARNING,
            message="end before start; end date discarded",
        )
        end = None

    return CanonicalEncounter(
        record_id=raw.record_id,
        mrn=mrn,
        encounter_id=eid,
        encounter_class=enc_class,
        start_date=start,
        end_date=end,
        reason=_optional_str(f.get("reason")),
    )


def build_document(
    raw: RawRecord, ctx: BuildContext, encounter_starts: Mapping[EncounterKey, date]
) -> ClinicalDocument:
    mrn = resolve_mrn(raw, ctx)
    f = raw.fields
    raw_text = f.get("text") or ""
    text = clean_text(raw_text)
    if not text:
        raise RecordRejected("EMPTY_TEXT", "document has no text content", mrn)
    if text != raw_text.strip():
        _audit(
            ctx,
            raw,
            mrn,
            field="text",
            rule="TEXT_NORMALIZED",
            action=AuditAction.NORMALIZED,
            message="de-hyphenated line breaks / collapsed whitespace",
        )

    kind = classify_document(f.get("doc_type"), text)
    if kind.inferred:
        _audit(
            ctx,
            raw,
            mrn,
            field="kind",
            rule="DOC_KIND_INFERRED",
            action=AuditAction.FLAGGED,
            before=f.get("doc_type"),
            after=kind.kind.value,
            severity=Severity.WARNING,
            message="document type unknown; inferred from content",
        )

    eid = _optional_str(f.get("encounter_id"))
    doc_date = resolve_date(raw, "date", mrn, ctx).value
    if doc_date is None:
        start = encounter_starts.get((mrn, eid)) if eid else None
        if start is None:
            raise RecordRejected(
                "MISSING_DOCUMENT_DATE", "document has no date and no resolvable encounter", mrn
            )
        doc_date = start
        _audit(
            ctx,
            raw,
            mrn,
            field="date",
            rule="DATE_IMPUTED",
            action=AuditAction.IMPUTED,
            after=start.isoformat(),
            severity=Severity.WARNING,
            message=f"missing date taken from start of encounter {eid}",
        )

    doc_id = content_hash(mrn, kind.kind.value, doc_date.isoformat(), text_signature(text))[:16]
    return ClinicalDocument(
        record_id=raw.record_id,
        doc_id=doc_id,
        mrn=mrn,
        kind=kind.kind,
        title=KIND_TITLES[kind.kind],
        document_date=doc_date,
        encounter_id=eid,
        text=text,
        source=raw.source,
    )


_NUMBER = re.compile(r"^[<>]?\s*(-?\d+(?:[.,]\d+)?)$")


def build_lab(raw: RawRecord, ctx: BuildContext) -> LabResult:
    mrn = resolve_mrn(raw, ctx)
    f = raw.fields
    raw_name = _optional_str(f.get("test"))
    if raw_name is None:
        raise RecordRejected("MISSING_LAB_TEST", "lab row has no test name", mrn)

    test = lookup_test(raw_name)
    if test is None:
        key = re.sub(r"[^a-z0-9]+", "_", raw_name.lower()).strip("_")
        name = raw_name
        _audit(
            ctx,
            raw,
            mrn,
            field="test",
            rule="LAB_TEST_UNMAPPED",
            action=AuditAction.FLAGGED,
            before=raw_name,
            severity=Severity.WARNING,
            message="test name not in catalogue; kept as free text",
        )
    else:
        key, name = test.key, test.name
        if raw_name != name:
            _audit(
                ctx,
                raw,
                mrn,
                field="test",
                rule="LAB_TEST_NAME",
                action=AuditAction.NORMALIZED,
                before=raw_name,
                after=name,
            )

    number = _NUMBER.match((f.get("result") or "").strip())
    if number is None:
        raise RecordRejected(
            "LAB_VALUE_NONNUMERIC", f"result {f.get('result')!r} is not numeric", mrn
        )
    value = float(number[1].replace(",", "."))

    unit, recognised = normalize_unit(f.get("unit"))
    if not recognised:
        _audit(
            ctx,
            raw,
            mrn,
            field="unit",
            rule="LAB_UNIT_UNMAPPED",
            action=AuditAction.FLAGGED,
            before=f.get("unit"),
            severity=Severity.WARNING,
        )
    elif unit != (f.get("unit") or None):
        _audit(
            ctx,
            raw,
            mrn,
            field="unit",
            rule="LAB_UNIT",
            action=AuditAction.NORMALIZED,
            before=f.get("unit"),
            after=unit,
        )

    low, high = parse_reference_range(f.get("ref_range"))
    source_flag = parse_flag(f.get("flag"))
    derived = derive_flag(value, low, high)
    flag = source_flag
    if source_flag is None and derived is not None:
        flag = derived
        _audit(
            ctx,
            raw,
            mrn,
            field="flag",
            rule="LAB_FLAG_DERIVED",
            action=AuditAction.IMPUTED,
            before=f.get("flag"),
            after=derived.value,
            message="flag computed from value and reference range",
        )
    elif source_flag is not None and derived is not None and source_flag != derived:
        flag = derived
        _audit(
            ctx,
            raw,
            mrn,
            field="flag",
            rule="LAB_FLAG_MISMATCH",
            action=AuditAction.FLAGGED,
            before=source_flag.value,
            after=derived.value,
            severity=Severity.WARNING,
            message="source flag disagrees with value/reference range; recomputed",
        )

    collected = resolve_date(raw, "collected", mrn, ctx).value
    if collected is None:
        raise RecordRejected("MISSING_LAB_DATE", "lab result has no usable collection date", mrn)

    return LabResult(
        record_id=raw.record_id,
        mrn=mrn,
        test_key=key,
        test_name=name,
        value=value,
        unit=unit,
        ref_low=low,
        ref_high=high,
        flag=flag,
        collected_date=collected,
        encounter_id=_optional_str(f.get("encounter_id")),
        source=raw.source,
    )


__all__ = [
    "BuildContext",
    "EncounterKey",
    "PatientClaim",
    "RecordRejected",
    "RecordType",
    "build_document",
    "build_encounter",
    "build_lab",
    "build_patient_claim",
    "resolve_mrn",
]
