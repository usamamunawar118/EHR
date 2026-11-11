"""Patient reconciliation: merge per-source demographic claims and detect identifier conflicts.

Patient-safety policy: two records that share an MRN but describe *different people* are never
merged. Every record under a conflicting MRN is quarantined for human review.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from ehr_media.ingestion.audit import AuditTrail
from ehr_media.ingestion.builders import PatientClaim
from ehr_media.ingestion.models import (
    AuditAction,
    CanonicalPatient,
    Gender,
    RecordType,
    Severity,
    SourceRef,
)
from ehr_media.ingestion.normalizers.names import NameParse, name_key, name_similarity

NAME_CONFLICT_THRESHOLD = 0.6
_SOURCE_PRIORITY = {"json": 0, "csv": 1, "txt": 2}


@dataclass
class ReconcileOutcome:
    patients: dict[str, CanonicalPatient] = field(default_factory=dict)
    conflicts: dict[str, str] = field(default_factory=dict)  # mrn -> human-readable reason


def _detect_conflict(claims: Sequence[PatientClaim]) -> str | None:
    dobs = sorted({c.dob.isoformat() for c in claims if c.dob})
    if len(dobs) > 1:
        return f"conflicting dates of birth under one MRN: {', '.join(dobs)}"
    names = {(c.family, c.given) for c in claims if c.family or c.given}
    for a, b in combinations(sorted(names, key=str), 2):
        if name_similarity(NameParse(*a), NameParse(*b)) < NAME_CONFLICT_THRESHOLD:
            return f"conflicting names under one MRN: {a[1]} {a[0]} vs {b[1]} {b[0]}"
    return None


def _best_name(claims: Sequence[PatientClaim]) -> tuple[str | None, str | None]:
    counts: Counter[tuple[str | None, str | None]] = Counter()
    priority: dict[tuple[str | None, str | None], int] = {}
    for c in claims:
        key = (c.family, c.given)
        if not (c.family or c.given):
            continue
        counts[key] += 1
        priority[key] = min(priority.get(key, 9), _SOURCE_PRIORITY[c.source.format])
    if not counts:
        return None, None
    best = max(counts, key=lambda k: (counts[k], -priority[k]))
    return best


def reconcile_patients(
    claims: Iterable[PatientClaim], all_mrns: Iterable[str], trail: AuditTrail
) -> ReconcileOutcome:
    """Merge claims per MRN, flag conflicts and possible duplicate patients."""
    by_mrn: dict[str, list[PatientClaim]] = defaultdict(list)
    for claim in claims:
        by_mrn[claim.mrn].append(claim)

    outcome = ReconcileOutcome()
    pending: dict[str, CanonicalPatient] = {}

    for mrn in sorted(set(all_mrns) | by_mrn.keys()):
        group = by_mrn.get(mrn, [])
        record_id = f"patient:{mrn}"

        def audit(
            rule: str,
            action: AuditAction,
            severity: Severity,
            message: str,
            *,
            _rid: str = record_id,
            _mrn: str = mrn,
        ) -> None:
            trail.add(
                record_id=_rid,
                record_type=RecordType.PATIENT,
                mrn=_mrn,
                field="patient",
                rule=rule,
                action=action,
                severity=severity,
                message=message,
            )

        if (reason := _detect_conflict(group)) is not None:
            outcome.conflicts[mrn] = reason
            audit(
                "IDENTIFIER_CONFLICT",
                AuditAction.REJECTED,
                Severity.ERROR,
                f"{reason}; all records under this MRN quarantined for review",
            )
            continue

        flags: list[str] = []
        family, given = _best_name(group)
        dobs = {c.dob for c in group if c.dob}
        dob = next(iter(dobs)) if dobs else None

        genders = Counter(c.gender for c in group if c.gender is not Gender.UNKNOWN)
        gender = genders.most_common(1)[0][0] if genders else Gender.UNKNOWN
        if len(genders) > 1:
            flags.append("GENDER_CONFLICT")
            audit(
                "GENDER_CONFLICT",
                AuditAction.FLAGGED,
                Severity.WARNING,
                f"sources disagree on gender ({dict(genders)}); most common kept",
            )

        if dob is None:
            flags.append("DOB_MISSING")
            audit(
                "DOB_MISSING",
                AuditAction.FLAGGED,
                Severity.WARNING,
                "no valid date of birth in any source",
            )
        if not group:
            flags.append("NO_DEMOGRAPHICS")
            audit(
                "NO_DEMOGRAPHICS",
                AuditAction.FLAGGED,
                Severity.WARNING,
                "clinical records exist but no demographics were supplied",
            )
        if len(group) > 1:
            sources = sorted({c.source.format for c in group})
            audit(
                "PATIENT_MERGED",
                AuditAction.MERGED,
                Severity.INFO,
                f"merged {len(group)} demographic claims from {', '.join(sources)}",
            )

        sources_seen: dict[str, SourceRef] = {c.source.file: c.source for c in group}
        pending[mrn] = CanonicalPatient(
            mrn=mrn,
            family_name=family,
            given_name=given,
            birth_date=dob,
            gender=gender,
            flags=tuple(flags),
            sources=tuple(sources_seen.values()),
        )

    outcome.patients = _flag_possible_duplicates(pending, trail)
    return outcome


def _flag_possible_duplicates(
    patients: dict[str, CanonicalPatient], trail: AuditTrail
) -> dict[str, CanonicalPatient]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mrn, p in patients.items():
        key = name_key(p.family_name, p.given_name)
        if key and p.birth_date:
            groups[(key, p.birth_date.isoformat())].append(mrn)

    result = dict(patients)
    for mrns in groups.values():
        if len(mrns) < 2:
            continue
        for mrn in mrns:
            others = ", ".join(m for m in mrns if m != mrn)
            trail.add(
                record_id=f"patient:{mrn}",
                record_type=RecordType.PATIENT,
                mrn=mrn,
                field="patient",
                rule="POSSIBLE_DUPLICATE_PATIENT",
                action=AuditAction.FLAGGED,
                severity=Severity.WARNING,
                message=f"same name and date of birth also filed under MRN {others}; "
                "not merged automatically",
            )
            p = result[mrn]
            result[mrn] = p.model_copy(update={"flags": (*p.flags, "POSSIBLE_DUPLICATE_PATIENT")})
    return result
