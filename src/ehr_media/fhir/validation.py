"""Three-layer FHIR validation of a patient Bundle.

``fhir.resources`` guarantees *structure* (types, cardinality, date syntax) but not required
value-set bindings, reference integrity or business rules, so validation is layered:

1. **schema**    - every resource parses with its R4B model, then the Bundle as a whole
2. **reference** - every ``Reference`` resolves inside the Bundle and points at the right type
3. **profile**   - required bindings (status/gender/bundle type), required elements, code-system
                   whitelist, attachment integrity, period ordering

Not covered (documented limitation): FHIRPath invariants and terminology-server binding checks.
The official HL7 validator (Java) is the next step for full conformance.
"""

from __future__ import annotations

import base64
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from typing import Any

from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.diagnosticreport import DiagnosticReport
from fhir.resources.R4B.documentreference import DocumentReference
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.patient import Patient
from pydantic import BaseModel, ConfigDict, ValidationError

from ehr_media.fhir.terminology import ALLOWED_CODE_SYSTEMS, MRN_TYPE, VALUE_SETS

RESOURCE_CLASSES: dict[str, Any] = {
    "Patient": Patient,
    "Encounter": Encounter,
    "DocumentReference": DocumentReference,
    "DiagnosticReport": DiagnosticReport,
    "Observation": Observation,
}


class Layer(StrEnum):
    SCHEMA = "schema"
    REFERENCE = "reference"
    PROFILE = "profile"


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer: Layer
    severity: IssueSeverity
    code: str
    resource: str  # "Patient/<id>" or "Bundle"
    path: str  # e.g. "entry[2].resource.gender"
    message: str


class BundleValidation(BaseModel):
    patient_id: str | None = None
    mrn: str | None = None
    valid: bool = True
    resource_counts: dict[str, int] = {}
    issues: list[ValidationIssue] = []

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is IssueSeverity.WARNING]


def _fmt_loc(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(
        self,
        layer: Layer,
        code: str,
        resource: str,
        path: str,
        message: str,
        severity: IssueSeverity = IssueSeverity.ERROR,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                layer=layer,
                severity=severity,
                code=code,
                resource=resource,
                path=path,
                message=message,
            )
        )


def _label(resource: Mapping[str, Any]) -> str:
    return f"{resource.get('resourceType', '?')}/{resource.get('id', '?')}"


# --------------------------------------------------------------------------- #
# Layer 1 - schema
# --------------------------------------------------------------------------- #
def _validate_schema(bundle: Mapping[str, Any], out: _Collector) -> bool:
    ok = True
    for i, entry in enumerate(bundle.get("entry") or []):
        resource = entry.get("resource") if isinstance(entry, Mapping) else None
        base = f"entry[{i}].resource"
        if not isinstance(resource, Mapping):
            out.add(
                Layer.SCHEMA, "missing-resource", "Bundle", f"entry[{i}]", "entry has no resource"
            )
            ok = False
            continue
        cls = RESOURCE_CLASSES.get(str(resource.get("resourceType")))
        if cls is None:
            out.add(
                Layer.SCHEMA,
                "unsupported-resource-type",
                _label(resource),
                f"{base}.resourceType",
                f"resource type {resource.get('resourceType')!r} is not mapped by this pipeline",
            )
            ok = False
            continue
        try:
            cls.model_validate(resource)
        except ValidationError as exc:
            ok = False
            for err in exc.errors():
                out.add(
                    Layer.SCHEMA,
                    str(err["type"]),
                    _label(resource),
                    f"{base}.{_fmt_loc(err['loc'])}",
                    str(err["msg"]),
                )
    if ok:  # validate the envelope too (bundle-level structure) only when entries are clean
        try:
            Bundle.model_validate(bundle)
        except ValidationError as exc:
            ok = False
            for err in exc.errors():
                out.add(
                    Layer.SCHEMA, str(err["type"]), "Bundle", _fmt_loc(err["loc"]), str(err["msg"])
                )
    return ok


# --------------------------------------------------------------------------- #
# Layer 2 - references
# --------------------------------------------------------------------------- #
def _iter_references(node: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            sub = f"{path}.{key}" if path else str(key)
            if key == "reference" and isinstance(value, str):
                yield path, value
            else:
                yield from _iter_references(value, sub)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _iter_references(value, f"{path}[{idx}]")


# reference path (suffix) -> allowed target resource types
_TARGETS: dict[str, frozenset[str]] = {
    ".subject": frozenset({"Patient"}),
    ".encounter": frozenset({"Encounter"}),
    ".result": frozenset({"Observation"}),
}


def _expected_targets(path: str) -> frozenset[str] | None:
    for suffix, targets in _TARGETS.items():
        if re.search(rf"{re.escape(suffix)}(\[\d+\])?$", path):
            return targets
    return None


def _validate_references(bundle: Mapping[str, Any], out: _Collector) -> None:
    entries = [e for e in bundle.get("entry") or [] if isinstance(e.get("resource"), Mapping)]
    by_url: dict[str, Mapping[str, Any]] = {}
    seen_ids: Counter[tuple[str, str]] = Counter()
    patients = 0

    for i, entry in enumerate(entries):
        res = entry["resource"]
        url = entry.get("fullUrl")
        rid, rtype = res.get("id"), res.get("resourceType")
        patients += rtype == "Patient"
        seen_ids[(str(rtype), str(rid))] += 1
        if not url:
            out.add(
                Layer.REFERENCE,
                "missing-fullurl",
                _label(res),
                f"entry[{i}].fullUrl",
                "entries in a collection Bundle need a fullUrl so references can resolve",
            )
            continue
        if url in by_url:
            out.add(
                Layer.REFERENCE,
                "duplicate-fullurl",
                _label(res),
                f"entry[{i}].fullUrl",
                f"fullUrl {url} is used by more than one entry",
            )
        by_url[url] = res
        if url.startswith("urn:uuid:") and url != f"urn:uuid:{rid}":
            out.add(
                Layer.REFERENCE,
                "fullurl-id-mismatch",
                _label(res),
                f"entry[{i}].fullUrl",
                f"fullUrl {url} does not match resource id {rid}",
            )

    for (rtype, rid), n in seen_ids.items():
        if n > 1:
            out.add(
                Layer.REFERENCE,
                "duplicate-resource-id",
                f"{rtype}/{rid}",
                "entry",
                f"{rtype}/{rid} appears {n} times in the Bundle",
            )
    if patients != 1:
        out.add(
            Layer.REFERENCE,
            "patient-count",
            "Bundle",
            "entry",
            f"a patient Bundle must contain exactly one Patient, found {patients}",
        )

    relative = {f"{r.get('resourceType')}/{r.get('id')}": r for r in by_url.values()}

    for i, entry in enumerate(entries):
        res = entry["resource"]
        for path, ref in _iter_references(res, ""):
            full_path = f"entry[{i}].resource.{path}"
            target = by_url.get(ref) or relative.get(ref)
            if target is None:
                out.add(
                    Layer.REFERENCE,
                    "unresolved-reference",
                    _label(res),
                    full_path,
                    f"reference {ref!r} does not resolve to any entry in the Bundle",
                )
                continue
            allowed = _expected_targets(f".{path}")
            if allowed and target.get("resourceType") not in allowed:
                out.add(
                    Layer.REFERENCE,
                    "wrong-reference-type",
                    _label(res),
                    full_path,
                    f"{path} must reference {sorted(allowed)}, "
                    f"but {ref!r} is a {target.get('resourceType')}",
                )


# --------------------------------------------------------------------------- #
# Layer 3 - profile-lite rules
# --------------------------------------------------------------------------- #
def _walk_codings(node: Any, path: str) -> Iterator[tuple[str, Mapping[str, Any]]]:
    if isinstance(node, Mapping):
        if "system" in node and "code" in node:
            yield path, node
        for key, value in node.items():
            yield from _walk_codings(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _walk_codings(value, f"{path}[{idx}]")


def _check_binding(
    res: Mapping[str, Any],
    field: str,
    vs: str,
    base: str,
    out: _Collector,
    *,
    required: bool = True,
) -> None:
    value = res.get(field)
    if value is None:
        if required:
            out.add(
                Layer.PROFILE,
                "missing-required-element",
                _label(res),
                f"{base}.{field}",
                f"{field} is required",
            )
        return
    if value not in VALUE_SETS[vs]:
        out.add(
            Layer.PROFILE,
            "invalid-code",
            _label(res),
            f"{base}.{field}",
            f"{value!r} is not in the required value set '{vs}'",
        )


def _p_patient(res: Mapping[str, Any], base: str, out: _Collector) -> None:
    _check_binding(res, "gender", "administrative-gender", base, out, required=False)
    ids = res.get("identifier") or []
    has_mrn = any(
        any(c.get("code") == MRN_TYPE.code for c in (i.get("type") or {}).get("coding", []))
        and i.get("value")
        for i in ids
    )
    if not has_mrn:
        out.add(
            Layer.PROFILE,
            "missing-mrn",
            _label(res),
            f"{base}.identifier",
            "Patient needs an identifier of type MR with a value",
        )
    if not res.get("birthDate"):
        out.add(
            Layer.PROFILE,
            "missing-birthdate",
            _label(res),
            f"{base}.birthDate",
            "birthDate is missing",
            IssueSeverity.WARNING,
        )
    if not res.get("name"):
        out.add(
            Layer.PROFILE,
            "missing-name",
            _label(res),
            f"{base}.name",
            "name is missing",
            IssueSeverity.WARNING,
        )


def _p_encounter(res: Mapping[str, Any], base: str, out: _Collector) -> None:
    _check_binding(res, "status", "encounter-status", base, out)
    if not res.get("class"):
        out.add(
            Layer.PROFILE,
            "missing-required-element",
            _label(res),
            f"{base}.class",
            "class is required",
        )
    period = res.get("period") or {}
    if not period.get("start"):
        out.add(
            Layer.PROFILE,
            "missing-required-element",
            _label(res),
            f"{base}.period.start",
            "period.start is required",
        )
    elif period.get("end") and str(period["end"]) < str(period["start"]):
        out.add(
            Layer.PROFILE,
            "period-order",
            _label(res),
            f"{base}.period",
            "period.end is before period.start",
        )


def _p_attachment(
    res: Mapping[str, Any], att: Mapping[str, Any], base: str, out: _Collector
) -> None:
    data = att.get("data")
    if not data and not att.get("url"):
        out.add(
            Layer.PROFILE, "empty-attachment", _label(res), base, "attachment needs data or url"
        )
        return
    if data:
        try:
            raw = base64.b64decode(data, validate=True)
        except ValueError:
            out.add(
                Layer.PROFILE,
                "invalid-base64",
                _label(res),
                f"{base}.data",
                "attachment.data is not valid base64",
            )
            return
        if att.get("size") is not None and int(att["size"]) != len(raw):
            out.add(
                Layer.PROFILE,
                "attachment-size-mismatch",
                _label(res),
                f"{base}.size",
                f"declared size {att['size']} != decoded length {len(raw)}",
            )


def _p_docref(res: Mapping[str, Any], base: str, out: _Collector) -> None:
    _check_binding(res, "status", "document-reference-status", base, out)
    if not (res.get("type") or {}).get("coding"):
        out.add(
            Layer.PROFILE,
            "missing-required-element",
            _label(res),
            f"{base}.type",
            "a coded document type is required",
        )
    for i, content in enumerate(res.get("content") or []):
        _p_attachment(res, content.get("attachment") or {}, f"{base}.content[{i}].attachment", out)
    if not (res.get("context") or {}).get("encounter"):
        out.add(
            Layer.PROFILE,
            "missing-encounter",
            _label(res),
            f"{base}.context.encounter",
            "document is not linked to an encounter",
            IssueSeverity.WARNING,
        )


def _p_report(res: Mapping[str, Any], base: str, out: _Collector) -> None:
    _check_binding(res, "status", "diagnostic-report-status", base, out)
    if not (res.get("code") or {}).get("coding"):
        out.add(
            Layer.PROFILE,
            "missing-required-element",
            _label(res),
            f"{base}.code",
            "a coded report type is required",
        )
    if not (res.get("effectiveDateTime") or res.get("effectivePeriod")):
        out.add(
            Layer.PROFILE,
            "missing-effective",
            _label(res),
            f"{base}.effective[x]",
            "report has no effective date",
            IssueSeverity.WARNING,
        )
    if not res.get("encounter"):
        out.add(
            Layer.PROFILE,
            "missing-encounter",
            _label(res),
            f"{base}.encounter",
            "report is not linked to an encounter",
            IssueSeverity.WARNING,
        )
    for i, form in enumerate(res.get("presentedForm") or []):
        _p_attachment(res, form, f"{base}.presentedForm[{i}]", out)


def _p_observation(res: Mapping[str, Any], base: str, out: _Collector) -> None:
    _check_binding(res, "status", "observation-status", base, out)
    if not any(k.startswith("value") for k in res) and not res.get("dataAbsentReason"):
        out.add(
            Layer.PROFILE,
            "missing-value",
            _label(res),
            f"{base}.value[x]",
            "Observation needs a value or a dataAbsentReason",
        )
    code = res.get("code") or {}
    if not code.get("coding"):
        out.add(
            Layer.PROFILE,
            "uncoded-observation",
            _label(res),
            f"{base}.code",
            "no standard code (LOINC) for this test; text only",
            IssueSeverity.WARNING,
        )


_PROFILE_RULES: dict[str, Callable[[Mapping[str, Any], str, _Collector], None]] = {
    "Patient": _p_patient,
    "Encounter": _p_encounter,
    "DocumentReference": _p_docref,
    "DiagnosticReport": _p_report,
    "Observation": _p_observation,
}


def _validate_profile(bundle: Mapping[str, Any], out: _Collector) -> None:
    btype = bundle.get("type")
    if btype not in VALUE_SETS["bundle-type"]:
        out.add(
            Layer.PROFILE,
            "invalid-code",
            "Bundle",
            "type",
            f"{btype!r} is not in the required value set 'bundle-type'",
        )
    for i, entry in enumerate(bundle.get("entry") or []):
        res = entry.get("resource")
        if not isinstance(res, Mapping):
            continue
        base = f"entry[{i}].resource"
        if rule := _PROFILE_RULES.get(str(res.get("resourceType"))):
            rule(res, base, out)
        for path, coding in _walk_codings(res, ""):
            if coding.get("system") not in ALLOWED_CODE_SYSTEMS:
                out.add(
                    Layer.PROFILE,
                    "unknown-code-system",
                    _label(res),
                    f"{base}.{path}",
                    f"code system {coding.get('system')!r} is not in the allowed set",
                    IssueSeverity.WARNING,
                )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def validate_bundle(bundle: Mapping[str, Any]) -> BundleValidation:
    """Validate one Bundle through all three layers. Never raises on bad input."""
    out = _Collector()
    if not isinstance(bundle, Mapping) or bundle.get("resourceType") != "Bundle":
        out.add(
            Layer.SCHEMA, "not-a-bundle", "Bundle", "resourceType", "document is not a FHIR Bundle"
        )
        return BundleValidation(valid=False, issues=out.issues)

    schema_ok = _validate_schema(bundle, out)
    # Reference and profile checks are meaningful even when some resources are malformed, but
    # they assume mapping-shaped entries, which _validate_schema has already reported on.
    try:
        _validate_references(bundle, out)
        _validate_profile(bundle, out)
    except (AttributeError, TypeError) as exc:  # defensive: malformed structure
        out.add(
            Layer.SCHEMA,
            "malformed-structure",
            "Bundle",
            "entry",
            f"could not run structural checks: {exc}",
        )
        schema_ok = False

    resources = [
        e["resource"]
        for e in bundle.get("entry") or []
        if isinstance(e, Mapping) and isinstance(e.get("resource"), Mapping)
    ]
    patient = next((r for r in resources if r.get("resourceType") == "Patient"), None)
    mrn = next(
        (
            i.get("value")
            for i in (patient or {}).get("identifier", [])
            if (i.get("type") or {}).get("coding", [{}])[0].get("code") == "MR"
        ),
        None,
    )
    return BundleValidation(
        patient_id=patient.get("id") if patient else None,
        mrn=mrn,
        valid=schema_ok and not any(i.severity is IssueSeverity.ERROR for i in out.issues),
        resource_counts=dict(Counter(str(r.get("resourceType")) for r in resources)),
        issues=out.issues,
    )
