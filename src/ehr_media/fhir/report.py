"""Aggregate validation results into a clear JSON + Markdown report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from ehr_media.fhir.validation import BundleValidation, IssueSeverity

REPORT_JSON = "validation_report.json"
REPORT_MD = "validation_report.md"
MAX_ROWS_PER_BUNDLE = 25


class ValidationReport(BaseModel):
    bundles_total: int
    bundles_valid: int
    bundles_invalid: int
    total_errors: int
    total_warnings: int
    resource_counts: dict[str, int]
    issues_by_code: dict[str, int]
    issues_by_layer: dict[str, int]
    bundles: list[BundleValidation]  # only bundles that have at least one issue


def build_report(validations: Iterable[BundleValidation]) -> ValidationReport:
    items = list(validations)
    issues = [i for v in items for i in v.issues]
    resources: Counter[str] = Counter()
    for v in items:
        resources.update(v.resource_counts)
    return ValidationReport(
        bundles_total=len(items),
        bundles_valid=sum(v.valid for v in items),
        bundles_invalid=sum(not v.valid for v in items),
        total_errors=sum(i.severity is IssueSeverity.ERROR for i in issues),
        total_warnings=sum(i.severity is IssueSeverity.WARNING for i in issues),
        resource_counts=dict(sorted(resources.items())),
        issues_by_code=dict(Counter(f"{i.severity.value}:{i.code}" for i in issues).most_common()),
        issues_by_layer=dict(Counter(i.layer.value for i in issues).most_common()),
        bundles=[v for v in items if v.issues],
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: ValidationReport) -> str:
    verdict = "PASS" if report.bundles_invalid == 0 else "FAIL"
    lines = [
        "# FHIR R4 validation report",
        "",
        f"**Result: {verdict}** - {report.bundles_valid}/{report.bundles_total} bundles valid, "
        f"{report.total_errors} error(s), {report.total_warnings} warning(s).",
        "",
        "Layers: `schema` (R4B models), `reference` (in-bundle integrity), `profile` (bindings "
        "and required elements). See the README for what this does *not* cover.",
        "",
        "## Resources validated",
        "",
        "| Resource | Count |",
        "|---|---:|",
        *(f"| {k} | {v} |" for k, v in report.resource_counts.items()),
        "",
    ]
    if report.issues_by_code:
        lines += ["## Issues by type", "", "| Severity:code | Count |", "|---|---:|"]
        lines += [f"| `{k}` | {v} |" for k, v in report.issues_by_code.items()]
        lines.append("")
    if report.bundles:
        lines += ["## Details", ""]
    for v in report.bundles:
        status = "valid (warnings only)" if v.valid else "INVALID"
        lines += [
            f"### MRN {v.mrn or '?'} - {status}",
            "",
            "| Layer | Severity | Resource | Path | Message |",
            "|---|---|---|---|---|",
        ]
        ordered = sorted(v.issues, key=lambda i: (i.severity.value, i.layer.value, i.path))
        for i in ordered[:MAX_ROWS_PER_BUNDLE]:
            lines.append(
                f"| {i.layer.value} | {i.severity.value} | `{_cell(i.resource)}` | "
                f"`{_cell(i.path)}` | {_cell(i.message)} |"
            )
        if len(ordered) > MAX_ROWS_PER_BUNDLE:
            lines.append(f"| | | | | ... {len(ordered) - MAX_ROWS_PER_BUNDLE} more |")
        lines.append("")
    return "\n".join(lines)


def write_report(report: ValidationReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / REPORT_JSON, out_dir / REPORT_MD
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
