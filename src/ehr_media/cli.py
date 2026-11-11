"""Command line interface: one idempotent command per pipeline stage.

ehr generate-data    create the synthetic raw inputs
ehr ingest           Task 1 - clean raw inputs into the canonical IR + audit log
ehr build-fhir       Task 2 - map to FHIR R4 Bundles, validate, report, store (SQLite)
ehr validate         Task 2 - (re)validate stored Bundles or a single Bundle file
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from ehr_media.config import get_settings
from ehr_media.logging import configure_logging

if TYPE_CHECKING:
    from ehr_media.fhir.report import ValidationReport

app = typer.Typer(
    name="ehr",
    no_args_is_help=True,
    add_completion=False,
    help="EHR Media Intelligence Platform - synthetic data only, not for clinical use.",
)
logger = logging.getLogger("ehr_media.cli")


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    configure_logging(logging.DEBUG if verbose else logging.INFO)


@app.command("generate-data")
def generate_data(
    out: Annotated[Path | None, typer.Option(help="Output dir (default: data/sample/raw).")] = None,
    seed: Annotated[int, typer.Option(help="RNG seed (deterministic output).")] = 20250416,
) -> None:
    """Generate the messy, fictional EHR inputs and the defect manifest."""
    from ehr_media.synthetic import generate_dataset

    target = out or get_settings().raw_dir
    manifest = generate_dataset(target, seed=seed)
    typer.echo(f"Wrote synthetic data to {target}")
    typer.echo(json.dumps(manifest.totals, indent=2))


@app.command()
def ingest(
    input_dir: Annotated[Path | None, typer.Option("--input", help="Raw inputs dir.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output dir for clean/audit/quarantine.")] = None,
    reference_date: Annotated[
        str | None,
        typer.Option(
            help="ISO date used for 'future date' / 2-digit-year checks (default: today)."
        ),
    ] = None,
) -> None:
    """Task 1: ingest, clean and normalise raw EHR records."""
    from ehr_media.ingestion.io import write_ingest_outputs
    from ehr_media.ingestion.pipeline import run_ingest

    settings = get_settings()
    source = input_dir or settings.raw_dir
    if not source.exists():
        raise typer.BadParameter(f"{source} does not exist; run `ehr generate-data` first.")
    target = out or settings.ingest_dir
    result = run_ingest(
        [source], reference_date=date.fromisoformat(reference_date) if reference_date else None
    )
    paths = write_ingest_outputs(result, target)
    typer.echo(json.dumps(result.stats, indent=2))
    for name, path in paths.items():
        typer.echo(f"  {name:<10} {path}")


def _echo_validation(report_dir: Path, report: ValidationReport) -> None:
    from ehr_media.fhir.report import write_report

    json_path, md_path = write_report(report, report_dir)
    verdict = "PASS" if report.bundles_invalid == 0 else "FAIL"
    typer.echo(
        f"Validation {verdict}: {report.bundles_valid}/{report.bundles_total} bundles valid, "
        f"{report.total_errors} error(s), {report.total_warnings} warning(s)"
    )
    for code, count in report.issues_by_code.items():
        typer.echo(f"  {count:>4}  {code}")
    typer.echo(f"  report: {md_path}\n          {json_path}")


@app.command("build-fhir")
def build_fhir(
    strict: Annotated[bool, typer.Option(help="Exit non-zero if any bundle is invalid.")] = False,
) -> None:
    """Task 2: map cleaned records to FHIR R4 Bundles, validate, report and store them."""
    from ehr_media.fhir.service import build_and_validate, store_bundles
    from ehr_media.ingestion.io import CLEAN_FILE, iter_clean_records
    from ehr_media.storage.bundle_repo import BundleRepository
    from ehr_media.storage.db import Database

    settings = get_settings()
    clean = settings.ingest_dir / CLEAN_FILE
    if not clean.exists():
        raise typer.BadParameter(f"{clean} not found; run `ehr ingest` first.")
    result = build_and_validate(iter_clean_records(clean))
    repo = BundleRepository(Database(settings.db_path))
    stored = store_bundles(result, repo)
    typer.echo(f"Stored {stored} bundles in {settings.db_path}  {repo.count()}")
    _echo_validation(settings.reports_dir, result.report)
    if strict and result.report.bundles_invalid:
        raise typer.Exit(code=1)


@app.command()
def validate(
    bundle: Annotated[
        Path | None, typer.Option(help="Validate this Bundle JSON file instead of the store.")
    ] = None,
    strict: Annotated[bool, typer.Option(help="Exit non-zero if any bundle is invalid.")] = False,
) -> None:
    """Validate stored Bundles (or one file) against the three FHIR validation layers."""
    from ehr_media.fhir.report import build_report
    from ehr_media.fhir.validation import validate_bundle
    from ehr_media.storage.bundle_repo import BundleRepository
    from ehr_media.storage.db import Database

    settings = get_settings()
    if bundle is not None:
        validations = [validate_bundle(json.loads(bundle.read_text(encoding="utf-8")))]
    else:
        repo = BundleRepository(Database(settings.db_path))
        validations = [validate_bundle(s.bundle) for s in repo.iter_bundles()]
        if not validations:
            raise typer.BadParameter("no bundles stored; run `ehr build-fhir` first.")
    report = build_report(validations)
    _echo_validation(settings.reports_dir, report)
    if strict and report.bundles_invalid:
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
