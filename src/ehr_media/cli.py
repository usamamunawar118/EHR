"""Command line interface: one idempotent command per pipeline stage.

ehr generate-data    create the synthetic raw inputs
ehr ingest           Task 1 - clean raw inputs into the canonical IR + audit log
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from ehr_media.config import get_settings
from ehr_media.logging import configure_logging

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


if __name__ == "__main__":  # pragma: no cover
    app()
