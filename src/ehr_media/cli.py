"""Command line interface: one idempotent command per pipeline stage.

ehr generate-data    create the synthetic raw inputs
ehr ingest           Task 1 - clean raw inputs into the canonical IR + audit log
ehr build-fhir       Task 2 - map to FHIR R4 Bundles, validate, report, store (SQLite)
ehr validate         Task 2 - (re)validate stored Bundles or a single Bundle file
ehr summarize        Task 3 - AI summaries (cached by patient + record hash + prompt + model)
ehr eval-summaries   Task 3 - score stored summaries (word limit, grounding, recall)
ehr index            Task 4 - embed documents + summaries into the vector store
ehr serve            Task 4/5 - run the API and web UI
ehr run-all          every stage above, in order
ehr download-models  fetch the embedding model once
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


@app.command()
def summarize(
    limit: Annotated[int | None, typer.Option(help="Only summarise the first N patients.")] = None,
    force: Annotated[bool, typer.Option(help="Ignore the cache and call the LLM again.")] = False,
    provider: Annotated[
        str | None, typer.Option(help="Override EHR_LLM_PROVIDER: auto | anthropic | fake.")
    ] = None,
) -> None:
    """Task 3: generate (or reuse cached) AI summaries for every stored patient."""
    from ehr_media.storage.bundle_repo import BundleRepository
    from ehr_media.storage.db import Database
    from ehr_media.storage.summary_cache import SummaryCache
    from ehr_media.summarization.llm import create_llm
    from ehr_media.summarization.service import SummaryService

    settings = get_settings()
    if provider:
        if provider not in ("auto", "anthropic", "fake"):
            raise typer.BadParameter("provider must be auto, anthropic or fake")
        settings = settings.model_copy(update={"llm_provider": provider})
    db = Database(settings.db_path)
    repo = BundleRepository(db)
    bundles = list(repo.iter_bundles())
    if not bundles:
        raise typer.BadParameter("no bundles stored; run `ehr build-fhir` first.")

    llm = create_llm(settings)
    service = SummaryService(llm, SummaryCache(db))
    result = service.summarize_all(
        bundles, concurrency=settings.summary_concurrency, limit=limit, force=force
    )
    typer.echo(
        f"provider={llm.provider} model={llm.model}: generated={result.generated} "
        f"cache_hits={result.cache_hits} failed={len(result.failures)}"
    )
    for patient_id, reason in list(result.failures.items())[:10]:
        typer.echo(f"  FAILED {patient_id}: {reason}")
    if result.failures and not result.summaries:
        raise typer.Exit(code=1)


@app.command("eval-summaries")
def eval_summaries() -> None:
    """Score stored summaries against the source data (word limit, grounding, recall)."""
    from ehr_media.storage.bundle_repo import BundleRepository
    from ehr_media.storage.db import Database
    from ehr_media.storage.summary_cache import SummaryCache
    from ehr_media.summarization.evaluation import evaluate

    settings = get_settings()
    db = Database(settings.db_path)
    repo, cache = BundleRepository(db), SummaryCache(db)
    pairs = []
    for stored in repo.iter_bundles():
        summary = cache.latest_for(stored.patient.patient_id, stored.content_hash)
        if summary is not None:
            pairs.append((summary, stored))
    if not pairs:
        raise typer.BadParameter("no summaries found; run `ehr summarize` first.")
    report = evaluate(pairs)
    settings.ensure_dirs()
    out = settings.reports_dir / "summary_eval.json"
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(f"report: {out}")
    if not report.passes_hard_requirements():
        raise typer.Exit(code=1)


@app.command("download-models")
def download_models() -> None:
    """Pre-download the sentence-transformer model (needs network once; then works offline)."""
    from ehr_media.search.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(get_settings().embedding_model)
    typer.echo(f"ready: {embedder.name} ({embedder.dimension} dimensions)")


@app.command()
def index(
    backend: Annotated[
        str | None,
        typer.Option(help="Override EHR_EMBEDDING_BACKEND: sentence-transformers|hashing."),
    ] = None,
) -> None:
    """Task 4: embed FHIR documents and AI summaries into the Chroma vector store."""
    from ehr_media.search.embedder import create_embedder
    from ehr_media.search.indexer import Indexer
    from ehr_media.search.vector_store import ChromaVectorStore
    from ehr_media.storage.bundle_repo import BundleRepository
    from ehr_media.storage.db import Database
    from ehr_media.storage.summary_cache import SummaryCache

    settings = get_settings()
    if backend:
        if backend not in ("sentence-transformers", "hashing"):
            raise typer.BadParameter("backend must be sentence-transformers or hashing")
        settings = settings.model_copy(update={"embedding_backend": backend})
    db = Database(settings.db_path)
    repo = BundleRepository(db)
    if repo.count()["patients"] == 0:
        raise typer.BadParameter("no bundles stored; run `ehr build-fhir` first.")

    embedder = create_embedder(settings)
    stats = Indexer(
        repo,
        SummaryCache(db),
        embedder,
        ChromaVectorStore(settings.chroma_dir),
        chunk_size=settings.chunk_words,
        chunk_overlap=settings.chunk_overlap_words,
    ).index_all()
    typer.echo(
        f"embedder={embedder.name}: embedded={stats.embedded} unchanged={stats.unchanged} "
        f"deleted={stats.deleted} total_in_index={stats.total_in_store}"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev).")] = False,
) -> None:
    """Serve the API and the web UI."""
    import uvicorn

    typer.echo(f"EHR Media Intelligence -> http://{host}:{port}  (API docs: /docs)")
    uvicorn.run("ehr_media.api.app:create_app", factory=True, host=host, port=port, reload=reload)


@app.command("run-all")
def run_all(
    provider: Annotated[
        str | None, typer.Option(help="LLM provider override: auto | anthropic | fake.")
    ] = None,
    reference_date: Annotated[str | None, typer.Option(help="ISO date for ingest checks.")] = None,
) -> None:
    """Run every stage: generate (if missing), ingest, build-fhir, summarize, index."""
    settings = get_settings()
    if not settings.raw_dir.exists():
        generate_data(out=None, seed=20250416)
    ingest(input_dir=None, out=None, reference_date=reference_date)
    build_fhir(strict=True)
    summarize(limit=None, force=False, provider=provider)
    index(backend=None)
    typer.echo("\nAll stages complete. Start the app with:  ehr serve")


if __name__ == "__main__":  # pragma: no cover
    app()
