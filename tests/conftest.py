"""Shared fixtures."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ehr_media.api.deps import Container, build_container
from ehr_media.config import Settings
from ehr_media.fhir.builder import PatientBundle, build_patient_bundle
from ehr_media.fhir.service import build_and_validate, store_bundles
from ehr_media.ingestion.models import IngestResult, PatientRecord
from ehr_media.ingestion.pipeline import run_ingest
from ehr_media.search.embedder import HashingEmbedder
from ehr_media.search.indexer import Indexer
from ehr_media.summarization.llm import FakeSummaryLLM
from ehr_media.summarization.service import SummaryService
from ehr_media.synthetic import generate_dataset
from tests.helpers import IMAGING_TEXT, doc_a, enc_a, patient_a, write_export

# Fixed so that results never depend on the day the suite runs.
REFERENCE_DATE = date(2026, 11, 11)


@pytest.fixture(scope="session")
def reference_date() -> date:
    return REFERENCE_DATE


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The full synthetic dataset generated once per test session."""
    root = tmp_path_factory.mktemp("synthetic")
    generate_dataset(root / "raw")
    return root


@pytest.fixture(scope="session")
def manifest(synthetic_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((synthetic_dir / "DEFECT_MANIFEST.json").read_text("utf-8"))
    return data


@pytest.fixture(scope="session")
def ingest_result(synthetic_dir: Path) -> IngestResult:
    return run_ingest([synthetic_dir / "raw"], reference_date=REFERENCE_DATE)


# --- full search environment over the synthetic dataset (fast: hashing embedder) -------


@pytest.fixture(scope="session")
def search_env(ingest_result: IngestResult, tmp_path_factory: pytest.TempPathFactory) -> Container:
    """All 60 patients stored, summarised (offline provider) and indexed. Treat as read-only."""
    root = tmp_path_factory.mktemp("searchenv")
    settings = Settings(
        data_dir=root / "data",
        frontend_dir=root / "no-frontend",
        embedding_backend="hashing",
        llm_provider="fake",
    )
    container = build_container(settings, embedder=HashingEmbedder())
    store_bundles(build_and_validate(ingest_result.patients), container.repo)
    SummaryService(FakeSummaryLLM(), container.summaries).summarize_all(
        list(container.repo.iter_bundles())
    )
    Indexer(
        container.repo,
        container.summaries,
        container.embedder,
        container.store,
        chunk_size=settings.chunk_words,
        chunk_overlap=settings.chunk_overlap_words,
    ).index_all()
    return container


# --- a small, fully-featured patient built through the real pipeline -----------------


@pytest.fixture
def record(tmp_path: Path) -> PatientRecord:
    write_export(
        tmp_path,
        [
            patient_a(
                "00000042",
                family="Fixture",
                given="Fay",
                dob="1960-02-03",
                sex="F",
                encounters=[enc_a("E1", "2025-03-01", "2025-03-04", "Inpatient")],
                documents=[
                    doc_a(
                        "d1",
                        date_="2025-03-01",
                        encounter_id="E1",
                        text="ASSESSMENT: Community-acquired pneumonia (J18.9). PLAN: antibiotics.",
                    ),
                    doc_a(
                        "d2",
                        date_="2025-03-01",
                        encounter_id="E1",
                        type_="Radiology Report",
                        text=IMAGING_TEXT,
                    ),
                    doc_a(
                        "d3",
                        date_="2025-03-04",
                        encounter_id="E1",
                        type_="Discharge Summary",
                        text="DISCHARGE SUMMARY\nHOSPITAL COURSE: improved.",
                    ),
                ],
            )
        ],
    )
    (tmp_path / "labs.csv").write_text(
        "MRN,PatientName,DOB,CollectedDate,TestName,Result,Units,RefRange,Flag,EncounterID\n"
        '42,"FIXTURE, FAY",,2025-03-01,WBC,15.2,10*3/uL,4.5-11,H,E1\n'
        '42,"FIXTURE, FAY",,2025-03-01,Sodium,140,mmol/L,135-145,N,E1\n'
        '42,"FIXTURE, FAY",,2025-03-01,Mystery Marker,3.3,widgets,,,E1\n',
        encoding="utf-8",
    )
    result = run_ingest([tmp_path], reference_date=date(2026, 11, 11))
    assert len(result.patients) == 1
    return result.patients[0]


@pytest.fixture
def patient_bundle(record: PatientRecord) -> PatientBundle:
    return build_patient_bundle(record)


@pytest.fixture
def bundle(patient_bundle: PatientBundle) -> dict[str, Any]:
    """A fresh deep copy per test so tests can corrupt it freely."""
    return copy.deepcopy(patient_bundle.bundle)
