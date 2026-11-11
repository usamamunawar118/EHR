"""Ground-truth check: every defect the generator seeded must be detected by the pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ehr_media.ingestion.io import iter_clean_records, write_ingest_outputs
from ehr_media.ingestion.models import IngestResult


def test_every_seeded_defect_is_in_the_audit_log(
    ingest_result: IngestResult, manifest: dict[str, Any]
) -> None:
    present = {(e.rule, e.mrn, e.record_type.value) for e in ingest_result.audit}
    missing = [
        d for d in manifest["expected"] if (d["rule"], d["mrn"], d["record_type"]) not in present
    ]
    assert not missing, f"seeded defects not detected: {missing}"


def test_generic_noise_is_audited_at_expected_volume(
    ingest_result: IngestResult, manifest: dict[str, Any]
) -> None:
    counts = Counter(e.rule for e in ingest_result.audit)
    short = {
        rule: (counts[rule], want)
        for rule, want in manifest["expected_min_counts"].items()
        if counts[rule] < want
    }
    assert not short, f"rule -> (found, wanted): {short}"


def test_conflicting_mrns_are_fully_quarantined_and_absent_from_clean_set(
    ingest_result: IngestResult, manifest: dict[str, Any]
) -> None:
    conflicted = {d["mrn"] for d in manifest["expected"] if d["rule"] == "IDENTIFIER_CONFLICT"}
    assert conflicted
    assert conflicted.isdisjoint({r.patient.mrn for r in ingest_result.patients})
    quarantined = {q.mrn for q in ingest_result.quarantine if q.reason == "IDENTIFIER_CONFLICT"}
    assert conflicted <= quarantined


def test_clean_output_satisfies_invariants(ingest_result: IngestResult) -> None:
    assert ingest_result.patients
    for rec in ingest_result.patients:
        assert rec.patient.mrn.isdigit()
        assert len(rec.patient.mrn) == 8
        encounter_ids = {e.encounter_id for e in rec.encounters}
        for doc in rec.documents:
            assert doc.mrn == rec.patient.mrn
            assert doc.encounter_id in encounter_ids  # references will resolve in FHIR
        for lab in rec.lab_results:
            assert lab.encounter_id in encounter_ids
        # No two surviving documents are the same note.
        keys = [
            (d.kind, d.document_date, " ".join(d.text.split()).casefold()) for d in rec.documents
        ]
        assert len(keys) == len(set(keys))


def test_no_record_is_silently_dropped(ingest_result: IngestResult) -> None:
    """raw = clean + quarantined + deduplicated + claims folded into patients."""
    stats = ingest_result.stats
    dedup = sum(1 for e in ingest_result.audit if e.action.value == "deduplicated")
    assert stats["quarantined"] > 0
    assert stats["raw_records"] >= stats["documents"] + stats["lab_results"] + dedup


def test_outputs_round_trip_through_jsonl(ingest_result: IngestResult, tmp_path: Path) -> None:
    paths = write_ingest_outputs(ingest_result, tmp_path)
    reloaded = list(iter_clean_records(paths["clean"]))
    assert [r.model_dump() for r in reloaded] == [r.model_dump() for r in ingest_result.patients]

    audit_lines = [json.loads(line) for line in paths["audit"].read_text("utf-8").splitlines()]
    assert sum(len(line["entries"]) for line in audit_lines) == len(ingest_result.audit)
    assert all({"record_id", "record_type", "entries"} <= line.keys() for line in audit_lines)


@pytest.mark.parametrize("rule", ["DOB_MISSING", "POSSIBLE_DUPLICATE_PATIENT"])
def test_patient_flags_are_visible_on_the_patient_record(
    ingest_result: IngestResult, rule: str
) -> None:
    assert any(rule in r.patient.flags for r in ingest_result.patients)
