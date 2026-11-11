"""Persistence of ingestion outputs as JSON Lines (human-diffable, stream-friendly)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ehr_media.ingestion.models import IngestResult, PatientRecord

CLEAN_FILE = "clean_records.jsonl"
AUDIT_FILE = "audit_log.jsonl"
QUARANTINE_FILE = "quarantine.jsonl"
SUMMARY_FILE = "ingest_summary.json"


def write_ingest_outputs(result: IngestResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "clean": out_dir / CLEAN_FILE,
        "audit": out_dir / AUDIT_FILE,
        "quarantine": out_dir / QUARANTINE_FILE,
        "summary": out_dir / SUMMARY_FILE,
    }
    with paths["clean"].open("w", encoding="utf-8") as fh:
        for record in result.patients:
            fh.write(record.model_dump_json() + "\n")

    # One line per *record* with every change made to it (the per-record audit log).
    with paths["audit"].open("w", encoding="utf-8") as fh:
        for record_id, entries in sorted(result.audit_by_record().items()):
            fh.write(
                json.dumps(
                    {
                        "record_id": record_id,
                        "record_type": entries[0].record_type.value,
                        "mrn": entries[0].mrn,
                        "entries": [
                            e.model_dump(mode="json", exclude={"record_id", "record_type"})
                            for e in entries
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with paths["quarantine"].open("w", encoding="utf-8") as fh:
        for q in result.quarantine:
            fh.write(q.model_dump_json() + "\n")

    summary = {
        "stats": result.stats,
        "audit_by_rule": dict(Counter(e.rule for e in result.audit).most_common()),
        "audit_by_action": dict(Counter(e.action.value for e in result.audit).most_common()),
        "quarantine_by_reason": dict(Counter(q.reason for q in result.quarantine).most_common()),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def iter_clean_records(path: Path) -> Iterator[PatientRecord]:
    """Stream ``PatientRecord`` objects back from ``clean_records.jsonl``."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield PatientRecord.model_validate_json(line)
