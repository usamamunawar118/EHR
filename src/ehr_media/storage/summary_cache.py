"""SQLite cache for AI summaries.

Key = (patient_id, record_hash, prompt_version, model). Re-running the pipeline on unchanged
records makes **zero** LLM calls; changing a record, the prompt wording or the model misses.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ehr_media.storage.db import Database
from ehr_media.summarization.models import ClinicalSummary


class SummaryCache:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get(
        self, patient_id: str, record_hash: str, prompt_version: str, model: str
    ) -> ClinicalSummary | None:
        with self._db.connection() as conn:
            row = conn.execute(
                """SELECT summary_json FROM summary_cache
                   WHERE patient_id = ? AND record_hash = ? AND prompt_version = ? AND model = ?""",
                (patient_id, record_hash, prompt_version, model),
            ).fetchone()
        return ClinicalSummary.model_validate_json(row["summary_json"]) if row else None

    def put(self, summary: ClinicalSummary) -> None:
        gen = summary.generated_by
        with self._db.connection() as conn:
            conn.execute(
                """INSERT INTO summary_cache
                       (patient_id, record_hash, prompt_version, model, summary_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(patient_id, record_hash, prompt_version, model)
                   DO UPDATE SET summary_json = excluded.summary_json,
                                 created_at = excluded.created_at""",
                (
                    summary.patient_id,
                    summary.record_hash,
                    gen.prompt_version,
                    gen.model,
                    summary.model_dump_json(),
                    datetime.now(UTC).isoformat(timespec="microseconds"),
                ),
            )

    def latest_for(self, patient_id: str, record_hash: str) -> ClinicalSummary | None:
        """Most recent summary of the *current* records, whichever model/prompt wrote it."""
        with self._db.connection() as conn:
            row = conn.execute(
                """SELECT summary_json FROM summary_cache
                   WHERE patient_id = ? AND record_hash = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (patient_id, record_hash),
            ).fetchone()
        return ClinicalSummary.model_validate_json(row["summary_json"]) if row else None

    def count(self) -> int:
        with self._db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM summary_cache").fetchone()[0])
