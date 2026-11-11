"""SQLite access: connection factory and versioned schema migrations.

Stdlib ``sqlite3`` behind small repository classes keeps the dependency surface minimal for a
handful of tables. Every query is parameterised. WAL mode allows the API to read while the
pipeline writes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Migration N is applied when PRAGMA user_version < N. Append only - never edit a shipped one.
MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE patients (
        patient_id   TEXT PRIMARY KEY,
        mrn          TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        family_name  TEXT,
        given_name   TEXT,
        birth_date   TEXT,
        gender       TEXT NOT NULL,
        flags        TEXT NOT NULL DEFAULT '[]'
    );

    CREATE TABLE fhir_bundles (
        patient_id        TEXT PRIMARY KEY REFERENCES patients(patient_id) ON DELETE CASCADE,
        bundle_json       TEXT NOT NULL,
        content_hash      TEXT NOT NULL,
        validation_status TEXT NOT NULL CHECK (validation_status IN ('valid', 'invalid')),
        validation_json   TEXT NOT NULL,
        updated_at        TEXT NOT NULL
    );

    CREATE TABLE fhir_resources (
        resource_id    TEXT PRIMARY KEY,
        patient_id     TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
        resource_type  TEXT NOT NULL,
        effective_date TEXT,
        doc_key        TEXT,
        display        TEXT NOT NULL,
        text           TEXT,
        json           TEXT NOT NULL
    );
    CREATE INDEX idx_resources_patient ON fhir_resources(patient_id);
    CREATE INDEX idx_resources_type_date ON fhir_resources(resource_type, effective_date);
    """,
    # 2: AI summary cache. Keyed by patient + content hash of the bundle + prompt version + model
    # so editing records, the prompt or the model can never serve a stale summary.
    """
    CREATE TABLE summary_cache (
        patient_id     TEXT NOT NULL,
        record_hash    TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        model          TEXT NOT NULL,
        summary_json   TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        PRIMARY KEY (patient_id, record_hash, prompt_version, model)
    );
    CREATE INDEX idx_summary_patient ON summary_cache(patient_id, created_at);
    """,
)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with sane defaults (rows as mappings, FK enforcement, WAL)."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations; return the resulting schema version."""
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    for number, script in enumerate(MIGRATIONS, start=1):
        if number <= version:
            continue
        with conn:  # one transaction per migration
            conn.executescript(script)
            conn.execute(f"PRAGMA user_version = {number}")
        version = number
    return version


class Database:
    """Path-bound handle that hands out short-lived, migrated connections.

    A connection per unit of work keeps SQLite safe across FastAPI's threadpool and the
    summariser's worker threads (connections are not shared between threads).
    """

    def __init__(self, path: Path | str) -> None:
        self.path = path
        with self.connection() as conn:
            migrate(conn)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.path)
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
