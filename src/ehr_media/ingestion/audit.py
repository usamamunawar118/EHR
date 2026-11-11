"""Per-record audit trail collector."""

from __future__ import annotations

from ehr_media.ingestion.models import AuditAction, AuditEntry, RecordType, Severity


class AuditTrail:
    """Append-only collector. Every change or decision made to a record goes through here."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def add(
        self,
        *,
        record_id: str,
        record_type: RecordType,
        mrn: str | None,
        field: str,
        rule: str,
        action: AuditAction,
        before: object = None,
        after: object = None,
        severity: Severity = Severity.INFO,
        message: str = "",
    ) -> None:
        self._entries.append(
            AuditEntry(
                record_id=record_id,
                record_type=record_type,
                mrn=mrn,
                field=field,
                rule=rule,
                action=action,
                before=None if before is None else str(before),
                after=None if after is None else str(after),
                severity=severity,
                message=message,
            )
        )

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def for_record(self, record_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.record_id == record_id]

    def __len__(self) -> int:
        return len(self._entries)
