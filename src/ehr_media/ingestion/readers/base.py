"""Reader abstraction: every input format yields format-agnostic :class:`RawRecord` objects."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from ehr_media.ingestion.models import RecordType, SourceRef


class UnsupportedInputError(ValueError):
    """The file looks like a supported extension but its structure is not understood."""


@dataclass(frozen=True)
class RawRecord:
    """An un-normalised record. ``fields`` uses reader-independent keys; values are raw strings.

    Keys by record type:

    * ``PATIENT``   - mrn, family, given, full_name, dob, sex
    * ``ENCOUNTER`` - mrn, encounter_id, setting, start, end, reason
    * ``DOCUMENT``  - mrn, doc_id, doc_type, date, encounter_id, text
    * ``LAB``       - mrn, collected, test, result, unit, ref_range, flag, encounter_id
    """

    record_type: RecordType
    source: SourceRef
    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        return f"{self.record_type.value}:{self.source.key}"


class Reader(Protocol):
    """Plug-in point for new input formats (register in :data:`readers.READERS`)."""

    format: ClassVar[str]
    suffixes: ClassVar[tuple[str, ...]]

    def read(self, path: Path) -> Iterator[RawRecord]: ...


def as_text(value: object) -> str | None:
    """Coerce a scalar JSON/CSV value to ``str`` (``None`` stays ``None``)."""
    if value is None:
        return None
    return str(value)
