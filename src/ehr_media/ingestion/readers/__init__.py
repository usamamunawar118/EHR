"""Reader registry. To support a new format, implement ``Reader`` and add it to ``READERS``."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from ehr_media.ingestion.readers.base import RawRecord, Reader, UnsupportedInputError
from ehr_media.ingestion.readers.csv_reader import LabCsvReader
from ehr_media.ingestion.readers.json_reader import JsonExportReader
from ehr_media.ingestion.readers.text_reader import ScannedNoteReader

logger = logging.getLogger(__name__)

READERS: tuple[Reader, ...] = (JsonExportReader(), LabCsvReader(), ScannedNoteReader())


def discover_inputs(root: Path) -> list[Path]:
    """All files under ``root`` (or ``root`` itself) that a registered reader can handle."""
    suffixes = {s for r in READERS for s in r.suffixes}
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def reader_for(path: Path) -> Reader | None:
    return next((r for r in READERS if path.suffix.lower() in r.suffixes), None)


def read_all(paths: Iterable[Path]) -> Iterator[RawRecord]:
    """Yield raw records from every path. Unreadable files are logged and skipped, not fatal."""
    for path in paths:
        reader = reader_for(path)
        if reader is None:
            logger.warning("no reader for %s; skipping", path.name)
            continue
        try:
            yield from reader.read(path)
        except UnsupportedInputError as exc:
            logger.warning("skipping unsupported input: %s", exc)


__all__ = [
    "READERS",
    "RawRecord",
    "Reader",
    "UnsupportedInputError",
    "discover_inputs",
    "read_all",
    "reader_for",
]
