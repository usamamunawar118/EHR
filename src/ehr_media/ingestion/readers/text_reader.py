"""Reader for OCR-style scanned clinical notes (plain text with a header block).

Expected layout::

    *** SCANNED DOCUMENT - OCR CONFIDENCE 0.91 ***
    PATIENT NAME: SMITH, JOHN
    MRN: 00012345
    DOB: 03/14/1956    SEX: M
    DATE OF SERVICE: 03/02/2025
    DOCUMENT TYPE: PROGRESS NOTE
    ENCOUNTER: E123456            (optional)
    ------------------------------------------------------------
    <free text body>
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from ehr_media.ingestion.models import RecordType, SourceRef
from ehr_media.ingestion.readers.base import RawRecord, UnsupportedInputError

_SEPARATOR = re.compile(r"^-{10,}\s*$")
_KV = re.compile(r"^([A-Z][A-Z ]+?):[ \t]*(.*)$")
_DOB_SEX = re.compile(r"^DOB:[ \t]*(.*?)(?:[ \t]{2,}SEX:[ \t]*(.*))?$")


class ScannedNoteReader:
    format: ClassVar[str] = "txt"
    suffixes: ClassVar[tuple[str, ...]] = (".txt",)

    def read(self, path: Path) -> Iterator[RawRecord]:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        try:
            sep = next(i for i, line in enumerate(lines) if _SEPARATOR.match(line))
        except StopIteration:
            raise UnsupportedInputError(f"{path.name}: no header/body separator line") from None

        header: dict[str, str] = {}
        for line in lines[:sep]:
            if dob := _DOB_SEX.match(line):
                header["DOB"] = dob[1] or ""
                header["SEX"] = dob[2] or ""
            elif kv := _KV.match(line):
                header[kv[1].strip()] = kv[2].strip()
        body = "\n".join(lines[sep + 1 :])

        source = SourceRef(file=path.name, format="txt", locator="document")
        mrn = header.get("MRN")
        yield RawRecord(
            RecordType.PATIENT,
            SourceRef(file=path.name, format="txt", locator="header"),
            {
                "mrn": mrn,
                "full_name": header.get("PATIENT NAME"),
                "dob": header.get("DOB"),
                "sex": header.get("SEX"),
            },
        )
        yield RawRecord(
            RecordType.DOCUMENT,
            source,
            {
                "mrn": mrn,
                "doc_id": path.stem,
                "doc_type": header.get("DOCUMENT TYPE"),
                "date": header.get("DATE OF SERVICE"),
                "encounter_id": header.get("ENCOUNTER"),
                "text": body,
            },
        )
