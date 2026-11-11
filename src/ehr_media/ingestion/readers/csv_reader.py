"""Reader for flat laboratory-result CSV exports."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from ehr_media.ingestion.models import RecordType, SourceRef
from ehr_media.ingestion.readers.base import RawRecord, UnsupportedInputError

_REQUIRED = {"mrn", "collecteddate", "testname", "result"}


class LabCsvReader:
    """Yields one LAB record per row, plus one PATIENT claim per distinct (MRN, name, DOB)."""

    format: ClassVar[str] = "csv"
    suffixes: ClassVar[tuple[str, ...]] = (".csv",)

    def read(self, path: Path) -> Iterator[RawRecord]:
        # utf-8-sig transparently strips the BOM that Excel-style exports prepend.
        with path.open(encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            if reader.fieldnames is None:
                raise UnsupportedInputError(f"{path.name}: empty CSV")
            columns = {name.strip().lower(): name for name in reader.fieldnames}
            if missing := _REQUIRED - columns.keys():
                raise UnsupportedInputError(f"{path.name}: missing columns {sorted(missing)}")

            def cell(row: dict[str, str], key: str) -> str | None:
                name = columns.get(key)
                value = row.get(name) if name else None
                return value.strip() if value is not None and value.strip() else None

            seen_claims: set[tuple[str | None, ...]] = set()
            for row in reader:
                locator = f"row {reader.line_num}"
                source = SourceRef(file=path.name, format="csv", locator=locator)
                mrn, name, dob = cell(row, "mrn"), cell(row, "patientname"), cell(row, "dob")
                claim_key = (mrn, name, dob)
                if claim_key not in seen_claims:
                    seen_claims.add(claim_key)
                    yield RawRecord(
                        RecordType.PATIENT,
                        SourceRef(file=path.name, format="csv", locator=f"claim {locator}"),
                        {"mrn": mrn, "full_name": name, "dob": dob},
                    )
                yield RawRecord(
                    RecordType.LAB,
                    source,
                    {
                        "mrn": mrn,
                        "collected": cell(row, "collecteddate"),
                        "test": cell(row, "testname"),
                        "result": cell(row, "result"),
                        "unit": cell(row, "units"),
                        "ref_range": cell(row, "refrange"),
                        "flag": cell(row, "flag"),
                        "encounter_id": cell(row, "encounterid"),
                    },
                )
