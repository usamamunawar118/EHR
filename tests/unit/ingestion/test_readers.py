"""Reader robustness: odd encodings/delimiters and malformed files must not crash the run."""

from __future__ import annotations

from pathlib import Path

import pytest

from ehr_media.ingestion.models import RecordType
from ehr_media.ingestion.readers import discover_inputs, read_all
from ehr_media.ingestion.readers.base import UnsupportedInputError
from ehr_media.ingestion.readers.csv_reader import LabCsvReader
from ehr_media.ingestion.readers.json_reader import JsonExportReader
from ehr_media.ingestion.readers.text_reader import ScannedNoteReader

CSV_HEADER = "MRN,PatientName,DOB,CollectedDate,TestName,Result,Units,RefRange,Flag,EncounterID"
CSV_ROW = '12,"DOE, JANE",01/02/1980,03/04/2025,Glucose,95,mg/dL,70-99,N,E1'


def test_csv_with_bom_and_semicolon_delimiter(tmp_path: Path) -> None:
    path = tmp_path / "labs.csv"
    text = (CSV_HEADER + "\n" + CSV_ROW).replace(",", ";").replace('"DOE; JANE"', "DOE JANE")
    path.write_text(text, encoding="utf-8-sig")

    records = list(LabCsvReader().read(path))

    lab = next(r for r in records if r.record_type is RecordType.LAB)
    assert lab.fields["test"] == "Glucose"
    assert lab.fields["result"] == "95"
    assert lab.fields["mrn"] == "12"


def test_csv_emits_one_patient_claim_per_distinct_identity(tmp_path: Path) -> None:
    path = tmp_path / "labs.csv"
    path.write_text("\n".join([CSV_HEADER, CSV_ROW, CSV_ROW, CSV_ROW]), encoding="utf-8")
    kinds = [r.record_type for r in LabCsvReader().read(path)]
    assert kinds.count(RecordType.PATIENT) == 1
    assert kinds.count(RecordType.LAB) == 3


def test_csv_missing_required_columns_is_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        list(LabCsvReader().read(path))


def test_json_without_patients_list_is_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"unrelated": true}', encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        list(JsonExportReader().read(path))


def test_text_note_without_separator_is_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("just some text with no header block", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        list(ScannedNoteReader().read(path))


def test_read_all_skips_unreadable_files_and_keeps_going(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    (tmp_path / "good.csv").write_text("\n".join([CSV_HEADER, CSV_ROW]), encoding="utf-8")
    (tmp_path / "ignored.md").write_text("# not an input", encoding="utf-8")

    found = discover_inputs(tmp_path)
    assert [p.name for p in found] == ["bad.json", "good.csv"]  # .md is not a supported suffix
    assert any(r.record_type is RecordType.LAB for r in read_all(found))


def test_all_three_json_key_styles_are_understood(tmp_path: Path) -> None:
    import json

    styles = [
        {"MRN": "1", "name": {"family": "A", "given": "B"}, "dob": "1990-01-01", "sex": "M"},
        {"mrn": "2", "patient_name": "C, D", "birthDate": "1991-01-01", "gender": "F"},
        {
            "MedicalRecordNumber": "3",
            "FirstName": "E",
            "LastName": "F",
            "DateOfBirth": "1992-01-01",
            "Gender": "1",
        },
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps({"patients": styles}), encoding="utf-8")

    claims = [
        r.fields for r in JsonExportReader().read(path) if r.record_type is RecordType.PATIENT
    ]

    assert [c["mrn"] for c in claims] == ["1", "2", "3"]
    assert [c["dob"] for c in claims] == ["1990-01-01", "1991-01-01", "1992-01-01"]
    assert (claims[0]["family"], claims[0]["given"]) == ("A", "B")
    assert claims[1]["full_name"] == "C, D"
    assert (claims[2]["family"], claims[2]["given"]) == ("F", "E")
