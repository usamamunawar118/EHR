"""End-to-end edge cases on tiny hand-made inputs (the assessment asks for >= 3)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ehr_media.ingestion.models import AuditAction, Gender, IngestResult, RecordType
from ehr_media.ingestion.pipeline import run_ingest
from tests.helpers import doc_a, enc_a, patient_a, write_export

REF = date(2026, 11, 11)


def ingest(path: Path) -> IngestResult:
    return run_ingest([path], reference_date=REF)


def rules(result: IngestResult, mrn: str | None = None) -> set[str]:
    return {e.rule for e in result.audit if mrn is None or e.mrn == mrn}


class TestConflictingIdentifiers:
    def test_same_mrn_for_two_different_people_is_quarantined_not_merged(
        self, tmp_path: Path
    ) -> None:
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000001",
                    family="Alpha",
                    given="Ann",
                    dob="1970-05-05",
                    documents=[doc_a("d1", date_="2025-01-02")],
                ),
                patient_a(
                    "MRN-1",
                    family="Zulu",
                    given="Zed",
                    dob="1999-09-09",
                    documents=[doc_a("d2", date_="2025-01-03", text="Different patient's note.")],
                ),
            ],
        )
        result = ingest(tmp_path)

        assert result.patients == ()  # nobody under that MRN is admitted
        assert {q.reason for q in result.quarantine} == {"IDENTIFIER_CONFLICT"}
        assert {q.record_type for q in result.quarantine} >= {
            RecordType.PATIENT,
            RecordType.DOCUMENT,
        }
        conflict = next(
            e
            for e in result.audit
            if e.rule == "IDENTIFIER_CONFLICT" and e.record_type is RecordType.PATIENT
        )
        assert conflict.action is AuditAction.REJECTED
        assert "1970-05-05" in conflict.message
        assert "1999-09-09" in conflict.message

    def test_typo_in_name_is_not_a_conflict(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [
                patient_a("00000002", family="Smith", given="Jon"),
                patient_a("00000002", family="Smith", given="John"),
            ],
        )
        result = ingest(tmp_path)
        assert len(result.patients) == 1
        assert "IDENTIFIER_CONFLICT" not in rules(result)

    def test_same_person_under_two_mrns_is_flagged_but_kept_separate(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [
                patient_a("00000003", family="Twin", given="Tia", dob="1991-01-01"),
                patient_a("00000004", family="Twin", given="Tia", dob="1991-01-01"),
            ],
        )
        result = ingest(tmp_path)
        assert len(result.patients) == 2
        assert all("POSSIBLE_DUPLICATE_PATIENT" in p.patient.flags for p in result.patients)


class TestMissingAndInvalidFields:
    def test_missing_dob_is_kept_and_flagged(self, tmp_path: Path) -> None:
        write_export(tmp_path, [patient_a("00000005", dob=None)])
        result = ingest(tmp_path)
        assert result.patients[0].patient.birth_date is None
        assert "DOB_MISSING" in result.patients[0].patient.flags

    @pytest.mark.parametrize("dob", ["2031-05-17", "1850-01-01"])
    def test_implausible_dob_is_discarded_with_audit(self, tmp_path: Path, dob: str) -> None:
        write_export(tmp_path, [patient_a("00000006", dob=dob)])
        result = ingest(tmp_path)
        assert result.patients[0].patient.birth_date is None
        entry = next(e for e in result.audit if e.rule == "DOB_IMPLAUSIBLE")
        assert entry.before == dob
        assert entry.action is AuditAction.REJECTED

    def test_missing_document_date_is_imputed_from_the_encounter(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000007",
                    encounters=[enc_a("E1", "03/04/2025")],
                    documents=[doc_a("d1", date_=None, encounter_id="E1")],
                )
            ],
        )
        result = ingest(tmp_path)
        doc = result.patients[0].documents[0]
        assert doc.document_date == date(2025, 3, 4)
        imputed = next(e for e in result.audit if e.rule == "DATE_IMPUTED")
        assert imputed.action is AuditAction.IMPUTED
        assert imputed.after == "2025-03-04"

    def test_document_with_no_date_and_no_encounter_is_quarantined(self, tmp_path: Path) -> None:
        write_export(tmp_path, [patient_a("00000008", documents=[doc_a("d1", date_=None)])])
        result = ingest(tmp_path)
        assert result.patients[0].documents == ()
        assert [q.reason for q in result.quarantine] == ["MISSING_DOCUMENT_DATE"]

    def test_blank_document_text_is_quarantined(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [patient_a("00000009", documents=[doc_a("d1", date_="2025-01-01", text="  \n ")])],
        )
        assert [q.reason for q in ingest(tmp_path).quarantine] == ["EMPTY_TEXT"]

    def test_unparseable_mrn_is_quarantined(self, tmp_path: Path) -> None:
        write_export(tmp_path, [patient_a("not-an-mrn")])
        result = ingest(tmp_path)
        assert result.patients == ()
        assert result.quarantine[0].reason == "INVALID_MRN"


class TestDuplicates:
    def test_identical_documents_collapse_to_one(self, tmp_path: Path) -> None:
        docs = [doc_a("d1", date_="2025-02-01"), doc_a("d2", date_="02/01/2025")]
        write_export(tmp_path, [patient_a("00000010", documents=docs)])
        result = ingest(tmp_path)
        assert len(result.patients[0].documents) == 1
        dup = next(e for e in result.audit if e.rule == "DUPLICATE_EXACT")
        assert dup.action is AuditAction.DEDUPLICATED

    def test_near_identical_documents_collapse_but_distinct_ones_do_not(
        self, tmp_path: Path
    ) -> None:
        base = "HPI: patient reports cough for three days with mild fever. PLAN: fluids and rest."
        ocr = base.replace("l", "1", 1)  # one OCR misread
        other = "HPI: completely different complaint about knee pain after a fall. PLAN: x-ray."
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000011",
                    documents=[
                        doc_a("d1", date_="2025-02-01", text=base),
                        doc_a("d2", date_="2025-02-01", text=ocr),
                        doc_a("d3", date_="2025-02-01", text=other),
                    ],
                )
            ],
        )
        result = ingest(tmp_path)
        assert len(result.patients[0].documents) == 2
        assert "DUPLICATE_NEAR" in rules(result)


class TestDateAndFieldNormalisation:
    def test_mixed_date_formats_become_iso_with_before_after_audit(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000012",
                    documents=[
                        doc_a("d1", date_="March 2, 2025", text="note one"),
                        doc_a("d2", date_="20250303", text="note two"),
                        doc_a("d3", date_="04-Mar-2025", text="note three"),
                    ],
                )
            ],
        )
        result = ingest(tmp_path)
        assert [d.document_date for d in result.patients[0].documents] == [
            date(2025, 3, 2),
            date(2025, 3, 3),
            date(2025, 3, 4),
        ]
        fmt = [e for e in result.audit if e.rule == "DATE_FORMAT"]
        assert {(e.before, e.after) for e in fmt} == {
            ("March 2, 2025", "2025-03-02"),
            ("20250303", "2025-03-03"),
            ("04-Mar-2025", "2025-03-04"),
        }

    def test_demographics_are_canonicalised_and_every_change_is_audited(
        self, tmp_path: Path
    ) -> None:
        write_export(
            tmp_path, [patient_a("MRN-12", family="SMITH", given="jOHN", sex="1", dob="3/4/56")]
        )
        result = ingest(tmp_path)
        p = result.patients[0].patient
        assert (p.mrn, p.family_name, p.given_name) == ("00000012", "Smith", "John")
        assert p.gender is Gender.MALE
        assert p.birth_date == date(1956, 3, 4)  # month-first; century chosen so not in future
        assert {
            "MRN_FORMAT",
            "NAME_FORMAT",
            "GENDER_CODE",
            "DATE_FORMAT",
            "TWO_DIGIT_YEAR",
            "AMBIGUOUS_DATE",
        } <= rules(result, "00000012")

    def test_unmapped_gender_is_flagged_not_guessed(self, tmp_path: Path) -> None:
        write_export(tmp_path, [patient_a("00000013", sex="N/A")])
        result = ingest(tmp_path)
        assert result.patients[0].patient.gender is Gender.UNKNOWN
        assert "GENDER_UNMAPPED" in rules(result)


class TestCrossSourceMerge:
    def test_json_csv_and_scanned_note_merge_into_one_patient(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000014",
                    family="Merge",
                    given="Mia",
                    dob="1980-01-02",
                    encounters=[enc_a("E7", "2025-03-01")],
                )
            ],
        )
        (tmp_path / "labs.csv").write_text(
            "MRN,PatientName,DOB,CollectedDate,TestName,Result,Units,RefRange,Flag,EncounterID\n"
            '14,"MERGE, MIA",01/02/1980,03/01/2025,GLU,180,MG/DL,70-99,,E7\n',
            encoding="utf-8-sig",
        )
        (tmp_path / "note.txt").write_text(
            "PATIENT NAME: MERGE, MIA\nMRN: 000O0014\nDOB: 01/02/1980    SEX: F\n"
            "DATE OF SERVICE: 03/01/2025\nDOCUMENT TYPE: PN\nENCOUNTER: E7\n" + "-" * 60 + "\n"
            "ASSESSMENT: elevated glu-\ncose. PLAN: adjust therapy.\n",
            encoding="utf-8",
        )

        result = ingest(tmp_path)

        assert len(result.patients) == 1
        rec = result.patients[0]
        assert rec.patient.gender is Gender.FEMALE  # filled in from the scanned note
        assert len(rec.documents) == 1
        assert "elevated glucose" in rec.documents[0].text
        lab = rec.lab_results[0]
        assert (lab.test_key, lab.unit, lab.flag.value if lab.flag else None) == (
            "glucose",
            "mg/dL",
            "H",
        )  # flag derived from value + reference range
        assert {"LAB_FLAG_DERIVED", "MRN_OCR_CORRECTION", "PATIENT_MERGED"} <= rules(result)
        assert rec.documents[0].encounter_id == "E7"


class TestEncounterLinking:
    def test_unknown_encounter_id_gets_a_synthesised_stub(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000015",
                    documents=[doc_a("d1", date_="2025-04-01", encounter_id="E-UNKNOWN")],
                )
            ],
        )
        rec = ingest(tmp_path).patients[0]
        assert rec.documents[0].encounter_id == "E-UNKNOWN"
        assert [(e.encounter_id, e.synthesized) for e in rec.encounters] == [("E-UNKNOWN", True)]

    def test_missing_encounter_id_links_to_the_only_covering_encounter(
        self, tmp_path: Path
    ) -> None:
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000016",
                    encounters=[enc_a("E9", "2025-04-01", "2025-04-05", "Inpatient")],
                    documents=[doc_a("d1", date_="2025-04-03")],
                )
            ],
        )
        result = ingest(tmp_path)
        assert result.patients[0].documents[0].encounter_id == "E9"
        assert "ENCOUNTER_LINKED_BY_DATE" in rules(result)

    def test_every_clean_item_ends_up_with_a_resolvable_encounter(self, tmp_path: Path) -> None:
        write_export(
            tmp_path,
            [
                patient_a(
                    "00000017",
                    documents=[
                        doc_a("d1", date_="2025-04-03", text="alpha"),
                        doc_a("d2", date_="2025-05-03", text="beta"),
                    ],
                )
            ],
        )
        rec = ingest(tmp_path).patients[0]
        known = {e.encounter_id for e in rec.encounters}
        assert all(d.encounter_id in known for d in rec.documents)


def test_pipeline_is_idempotent(tmp_path: Path) -> None:
    write_export(
        tmp_path,
        [
            patient_a(
                "00000018",
                documents=[doc_a("d1", date_="03/04/2025"), doc_a("d2", date_="2025-03-04")],
            )
        ],
    )
    first, second = ingest(tmp_path), ingest(tmp_path)
    assert first.model_dump() == second.model_dump()
