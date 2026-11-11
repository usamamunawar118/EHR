"""Mapping correctness: golden expectations per resource type + bundle-level guarantees."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from ehr_media.fhir.builder import PatientBundle, build_patient_bundle
from ehr_media.fhir.ids import resource_id, urn
from ehr_media.fhir.validation import validate_bundle
from ehr_media.ingestion.models import PatientRecord
from tests.unit.fhir.conftest import IMAGING_TEXT, find


def decode(attachment: dict[str, Any]) -> str:
    return base64.b64decode(attachment["data"]).decode("utf-8")


class TestPatient:
    def test_identifier_name_gender_birthdate(self, bundle: dict[str, Any]) -> None:
        (patient,) = find(bundle, "Patient")
        ident = patient["identifier"][0]
        assert ident["value"] == "00000042"
        assert ident["type"]["coding"][0]["code"] == "MR"
        assert patient["name"][0] == {"use": "official", "family": "Fixture", "given": ["Fay"]}
        assert patient["gender"] == "female"
        assert patient["birthDate"] == "1960-02-03"

    def test_data_quality_flags_travel_as_meta_tags(self, record: PatientRecord) -> None:
        flagged = record.model_copy(
            update={"patient": record.patient.model_copy(update={"flags": ("DOB_MISSING",)})}
        )
        (patient,) = find(build_patient_bundle(flagged).bundle, "Patient")
        assert [t["code"] for t in patient["meta"]["tag"]] == ["DOB_MISSING"]

    def test_missing_dob_is_omitted_not_invented(self, record: PatientRecord) -> None:
        no_dob = record.model_copy(
            update={"patient": record.patient.model_copy(update={"birth_date": None})}
        )
        (patient,) = find(build_patient_bundle(no_dob).bundle, "Patient")
        assert "birthDate" not in patient


class TestEncounter:
    def test_class_status_period_and_subject(self, bundle: dict[str, Any]) -> None:
        (patient,) = find(bundle, "Patient")
        (enc,) = find(bundle, "Encounter")
        assert enc["status"] == "finished"
        assert enc["class"]["code"] == "IMP"
        assert enc["period"] == {"start": "2025-03-01", "end": "2025-03-04"}
        assert enc["subject"]["reference"] == urn(patient["id"])


class TestDocumentReference:
    def test_loinc_type_and_encounter_context(self, bundle: dict[str, Any]) -> None:
        (enc,) = find(bundle, "Encounter")
        by_code = {d["type"]["coding"][0]["code"]: d for d in find(bundle, "DocumentReference")}
        assert set(by_code) == {"11506-3", "18842-5", "18748-4"}  # progress / discharge / imaging
        progress = by_code["11506-3"]
        assert progress["status"] == "current"
        assert progress["context"]["encounter"][0]["reference"] == urn(enc["id"])

    def test_attachment_round_trips_text_with_size_and_sha1(self, bundle: dict[str, Any]) -> None:
        imaging = next(
            d
            for d in find(bundle, "DocumentReference")
            if d["type"]["coding"][0]["code"] == "18748-4"
        )
        att = imaging["content"][0]["attachment"]
        assert decode(att) == IMAGING_TEXT
        assert att["size"] == len(IMAGING_TEXT.encode())
        assert (
            att["hash"] == base64.b64encode(hashlib.sha1(IMAGING_TEXT.encode()).digest()).decode()
        )


class TestDiagnosticReport:
    def test_imaging_report_has_radiology_category_and_impression(
        self, bundle: dict[str, Any]
    ) -> None:
        rad = [
            r
            for r in find(bundle, "DiagnosticReport")
            if r["category"][0]["coding"][0]["code"] == "RAD"
        ]
        assert len(rad) == 1
        assert rad[0]["conclusion"] == "Right lower lobe pneumonia."
        assert rad[0]["code"]["text"] == "XR CHEST 2 VIEWS"
        assert decode(rad[0]["presentedForm"][0]) == IMAGING_TEXT

    def test_lab_report_references_every_observation_and_summarises_abnormals(
        self, bundle: dict[str, Any]
    ) -> None:
        (lab,) = [
            r
            for r in find(bundle, "DiagnosticReport")
            if r["category"][0]["coding"][0]["code"] == "LAB"
        ]
        obs_urls = {urn(o["id"]) for o in find(bundle, "Observation")}
        assert {r["reference"] for r in lab["result"]} == obs_urls
        assert "White Blood Cell Count: 15.2" in lab["conclusion"]
        assert "Sodium" not in lab["conclusion"]  # normal results are not listed as abnormal


class TestObservation:
    def test_quantity_has_ucum_interpretation_and_reference_range(
        self, bundle: dict[str, Any]
    ) -> None:
        wbc = next(
            o
            for o in find(bundle, "Observation")
            if o["code"].get("coding", [{}])[0].get("code") == "6690-2"
        )
        assert wbc["valueQuantity"] == {
            "value": 15.2,
            "unit": "10*3/uL",
            "system": "http://unitsofmeasure.org",
            "code": "10*3/uL",
        }
        assert wbc["interpretation"][0]["coding"][0]["code"] == "H"
        assert wbc["referenceRange"][0]["low"]["value"] == 4.5
        assert wbc["referenceRange"][0]["high"]["value"] == 11

    def test_unmapped_test_is_text_only_never_an_invented_code(
        self, bundle: dict[str, Any]
    ) -> None:
        unknown = next(
            o for o in find(bundle, "Observation") if o["code"].get("text") == "Mystery Marker"
        )
        assert "coding" not in unknown["code"]


class TestBundle:
    def test_is_a_collection_with_matching_fullurls(self, bundle: dict[str, Any]) -> None:
        assert bundle["type"] == "collection"
        for entry in bundle["entry"]:
            assert entry["fullUrl"] == urn(entry["resource"]["id"])

    def test_contains_the_required_resource_types(self, patient_bundle: PatientBundle) -> None:
        types = {r.resource_type for r in patient_bundle.resources}
        assert {
            "Patient",
            "DocumentReference",
            "DiagnosticReport",
            "Encounter",
            "Observation",
        } <= types

    def test_passes_all_three_validation_layers(self, bundle: dict[str, Any]) -> None:
        result = validate_bundle(bundle)
        assert result.valid, [i.message for i in result.errors]
        assert result.resource_counts["Patient"] == 1

    def test_build_is_deterministic(self, record: PatientRecord) -> None:
        a, b = build_patient_bundle(record), build_patient_bundle(record)
        assert a.bundle == b.bundle
        assert a.content_hash == b.content_hash
        assert a.patient_id == resource_id("Patient", "00000042")

    def test_content_hash_changes_when_clinical_content_changes(
        self, record: PatientRecord
    ) -> None:
        original = build_patient_bundle(record).content_hash
        changed_lab = record.lab_results[0].model_copy(update={"value": 99.9})
        edited = record.model_copy(update={"lab_results": (changed_lab, *record.lab_results[1:])})
        assert build_patient_bundle(edited).content_hash != original

    def test_search_rows_share_a_doc_key_for_imaging_documentreference_and_report(
        self, patient_bundle: PatientBundle
    ) -> None:
        imaging = [
            r
            for r in patient_bundle.resources
            if r.text
            and "XR CHEST" in r.text
            and r.resource_type in ("DocumentReference", "DiagnosticReport")
        ]
        assert len(imaging) == 2
        assert imaging[0].doc_key == imaging[1].doc_key  # lets search de-duplicate them
