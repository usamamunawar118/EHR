"""Negative tests: corrupt a valid Bundle and check the report pinpoints the problem."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from ehr_media.fhir.ids import urn
from ehr_media.fhir.report import build_report, render_markdown, write_report
from ehr_media.fhir.validation import (
    BundleValidation,
    IssueSeverity,
    Layer,
    ValidationIssue,
    validate_bundle,
)
from tests.unit.fhir.conftest import find


def issues(result: BundleValidation, code: str) -> list[ValidationIssue]:
    return [i for i in result.issues if i.code == code]


def index_of(bundle: dict[str, Any], resource_type: str) -> int:
    return next(
        i for i, e in enumerate(bundle["entry"]) if e["resource"]["resourceType"] == resource_type
    )


def test_clean_bundle_is_valid(bundle: dict[str, Any]) -> None:
    result = validate_bundle(bundle)
    assert result.valid
    assert not result.errors
    assert result.mrn == "00000042"


class TestSchemaLayer:
    def test_malformed_date_is_reported_with_exact_path(self, bundle: dict[str, Any]) -> None:
        idx = index_of(bundle, "Patient")
        bundle["entry"][idx]["resource"]["birthDate"] = "03/04/1960"

        result = validate_bundle(bundle)

        assert not result.valid
        issue = next(i for i in result.errors if i.layer is Layer.SCHEMA)
        assert issue.path == f"entry[{idx}].resource.birthDate"
        assert issue.resource.startswith("Patient/")

    def test_missing_required_field_is_reported(self, bundle: dict[str, Any]) -> None:
        obs = find(bundle, "Observation")[0]
        del obs["code"]
        result = validate_bundle(bundle)
        assert not result.valid
        assert any(i.path.endswith(".code") and i.layer is Layer.SCHEMA for i in result.errors)

    def test_unsupported_resource_type(self, bundle: dict[str, Any]) -> None:
        bundle["entry"].append(
            {"fullUrl": "urn:uuid:x", "resource": {"resourceType": "Medication", "id": "x"}}
        )
        assert issues(validate_bundle(bundle), "unsupported-resource-type")

    @pytest.mark.parametrize("junk", [None, [], "bundle", {"resourceType": "Patient"}])
    def test_non_bundle_input_never_raises(self, junk: Any) -> None:
        result = validate_bundle(junk)
        assert not result.valid
        assert result.issues[0].code == "not-a-bundle"


class TestReferenceLayer:
    def test_dangling_reference(self, bundle: dict[str, Any]) -> None:
        enc_idx = index_of(bundle, "Encounter")
        bundle["entry"][enc_idx]["resource"]["subject"]["reference"] = "urn:uuid:does-not-exist"

        result = validate_bundle(bundle)

        (issue,) = issues(result, "unresolved-reference")
        assert issue.path == f"entry[{enc_idx}].resource.subject"
        assert "does-not-exist" in issue.message
        assert not result.valid

    def test_reference_to_wrong_resource_type(self, bundle: dict[str, Any]) -> None:
        (patient,) = find(bundle, "Patient")
        obs = find(bundle, "Observation")[0]
        obs["encounter"]["reference"] = urn(patient["id"])  # an Encounter is required here
        assert issues(validate_bundle(bundle), "wrong-reference-type")

    def test_report_result_must_reference_observations(self, bundle: dict[str, Any]) -> None:
        (patient,) = find(bundle, "Patient")
        lab = next(r for r in find(bundle, "DiagnosticReport") if "result" in r)
        lab["result"][0]["reference"] = urn(patient["id"])
        assert issues(validate_bundle(bundle), "wrong-reference-type")

    def test_second_patient_is_rejected(self, bundle: dict[str, Any]) -> None:
        extra = dict(find(bundle, "Patient")[0], id="other")
        bundle["entry"].append({"fullUrl": urn("other"), "resource": extra})
        assert issues(validate_bundle(bundle), "patient-count")

    def test_fullurl_must_match_resource_id(self, bundle: dict[str, Any]) -> None:
        bundle["entry"][0]["fullUrl"] = "urn:uuid:00000000-0000-0000-0000-000000000000"
        assert issues(validate_bundle(bundle), "fullurl-id-mismatch")

    def test_duplicate_resource_ids(self, bundle: dict[str, Any]) -> None:
        bundle["entry"].append(dict(bundle["entry"][1]))
        result = validate_bundle(bundle)
        assert issues(result, "duplicate-resource-id")
        assert issues(result, "duplicate-fullurl")


class TestProfileLayer:
    """The gap `fhir.resources` leaves open: required bindings are NOT enforced by the library."""

    def test_invalid_gender_code_is_caught_by_binding_check(self, bundle: dict[str, Any]) -> None:
        idx = index_of(bundle, "Patient")
        bundle["entry"][idx]["resource"]["gender"] = "banana"

        result = validate_bundle(bundle)

        (issue,) = issues(result, "invalid-code")
        assert issue.layer is Layer.PROFILE
        assert issue.path == f"entry[{idx}].resource.gender"
        assert "administrative-gender" in issue.message
        assert not result.valid

    @pytest.mark.parametrize(
        ("rtype", "vs"),
        [
            ("DocumentReference", "document-reference-status"),
            ("DiagnosticReport", "diagnostic-report-status"),
            ("Observation", "observation-status"),
            ("Encounter", "encounter-status"),
        ],
    )
    def test_bad_status_values(self, bundle: dict[str, Any], rtype: str, vs: str) -> None:
        find(bundle, rtype)[0]["status"] = "bogus"
        found = issues(validate_bundle(bundle), "invalid-code")
        assert found
        assert vs in found[0].message

    def test_bad_bundle_type(self, bundle: dict[str, Any]) -> None:
        bundle["type"] = "pile"
        assert any(i.path == "type" for i in issues(validate_bundle(bundle), "invalid-code"))

    def test_patient_without_mrn(self, bundle: dict[str, Any]) -> None:
        find(bundle, "Patient")[0]["identifier"] = []
        assert issues(validate_bundle(bundle), "missing-mrn")

    def test_encounter_period_order(self, bundle: dict[str, Any]) -> None:
        find(bundle, "Encounter")[0]["period"] = {"start": "2025-03-05", "end": "2025-03-01"}
        assert issues(validate_bundle(bundle), "period-order")

    def test_attachment_size_mismatch(self, bundle: dict[str, Any]) -> None:
        find(bundle, "DocumentReference")[0]["content"][0]["attachment"]["size"] = 1
        assert issues(validate_bundle(bundle), "attachment-size-mismatch")

    def test_attachment_with_invalid_base64(self, bundle: dict[str, Any]) -> None:
        find(bundle, "DocumentReference")[0]["content"][0]["attachment"]["data"] = "%%%not-base64"
        found = issues(validate_bundle(bundle), "invalid-base64")
        assert found
        assert base64  # imported for clarity of the scenario

    def test_unknown_code_system_is_a_warning_not_an_error(self, bundle: dict[str, Any]) -> None:
        coded = next(o for o in find(bundle, "Observation") if o["code"].get("coding"))
        coded["code"]["coding"][0]["system"] = "http://example.org/made-up"
        result = validate_bundle(bundle)
        (warning,) = issues(result, "unknown-code-system")
        assert warning.severity is IssueSeverity.WARNING
        assert result.valid  # warnings never fail a bundle

    def test_uncoded_observation_is_a_warning(self, bundle: dict[str, Any]) -> None:
        unknown = next(
            o for o in find(bundle, "Observation") if o["code"].get("text") == "Mystery Marker"
        )
        assert unknown["code"].get("coding") is None
        result = validate_bundle(bundle)
        assert issues(result, "uncoded-observation")[0].severity is IssueSeverity.WARNING
        assert result.valid


class TestReport:
    def _bad_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        find(bundle, "Patient")[0]["gender"] = "banana"
        return bundle

    def test_report_aggregates_counts_and_lists_failing_bundles(
        self, bundle: dict[str, Any]
    ) -> None:
        good = validate_bundle(bundle)
        bad = validate_bundle(self._bad_bundle(bundle))
        report = build_report([good, bad])
        assert (report.bundles_total, report.bundles_valid, report.bundles_invalid) == (2, 1, 1)
        assert report.total_errors == 1
        assert report.issues_by_code["error:invalid-code"] == 1
        assert report.issues_by_layer["profile"] >= 1
        assert len(report.bundles) == 2  # `good` has warnings (uncoded observation)

    def test_markdown_is_readable_and_names_the_path(self, bundle: dict[str, Any]) -> None:
        report = build_report([validate_bundle(self._bad_bundle(bundle))])
        md = render_markdown(report)
        assert "**Result: FAIL**" in md
        assert "resource.gender" in md
        assert "administrative-gender" in md

    def test_clean_report_says_pass_and_files_are_written(
        self, bundle: dict[str, Any], tmp_path: Any
    ) -> None:
        report = build_report([validate_bundle(bundle)])
        assert "**Result: PASS**" in render_markdown(report)
        json_path, md_path = write_report(report, tmp_path)
        assert json_path.exists()
        assert md_path.exists()
