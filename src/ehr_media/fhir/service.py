"""Orchestration for Task 2: clean records -> FHIR bundles -> validation -> storage."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from ehr_media.fhir.builder import PatientBundle, build_patient_bundle
from ehr_media.fhir.report import ValidationReport, build_report
from ehr_media.fhir.validation import BundleValidation, validate_bundle
from ehr_media.ingestion.models import PatientRecord
from ehr_media.storage.bundle_repo import BundleRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FhirBuildResult:
    bundles: tuple[PatientBundle, ...]
    validations: tuple[BundleValidation, ...]
    report: ValidationReport


def build_and_validate(records: Iterable[PatientRecord]) -> FhirBuildResult:
    bundles: list[PatientBundle] = []
    validations: list[BundleValidation] = []
    for record in records:
        pb = build_patient_bundle(record)
        bundles.append(pb)
        validations.append(validate_bundle(pb.bundle))
    report = build_report(validations)
    logger.info(
        "fhir: %d bundles (%d valid, %d invalid), %d errors, %d warnings",
        report.bundles_total,
        report.bundles_valid,
        report.bundles_invalid,
        report.total_errors,
        report.total_warnings,
    )
    return FhirBuildResult(tuple(bundles), tuple(validations), report)


def store_bundles(result: FhirBuildResult, repo: BundleRepository) -> int:
    """Persist every bundle (valid or not, with its validation verdict). Returns the count."""
    for pb, validation in zip(result.bundles, result.validations, strict=True):
        repo.save(pb, validation)
    return len(result.bundles)
