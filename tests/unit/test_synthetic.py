"""The synthetic generator must be deterministic and must never emit real-looking PII."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ehr_media.synthetic import generate_dataset


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_same_seed_gives_byte_identical_output(tmp_path: Path) -> None:
    generate_dataset(tmp_path / "a" / "raw", seed=7)
    generate_dataset(tmp_path / "b" / "raw", seed=7)
    assert _tree_bytes(tmp_path / "a") == _tree_bytes(tmp_path / "b")


def test_different_seed_gives_different_output(tmp_path: Path) -> None:
    generate_dataset(tmp_path / "a" / "raw", seed=1)
    generate_dataset(tmp_path / "b" / "raw", seed=2)
    assert _tree_bytes(tmp_path / "a") != _tree_bytes(tmp_path / "b")


def test_manifest_describes_every_seeded_defect(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path / "raw")
    on_disk = json.loads((tmp_path / "DEFECT_MANIFEST.json").read_text("utf-8"))
    assert on_disk["synthetic"] is True
    assert len(on_disk["expected"]) == len(manifest.expected) >= 30
    assert {d["rule"] for d in on_disk["expected"]} >= {
        "IDENTIFIER_CONFLICT",
        "POSSIBLE_DUPLICATE_PATIENT",
        "DUPLICATE_EXACT",
        "DUPLICATE_NEAR",
        "DATE_IMPUTED",
        "MISSING_DOCUMENT_DATE",
        "DOB_MISSING",
        "DOB_IMPLAUSIBLE",
    }


def test_dataset_is_large_enough_for_the_performance_requirement(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path / "raw")
    assert manifest.totals["patients"] >= 50


def test_output_is_labelled_synthetic_and_has_no_real_pii_patterns(tmp_path: Path) -> None:
    generate_dataset(tmp_path / "raw")
    export = (tmp_path / "raw" / "ehr_export.json").read_text("utf-8")
    assert "SYNTHETIC" in export

    corpus = "\n".join(
        p.read_text("utf-8-sig") for p in (tmp_path / "raw").rglob("*") if p.is_file()
    )
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", corpus), "SSN-like pattern"
    assert not re.search(r"[\w.]+@[\w.]+\.\w+", corpus), "email address"
    assert not re.search(r"\(?\b\d{3}\)?[ -]\d{3}-\d{4}\b", corpus), "phone number"
