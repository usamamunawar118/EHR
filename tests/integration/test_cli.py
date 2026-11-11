"""CLI smoke tests (real commands on a temp directory)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ehr_media.cli import app

runner = CliRunner()


def test_generate_then_ingest(tmp_path: Path) -> None:
    raw, out = tmp_path / "raw", tmp_path / "out"

    gen = runner.invoke(app, ["generate-data", "--out", str(raw)])
    assert gen.exit_code == 0, gen.output
    assert (raw / "ehr_export.json").exists()
    assert (tmp_path / "DEFECT_MANIFEST.json").exists()

    res = runner.invoke(
        app, ["ingest", "--input", str(raw), "--out", str(out), "--reference-date", "2026-11-11"]
    )
    assert res.exit_code == 0, res.output

    summary = json.loads((out / "ingest_summary.json").read_text("utf-8"))
    assert summary["stats"]["patients"] == 60
    assert summary["stats"]["conflicting_mrns"] == 2
    assert "IDENTIFIER_CONFLICT" in summary["quarantine_by_reason"]


def test_ingest_with_missing_input_dir_fails_cleanly(tmp_path: Path) -> None:
    res = runner.invoke(app, ["ingest", "--input", str(tmp_path / "nope")])
    assert res.exit_code != 0
