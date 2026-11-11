"""Build docs/WRITEUP.md and docs/WRITEUP.pdf (must stay on ONE page) from a single source.

    pip install -e ".[docs]"
    python docs/build_writeup.py

The font size is chosen automatically: the largest size at which everything fits one Letter page.
Only characters from the built-in Helvetica (WinAnsi) set are used; others would render as boxes.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent

TITLE = "EHR Media Intelligence Platform - Write-up"
SUBTITLE = (
    "Repo: github.com/usamamunawar118/EHR  |  Python 3.11+, FastAPI, Pydantic v2, fhir.resources "
    "(R4B), sentence-transformers + Chroma, Claude (Haiku 5.5)  |  354 Python + 28 JS tests, "
    "95% coverage  |  all data synthetic"
)

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Tradeoffs I made, and why",
        [
            "<b>FHIR R4B, not R4.</b> fhir.resources 8.x, the only Pydantic v2 line, ships R5, R4B "
            "and STU3 but no pure R4. R4B is wire-compatible for the five resources I map; the "
            "pure-R4 line (6.x) needs Pydantic v1, which the required stack rules out.",
            "<b>Three-layer FHIR validation.</b> The library checks structure only (it accepted "
            'gender="banana"), so I added reference-integrity and required-binding / profile '
            "checks. Cost: not full HL7 conformance (no FHIRPath invariants or terminology "
            "server).",
            "<b>Quarantine, never merge.</b> One MRN shared by different people is quarantined for "
            "human review; the same name and DOB under two MRNs is flagged, not merged. A wrong "
            "merge is a patient-safety event; the cost is missing patients until reviewed.",
            "<b>Do not trust the prompt alone.</b> Anomalies (abnormal labs, data-quality tags) "
            "are detected in code, given to the model, and re-added by a guardrail if dropped. The "
            "200-word limit is enforced in code (target 150, one retry, then trim); the disclaimer "
            "and provenance come from code, not the model. More code than a prompt, but every "
            "guarantee is testable.",
            "<b>Privacy by design.</b> The model sees age and sex, never name, MRN or exact date "
            "of birth; record text is wrapped as untrusted data to resist prompt injection.",
            "<b>Search.</b> Chroma over FAISS for native metadata filters (its range filters are "
            "numeric-only, so dates are stored as YYYYMMDD integers). AI-summary vectors are "
            "tagged "
            "resource_type=Patient and dated by the latest record, so one filter model covers "
            "records and summaries; an imaging DocumentReference and its DiagnosticReport collapse "
            "into one hit.",
            "<b>Front end.</b> Vanilla JS with committed Tailwind CSS (no Node needed to run) and "
            "a "
            "native dialog for focus trap and Esc. Cost: hand-written rendering code.",
        ],
    ),
    (
        "What I would improve with more time",
        [
            "Validate with the official HL7 validator and add Condition resources coded in "
            "ICD-10 / SNOMED. Real OCR and layout parsing instead of simulated scans. A review UI "
            "for the quarantine / master-patient-index workflow.",
            "Authentication (SSO), an access audit log, encryption at rest and a BAA / "
            "zero-retention agreement before any real PHI - the model currently receives clinical "
            "text.",
            "Hybrid BM25 + vector retrieval with a cross-encoder reranker; sentence-level "
            "citations from each summary claim to its source passage; the Batch API for bulk "
            "summaries; Docker packaging; a clinician-reviewed gold set to measure summaries "
            "properly.",
        ],
    ),
    (
        "FHIR and clinical concepts I had to research",
        [
            "R4 vs R4B and which library versions ship which; Bundle types and in-bundle "
            "urn:uuid references; DocumentReference (a pointer to a document) vs DiagnosticReport "
            "(a result set referencing Observations); why every document and result needs an "
            "Encounter; LOINC codes (verified against the NLM registry) and UCUM units; identifier "
            "type MR; Encounter class (v3-ActCode) and observation interpretation H / L / N; MRN "
            "canonicalisation and OCR confusions (O/0, l/1); duplicate-patient vs "
            "identifier-conflict handling; HIPAA minimum-necessary (age, not date of birth).",
        ],
    ),
    (
        "How I validated AI summary quality",
        [
            "<b>1. Schema:</b> the model returns constrained JSON. <b>2. Deterministic scoring</b> "
            "of every summary (ehr eval-summaries, 60 patients): at most 200 words, disclaimer "
            "present, <i>grounded</i> (every date, ICD-10 code and multi-digit number traceable to "
            "the source), and recall of system-detected anomalies and coded diagnoses. "
            "<b>3. Tests that the guardrails fire:</b> over-length gets a retry then a trim; a "
            "dropped anomaly is restored; an invented value lowers confidence; a rounded value "
            "(10 for 10.2) is caught - this test found a bug in my first grounding check. "
            "<b>4. A manual rubric</b> (docs/summary_review_rubric.md): accuracy, no invention, "
            "completeness, calibration, safety.",
            "<b>Result and honest limits.</b> The offline extractive baseline scores 100% on word "
            "limit, disclaimer, grounding, anomaly recall and diagnosis recall across 60 synthetic "
            "patients. I had no API key while building, so <b>Claude's own output was not "
            "scored</b>: the adapter is unit-tested against a stubbed SDK, and pytest -m live runs "
            "the same harness on real output. The rubric review of real output is still to do. "
            "These checks catch invented facts, not clinical misjudgement.",
        ],
    ),
    (
        "Measured",
        [
            "Search over HTTP with the real MiniLM model, 60 patients / 404 index entries, 32 "
            "queries: median 84 ms, p95 109 ms, max 137 ms (requirement: under 2 s). UI: axe-core "
            "reports 0 violations (WCAG 2.x A/AA) in the initial, results and drawer states; "
            "verified at phone width and keyboard-only.",
        ],
    ),
]


def to_markdown() -> str:
    def md(text: str) -> str:
        return re.sub(r"<i>(.*?)</i>", r"*\1*", re.sub(r"<b>(.*?)</b>", r"**\1**", text))

    lines = [f"# {TITLE}", "", f"_{SUBTITLE}_", ""]
    for heading, items in SECTIONS:
        lines += [f"## {heading}", ""]
        lines += [f"- {md(item)}" for item in items]
        lines.append("")
    return "\n".join(lines)


def build_pdf(font_size: float) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=TITLE,
        author="EHR Media Intelligence Platform",
    )
    leading = font_size * 1.28
    body = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=font_size,
        leading=leading,
        alignment=TA_LEFT,
        spaceAfter=font_size * 0.28,
    )
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=9, bulletIndent=0)
    heading = ParagraphStyle(
        "h",
        fontName="Helvetica-Bold",
        fontSize=font_size + 1.8,
        leading=(font_size + 1.8) * 1.25,
        textColor=colors.HexColor("#3730a3"),
        spaceBefore=font_size * 0.55,
        spaceAfter=font_size * 0.25,
    )
    title = ParagraphStyle(
        "t",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#111827"),
        spaceAfter=2,
    )
    sub = ParagraphStyle(
        "s",
        fontName="Helvetica",
        fontSize=font_size - 0.6,
        leading=font_size * 1.2,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=2,
    )

    story: list = [Paragraph(TITLE, title), Paragraph(SUBTITLE, sub)]
    for head, items in SECTIONS:
        story.append(Paragraph(head, heading))
        for item in items:
            story.append(Paragraph(item, bullet, bulletText="•"))
    story.append(Spacer(1, 1))
    doc.build(story)
    return buf.getvalue()


def main() -> int:
    (HERE / "WRITEUP.md").write_text(to_markdown(), encoding="utf-8")
    chosen: tuple[float, bytes] | None = None
    size = 9.4
    while size >= 7.0:
        pdf = build_pdf(size)
        if len(PdfReader(io.BytesIO(pdf)).pages) == 1:
            chosen = (size, pdf)
            break
        size = round(size - 0.1, 1)
    if chosen is None:
        print(
            "ERROR: content does not fit one page even at 7.0 pt - shorten the text",
            file=sys.stderr,
        )
        return 1
    (HERE / "WRITEUP.pdf").write_bytes(chosen[1])
    print(f"wrote WRITEUP.md and WRITEUP.pdf (1 page, body font {chosen[0]} pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
