"""Medical Record Number (MRN) normalisation.

Canonical form: exactly :data:`~ehr_media.ingestion.models.MRN_WIDTH` digits, zero padded.

Handled input variants: ``MRN-00012345``, ``mrn 12345``, ``#00012345``, ``12345``,
leading/trailing whitespace, and common OCR confusions in scanned notes
(``O``/``o`` -> ``0``, ``I``/``l``/``|`` -> ``1``) which are corrected but audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ehr_media.ingestion.models import MRN_WIDTH

_PREFIX = re.compile(r"^\s*(?:mrn|medical\s*record\s*(?:number|no))?\s*[#:.\-]*\s*", re.IGNORECASE)
_SEPARATORS = re.compile(r"[\s\-_.]")
_OCR_TABLE = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})
_OCR_CANDIDATE = re.compile(r"^[0-9OoIl|]+$")


@dataclass(frozen=True)
class MrnParse:
    value: str | None
    rules: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None


def normalize_mrn(raw: str | None) -> MrnParse:
    if raw is None or not raw.strip():
        return MrnParse(None, error="empty MRN")

    body = _SEPARATORS.sub("", _PREFIX.sub("", raw, count=1))
    if not body:
        return MrnParse(None, error="empty MRN")

    rules: list[str] = []
    if not body.isdigit():
        if _OCR_CANDIDATE.match(body):
            body = body.translate(_OCR_TABLE)
            rules.append("MRN_OCR_CORRECTION")
        else:
            return MrnParse(None, error=f"MRN contains non-numeric characters: {raw!r}")

    if len(body) > MRN_WIDTH:
        return MrnParse(None, error=f"MRN longer than {MRN_WIDTH} digits: {raw!r}")

    value = body.zfill(MRN_WIDTH)
    if value != raw.strip() and "MRN_OCR_CORRECTION" not in rules:
        rules.append("MRN_FORMAT")
    return MrnParse(value, tuple(rules))
