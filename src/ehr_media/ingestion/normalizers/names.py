"""Patient name parsing and comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class NameParse:
    family: str | None
    given: str | None


def _title(text: str) -> str:
    return _WS.sub(" ", text).strip().title()


def parse_name(
    *, family: str | None = None, given: str | None = None, full: str | None = None
) -> NameParse:
    """Parse structured (family/given) or free-text names.

    ``"SMITH, JOHN A"`` and ``"John A Smith"`` both become family ``Smith`` / given ``John A``.
    """
    fam = (family or "").strip()
    giv = (given or "").strip()
    if not fam and not giv and full and full.strip():
        text = full.strip()
        if "," in text:
            fam, _, giv = (part.strip() for part in text.partition(","))
        else:
            *firsts, last = text.split()
            fam, giv = last, " ".join(firsts)
    return NameParse(_title(fam) or None, _title(giv) or None)


def name_key(family: str | None, given: str | None) -> str:
    """Case/punctuation-insensitive key used for equality and similarity checks."""
    return re.sub(r"[^a-z0-9]", "", f"{family or ''}{given or ''}".lower())


def name_similarity(a: NameParse, b: NameParse) -> float:
    ka, kb = name_key(a.family, a.given), name_key(b.family, b.given)
    if not ka or not kb:
        return 1.0  # nothing to compare: do not treat as a conflict
    return SequenceMatcher(None, ka, kb).ratio()
