"""Date parsing for heterogeneous EHR sources.

Rules (documented because dates are the most common silent data-quality bug):

* ISO ``YYYY-MM-DD``, compact ``YYYYMMDD``, ``DD-Mon-YYYY`` and ``Month D, YYYY`` are unambiguous.
* Numeric ``a/b/c`` is read **month-first** (US convention; Epic is a US system). If the first
  number is > 12 it can only be a day, so it is read day-first. When both readings are valid and
  differ the result is flagged ``ambiguous`` so the audit log records the assumption.
* Two-digit years use the century that keeps the date **not in the future** relative to the
  reference date (e.g. with reference 2026: ``56`` -> 1956, ``07`` -> 2007).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

NULL_TOKENS = frozenset({"", "unknown", "n/a", "na", "null", "none", "-", "--", "?"})

_MONTHS = {
    name: idx
    for idx, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ].*)?$")
_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_NUMERIC = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})$")
_DMY_MON = re.compile(r"^(\d{1,2})[\s\-]([A-Za-z]{3,9})\.?[\s,\-]+(\d{4})$")
_LONG = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$")


@dataclass(frozen=True)
class DateParse:
    """Result of :func:`parse_date`."""

    value: date | None
    fmt: str | None = None
    ambiguous: bool = False
    two_digit_year: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


def is_null_token(raw: str | None) -> bool:
    return raw is None or raw.strip().lower() in NULL_TOKENS


def _expand_year(yy: int, reference: date) -> int:
    century_2000 = 2000 + yy
    return century_2000 if century_2000 <= reference.year else 1900 + yy


def _build(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(raw: str | None, *, reference: date) -> DateParse:
    """Parse ``raw`` into a :class:`date`. Never raises; failures are reported in ``error``."""
    if is_null_token(raw):
        return DateParse(None, error="empty")
    text = (raw or "").strip()

    def fail(reason: str) -> DateParse:
        return DateParse(None, error=reason)

    if m := _ISO.match(text):
        value = _build(int(m[1]), int(m[2]), int(m[3]))
        return DateParse(value, "iso") if value else fail("invalid calendar date")

    if m := _COMPACT.match(text):
        value = _build(int(m[1]), int(m[2]), int(m[3]))
        return DateParse(value, "compact") if value else fail("invalid calendar date")

    if m := _NUMERIC.match(text):
        a, b, y_raw = int(m[1]), int(m[2]), m[3]
        two_digit = len(y_raw) == 2
        year = _expand_year(int(y_raw), reference) if two_digit else int(y_raw)
        if a > 12 and b <= 12:  # can only be day-first
            value = _build(year, b, a)
            fmt, ambiguous = "dmy_numeric", False
        else:
            value = _build(year, a, b)
            fmt = "us_numeric"
            ambiguous = a <= 12 and b <= 12 and a != b
        if value is None:
            return fail("invalid calendar date")
        return DateParse(value, fmt, ambiguous=ambiguous, two_digit_year=two_digit)

    if m := _DMY_MON.match(text):
        month = _MONTHS.get(m[2].lower())
        value = _build(int(m[3]), month, int(m[1])) if month else None
        return DateParse(value, "dmy_month_name") if value else fail("invalid month name")

    if m := _LONG.match(text):
        month = _MONTHS.get(m[1].lower())
        value = _build(int(m[3]), month, int(m[2])) if month else None
        return DateParse(value, "month_name_long") if value else fail("invalid month name")

    return fail("unrecognised date format")
