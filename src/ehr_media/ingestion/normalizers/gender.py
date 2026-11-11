"""Gender code normalisation to the FHIR ``administrative-gender`` value set."""

from __future__ import annotations

from dataclasses import dataclass

from ehr_media.ingestion.models import Gender

_MAP: dict[str, Gender] = {
    "m": Gender.MALE,
    "male": Gender.MALE,
    "man": Gender.MALE,
    "1": Gender.MALE,
    "f": Gender.FEMALE,
    "female": Gender.FEMALE,
    "woman": Gender.FEMALE,
    "2": Gender.FEMALE,
    "o": Gender.OTHER,
    "other": Gender.OTHER,
    "x": Gender.OTHER,
    "nonbinary": Gender.OTHER,
    "non-binary": Gender.OTHER,
    "3": Gender.OTHER,
    "u": Gender.UNKNOWN,
    "unk": Gender.UNKNOWN,
    "unknown": Gender.UNKNOWN,
}


@dataclass(frozen=True)
class GenderParse:
    value: Gender
    recognised: bool


def normalize_gender(raw: str | None) -> GenderParse:
    """Map a raw gender code to :class:`Gender`.

    Empty input is *not* an error (``unknown``, recognised). A non-empty value we cannot map is
    reported as unrecognised so the caller can flag it rather than guess.
    """
    if raw is None or not raw.strip():
        return GenderParse(Gender.UNKNOWN, True)
    key = raw.strip().strip(".").lower()
    if key in _MAP:
        return GenderParse(_MAP[key], True)
    return GenderParse(Gender.UNKNOWN, False)
