"""Laboratory result normalisation: test names, units, reference ranges and abnormal flags.

Unit equivalences used here are *value preserving* (e.g. ``mEq/L`` == ``mmol/L`` for monovalent
ions, ``ug/L`` == ``ng/mL``), so no numeric conversion is ever applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ehr_media.ingestion.models import LabFlag


@dataclass(frozen=True)
class LabTest:
    key: str
    name: str
    unit: str
    aliases: tuple[str, ...] = ()


LAB_TESTS: tuple[LabTest, ...] = (
    LabTest("glucose", "Glucose", "mg/dL", ("glucose, serum", "glu", "blood glucose")),
    LabTest("hba1c", "Hemoglobin A1c", "%", ("hba1c", "a1c", "glycated hemoglobin")),
    LabTest("creatinine", "Creatinine", "mg/dL", ("creat", "serum creatinine")),
    LabTest("egfr", "eGFR", "mL/min/1.73m2", ("gfr est", "estimated gfr")),
    LabTest("potassium", "Potassium", "mmol/L", ("k", "serum potassium")),
    LabTest("sodium", "Sodium", "mmol/L", ("na", "serum sodium")),
    LabTest("wbc", "White Blood Cell Count", "10*3/uL", ("wbc", "leukocytes")),
    LabTest("hemoglobin", "Hemoglobin", "g/dL", ("hgb",)),
    LabTest("ferritin", "Ferritin", "ng/mL", ("ferritin, serum",)),
    LabTest("bnp", "B-type Natriuretic Peptide", "pg/mL", ("bnp",)),
    LabTest("crp", "C-Reactive Protein", "mg/L", ("crp", "c reactive protein")),
    LabTest("tsh", "Thyroid Stimulating Hormone", "mIU/L", ("tsh",)),
)

_BY_ALIAS: dict[str, LabTest] = {}
for _t in LAB_TESTS:
    for _name in (_t.name, _t.key, *_t.aliases):
        _BY_ALIAS[_name.casefold()] = _t

_UNIT_ALIASES: dict[str, str] = {
    "mg/dl": "mg/dL",
    "percent": "%",
    "%": "%",
    "mmol/l": "mmol/L",
    "meq/l": "mmol/L",
    "10*3/ul": "10*3/uL",
    "x10^3/ul": "10*3/uL",
    "k/ul": "10*3/uL",
    "g/dl": "g/dL",
    "ng/ml": "ng/mL",
    "ug/l": "ng/mL",
    "pg/ml": "pg/mL",
    "mg/l": "mg/L",
    "miu/l": "mIU/L",
    "uiu/ml": "mIU/L",
    "ml/min/1.73m2": "mL/min/1.73m2",
    "ml/min/1.73 m2": "mL/min/1.73m2",
}

_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(-?\d+(?:\.\d+)?)\s*$")
_LT = re.compile(r"^\s*(?:<|<=|≤)\s*(-?\d+(?:\.\d+)?)\s*$")
_GT = re.compile(r"^\s*(?:>|>=|≥)\s*(-?\d+(?:\.\d+)?)\s*$")

_FLAG_ALIASES: dict[str, LabFlag] = {
    "h": LabFlag.HIGH,
    "hh": LabFlag.HIGH,
    "high": LabFlag.HIGH,
    "l": LabFlag.LOW,
    "ll": LabFlag.LOW,
    "low": LabFlag.LOW,
    "n": LabFlag.NORMAL,
    "normal": LabFlag.NORMAL,
}


def lookup_test(raw_name: str) -> LabTest | None:
    return _BY_ALIAS.get(re.sub(r"\s+", " ", raw_name).strip().casefold())


def normalize_unit(raw: str | None) -> tuple[str | None, bool]:
    """Return ``(canonical_unit, recognised)``. Empty -> ``(None, True)``."""
    if raw is None or not raw.strip():
        return None, True
    key = re.sub(r"\s+", " ", raw.strip()).lower()
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key], True
    return raw.strip(), False


def parse_reference_range(raw: str | None) -> tuple[float | None, float | None]:
    """Parse ``"70-99"``, ``"70 - 99"``, ``"<5.7"`` or ``">10"`` into ``(low, high)``."""
    if raw is None or not raw.strip():
        return None, None
    if m := _RANGE.match(raw):
        return float(m[1]), float(m[2])
    if m := _LT.match(raw):
        return None, float(m[1])
    if m := _GT.match(raw):
        return float(m[1]), None
    return None, None


def parse_flag(raw: str | None) -> LabFlag | None:
    """Return the source flag, or ``None`` when absent/unrecognised."""
    if raw is None or not raw.strip():
        return None
    return _FLAG_ALIASES.get(raw.strip().lower())


def derive_flag(value: float, low: float | None, high: float | None) -> LabFlag | None:
    """Compute H/L/N from the value and reference range (``None`` when no range is known)."""
    if low is None and high is None:
        return None
    if high is not None and value > high:
        return LabFlag.HIGH
    if low is not None and value < low:
        return LabFlag.LOW
    return LabFlag.NORMAL
