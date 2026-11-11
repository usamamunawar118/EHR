"""Deterministic resource identifiers.

UUIDv5 over stable business keys means re-running the pipeline yields identical IDs
(idempotent upserts, stable cache keys, diff-able output).
"""

from __future__ import annotations

import uuid

NAMESPACE = uuid.UUID("6f1c5e2e-5a41-4c3f-9a3d-2f7c9b0e8d11")


def resource_id(resource_type: str, *key_parts: object) -> str:
    """Stable id for ``resource_type`` identified by ``key_parts``."""
    name = "|".join([resource_type, *(str(p) for p in key_parts)])
    return str(uuid.uuid5(NAMESPACE, name))


def urn(rid: str) -> str:
    """In-bundle reference/fullUrl form (valid for ``collection`` and ``transaction`` bundles)."""
    return f"urn:uuid:{rid}"
