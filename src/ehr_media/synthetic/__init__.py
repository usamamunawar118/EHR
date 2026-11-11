"""Synthetic (fictional) EHR data used for demos and tests. No real patient data."""

from ehr_media.synthetic.generator import DEFAULT_SEED, Manifest, generate_dataset

__all__ = ["DEFAULT_SEED", "Manifest", "generate_dataset"]
