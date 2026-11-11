"""Typed application settings (12-factor: environment variables / `.env`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LLMProvider = Literal["auto", "anthropic", "fake"]
EmbeddingBackend = Literal["sentence-transformers", "hashing"]


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with an ``EHR_`` env var."""

    model_config = SettingsConfigDict(
        env_prefix="EHR_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- paths ---------------------------------------------------------------
    raw_dir: Path = PROJECT_ROOT / "data" / "sample" / "raw"
    data_dir: Path = PROJECT_ROOT / "data" / "runtime"
    # Only `public/` is served; tooling (node_modules, package.json, tests) stays outside it.
    frontend_dir: Path = PROJECT_ROOT / "frontend" / "public"

    # --- LLM -----------------------------------------------------------------
    llm_provider: LLMProvider = "auto"
    llm_model: str = "claude-haiku-5-5"
    # Thinking tokens count toward max_tokens, so leave generous headroom for a ~200-word answer.
    llm_max_tokens: int = Field(default=4096, ge=1024, le=64000)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    summary_concurrency: int = Field(default=4, ge=1, le=16)
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "EHR_ANTHROPIC_API_KEY"),
        repr=False,
    )

    # --- embeddings / search -------------------------------------------------
    embedding_backend: EmbeddingBackend = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    search_top_k: int = Field(default=5, ge=1, le=50)
    search_overfetch: int = Field(default=6, ge=1, le=50)
    # MiniLM truncates at 256 word-pieces; ~150 clinical words stays safely inside that.
    chunk_words: int = Field(default=150, ge=40)
    chunk_overlap_words: int = Field(default=25, ge=0)

    # --- derived paths -------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "ehr.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def ingest_dir(self) -> Path:
        return self.data_dir / "ingest"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def resolved_llm_provider(self) -> Literal["anthropic", "fake"]:
        """Resolve ``auto`` to a concrete provider (anthropic only when a key exists)."""
        if self.llm_provider == "auto":
            return "anthropic" if self.anthropic_api_key else "fake"
        return self.llm_provider

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.ingest_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
