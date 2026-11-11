"""Text embedders behind one protocol.

* :class:`SentenceTransformerEmbedder` - the real model (``all-MiniLM-L6-v2`` by default).
  Loaded **once** and warmed up at start-up so no request pays the load cost.
* :class:`HashingEmbedder` - deterministic bag-of-words hashing. No download, no GPU, instant:
  used by fast tests and as an offline fallback. Similar wording -> similar vectors, but it has
  no real semantics (synonyms don't match), which is exactly why the real model is the default.

All vectors are L2-normalised, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ehr_media.config import Settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        # Imported lazily: it pulls in torch, which the offline/hashing paths never need.
        from sentence_transformers import SentenceTransformer

        logger.info("loading embedding model %s", model_name)
        self.name = model_name
        self._model = SentenceTransformer(model_name)
        # Renamed in newer sentence-transformers releases; support both.
        model: Any = self._model
        dim_fn = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        self.dimension = int(dim_fn() or 0)
        self.embed(["warm-up"])  # first call JIT-initialises kernels; do it before serving

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    name = "hashing-bow-v1"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec


def create_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
