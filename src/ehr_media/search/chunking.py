"""Split long documents into overlapping word windows that fit the embedding model."""

from __future__ import annotations


def chunk_words(text: str, size: int = 150, overlap: int = 25) -> list[str]:
    """Overlapping windows of ``size`` words; short texts come back as one chunk.

    Overlap keeps a sentence that straddles a boundary retrievable from either side.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks
