"""Document chunking."""

from __future__ import annotations

from advanced_graphrag.domain import Chunk, Document
from advanced_graphrag.utils import normalize_text, stable_id


class StructuralChunker:
    """Simple section-aware chunker.

    Production implementations should plug in PDF/HTML/code aware chunkers.
    This default keeps headings with nearby text and uses token windows with
    overlap, matching the practical pattern used by LightRAG/AWS/FalkorDB.
    """

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        words = normalize_text(document.text).split()
        if not words:
            return []
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        ordinal = 0
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            text = " ".join(window)
            chunk_id = stable_id("chunk", document.id, ordinal, text)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=text,
                    ordinal=ordinal,
                    metadata={"start_token": start, "end_token": start + len(window)},
                )
            )
            ordinal += 1
            if start + self.chunk_size >= len(words):
                break
        return chunks

