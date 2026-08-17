"""Split OCR page text into overlapping chunks that carry their page number.

Page numbers ride along on every chunk because the answer has to cite them —
"[p.4]" is what lets a judge verify the answer came from the book.
"""

from dataclasses import dataclass

from .config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str


def chunk_page(doc_id: str, page: int, text: str) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    if step <= 0:
        # Fail loudly. With a negative step this loop yields no chunks at all,
        # so the document indexes empty and every question about it comes back
        # "not in this chapter" — a misconfiguration wearing the costume of a
        # working guardrail.
        raise ValueError(
            f"CHUNK_OVERLAP_WORDS ({CHUNK_OVERLAP_WORDS}) must be less than "
            f"CHUNK_WORDS ({CHUNK_WORDS})"
        )

    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + CHUNK_WORDS]
        if not window:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}-p{page}-c{i}",
                doc_id=doc_id,
                page=page,
                text=" ".join(window),
            )
        )
        # The last window already reached the end; another pass would only
        # re-emit its tail as a near-duplicate chunk.
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def chunk_document(doc: dict) -> list[Chunk]:
    """`doc` is the IngestedDoc shape from docs/contract.md."""
    doc_id = doc["doc_id"]
    chunks: list[Chunk] = []
    for page in doc["pages"]:
        chunks.extend(chunk_page(doc_id, page["page"], page.get("text", "")))
    return chunks
