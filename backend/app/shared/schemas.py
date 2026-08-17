"""Cross-branch contract types.

⚠️  SHARED FILE — coordinate every change with the `feat/rag-grounding` owner
before editing. The tutor branch only *consumes* these shapes; the RAG branch
produces them. Keeping this file byte-identical across branches is what stops
the merge from turning into a rewrite.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One passage retrieved from the student's own uploaded textbook page."""

    chunk_id: str
    page_number: int
    text: str
    chapter_title: str | None = None
    concepts: list[str] = Field(default_factory=list)
    similarity_score: float = 0.0


class GroundingResult(BaseModel):
    """What the RAG layer hands us for a single question.

    `is_in_scope=False` means nothing relevant enough was retrieved — the tutor
    must refuse rather than answer from general knowledge.
    """

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    is_in_scope: bool = False
