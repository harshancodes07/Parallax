"""Retrieval + the first grounding gate.

Layer 1 of the guardrail: if nothing retrieved is similar enough to the
question, we refuse here and never call the model at all. This is the cheapest
and most reliable rejection — no tokens spent, nothing to hallucinate with.
"""

from .config import SCORE_THRESHOLD, TOP_K
from .store import query


def retrieve(doc_id: str, question: str, k: int = TOP_K) -> dict:
    """Returns the Retrieval shape from docs/contract.md."""
    chunks = query(doc_id, question, k)

    if not chunks:
        return {"in_scope": False, "top_score": 0.0, "reason": "empty_index", "chunks": []}

    top_score = chunks[0]["score"]
    if top_score < SCORE_THRESHOLD:
        return {
            "in_scope": False,
            "top_score": top_score,
            "reason": "below_threshold",
            "chunks": [],
        }

    return {"in_scope": True, "top_score": top_score, "chunks": chunks}


def format_context(chunks: list[dict]) -> str:
    """Chunks as the model sees them — page number attached so it can cite."""
    return "\n\n".join(f"[p.{c['page']}] {c['text']}" for c in chunks)
