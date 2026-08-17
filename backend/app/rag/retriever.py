"""Retrieval + the first grounding gate.

Layer 1 of the guardrail: if nothing retrieved is similar enough to the
question, we refuse here and never call the answering model at all. This is the
cheapest and most reliable rejection — no tokens spent, nothing to hallucinate
with.

The question is normalized to English first; see translate.py for why.
"""

from .config import SCORE_THRESHOLD, TOP_K, TRANSLATE_QUERIES
from .store import query
from .translate import to_english


def retrieve(doc_id: str, question: str, k: int = TOP_K) -> dict:
    """Returns the Retrieval shape from docs/contract.md.

    `question` is the student's own wording, in any language. `search_text` on
    the result is what was actually embedded — useful when a refusal looks
    wrong and you need to see whether translation was the cause.
    """
    search_text = to_english(question) if TRANSLATE_QUERIES else question
    chunks = query(doc_id, search_text, k)

    if not chunks:
        return {
            "in_scope": False,
            "top_score": 0.0,
            "reason": "empty_index",
            "chunks": [],
            "search_text": search_text,
        }

    top_score = chunks[0]["score"]
    if top_score < SCORE_THRESHOLD:
        return {
            "in_scope": False,
            "top_score": top_score,
            "reason": "below_threshold",
            "chunks": [],
            "search_text": search_text,
        }

    return {
        "in_scope": True,
        "top_score": top_score,
        "chunks": chunks,
        "search_text": search_text,
    }


def format_context(chunks: list[dict]) -> str:
    """Chunks as the model sees them — page number attached so it can cite."""
    return "\n\n".join(f"[p.{c['page']}] {c['text']}" for c in chunks)
