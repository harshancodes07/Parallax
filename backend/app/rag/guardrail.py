"""Layer 2 of the guardrail: grounded generation with a refusal sentinel.

Layer 1 (retriever.py) rejects questions nothing in the book is close to.
Layer 2 covers the case where a chunk *looks* similar but doesn't actually
answer the question — the model is told to emit a fixed marker instead of
guessing, and we check for that marker rather than trying to read refusal out
of prose.
"""

import re

import anthropic

from .config import ANSWER_MODEL, REFUSAL_SENTINEL
from .retriever import format_context, retrieve

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SYSTEM = f"""You are helping a school student understand one page of their own textbook.

You answer STRICTLY from the excerpts inside <context>. The excerpts are the
entire world: if they do not contain the answer, you do not know it.

If the excerpts do not answer the question, reply with exactly this and nothing
else:
{REFUSAL_SENTINEL}

Otherwise:
- Answer only from the excerpts. Never add outside knowledge, even if you are
  confident it is correct.
- Cite the page for every fact, like [p.4].
- If you are unsure whether the excerpts cover it, treat that as not covered and
  return {REFUSAL_SENTINEL}.
- Answer in the language the student asked in, simply enough for a 14-year-old."""

_CITATION_RE = re.compile(r"\[p\.(\d+)\]")


def answer(doc_id: str, question: str) -> dict:
    """Returns the Answer shape from docs/contract.md."""
    hit = retrieve(doc_id, question)
    if not hit["in_scope"]:
        return {
            "grounded": False,
            "text": None,
            "reason": hit["reason"],
            "top_score": hit["top_score"],
        }

    context = format_context(hit["chunks"])
    resp = _client.messages.create(
        model=ANSWER_MODEL,
        # max_tokens caps thinking + visible text together on Opus 5, so leave
        # headroom above the length of the answer you actually want.
        max_tokens=3000,
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}",
            }
        ],
    )

    text = "".join(b.text for b in resp.content if b.type == "text").strip()

    if REFUSAL_SENTINEL in text:
        return {
            "grounded": False,
            "text": None,
            "reason": "model_refused",
            "top_score": hit["top_score"],
        }

    return {
        "grounded": True,
        "text": text,
        "citations": sorted({int(p) for p in _CITATION_RE.findall(text)}),
        "top_score": hit["top_score"],
    }
