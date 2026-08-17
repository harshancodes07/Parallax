"""Layer 2 of the guardrail: grounded generation with a refusal sentinel.

Layer 1 (retriever.py) rejects questions nothing in the book is close to.
Layer 2 covers the case where a chunk *looks* similar but doesn't actually
answer the question — the model is told to emit a fixed marker instead of
guessing, and we check for that marker rather than trying to read refusal out
of prose.
"""

import functools
import re

import anthropic

from .config import ANSWER_MODEL, REFUSAL_SENTINEL
from .retriever import format_context, retrieve


@functools.lru_cache(maxsize=1)
def _get_client() -> anthropic.Anthropic:
    """Built on first use, not at import — reading ANTHROPIC_API_KEY eagerly
    would break `import rag` for anyone who only needs retrieval."""
    return anthropic.Anthropic()

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
    resp = _get_client().messages.create(
        model=ANSWER_MODEL,
        # max_tokens caps thinking + visible text together on Opus 5, so leave
        # headroom above the length of the answer you actually want.
        max_tokens=3000,
        # Not "low". Calibration showed the similarity gate cannot separate
        # same-subject, adjacent-topic questions ("respiration" scores 0.844
        # against a photosynthesis page; the worst genuine in-scope question
        # scores 0.846). So this call is the primary grounding gate, not a
        # backstop, and it gets the effort that deserves.
        output_config={"effort": "high"},
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}",
            }
        ],
    )

    text = "".join(b.text for b in resp.content if b.type == "text").strip()

    # No text at all means the safety classifier declined the request, or thinking
    # consumed max_tokens before any answer was written. An empty string does not
    # contain the sentinel, so without this it would fall through as a grounded
    # answer that says nothing — the exact failure this module exists to prevent.
    if resp.stop_reason == "refusal" or not text:
        return {
            "grounded": False,
            "text": None,
            "reason": "no_answer",
            "top_score": hit["top_score"],
        }

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
