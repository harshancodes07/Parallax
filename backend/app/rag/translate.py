"""Normalize the student's question to English before embedding.

Why this exists: the student asks in their mother tongue about an English
textbook page, so query and passage are in different languages. Measured on a
real chapter, cross-lingual embedding does not separate in-scope from
out-of-scope questions — a Hindi question about the digestive system scored
higher against a photosynthesis page than three genuine Tamil questions about
it. That was true of both multilingual-e5-base and bge-m3.

Translating the query first moves the match into one language, where the same
questions separate cleanly (0.827 in-scope vs 0.795 out-of-scope).

Only the *search text* is translated. The student's original wording is what
gets sent to the answering model, so the reply comes back in their language.
"""

import functools
import logging

from .config import TRANSLATE_MODEL

log = logging.getLogger(__name__)

SYSTEM = """Translate the user's question into English.

Output only the translation, nothing else — no quotes, no notes, no preamble.
If the question is already in English, output it unchanged. Keep technical and
scientific terms accurate; this text is used to search a textbook."""


@functools.lru_cache(maxsize=512)
def to_english(question: str) -> str:
    """Best-effort. On any failure the original question is returned, so
    retrieval degrades rather than breaking — English questions are unaffected
    either way."""
    from .guardrail import _get_client  # local import avoids a circular import

    try:
        resp = _get_client().messages.create(
            model=TRANSLATE_MODEL,
            max_tokens=1000,
            output_config={"effort": "low"},
            system=SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or question
    except Exception as exc:  # noqa: BLE001 - never let translation break a lookup
        log.warning("query translation failed, searching with the original: %s", exc)
        return question
