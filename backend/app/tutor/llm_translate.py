"""LLM translation, used only where IndicTrans2 cannot run.

IndicTransToolkit ships no Windows wheels and needs a C++ compiler, so on a
Windows demo machine the IndicTrans2 directions are unavailable. Rather than
lose the two things that depend on them, we fall back to the LLM.

Where this is a fine substitute:
    query translation — a student's Tanglish question into English so retrieval
    can match an English textbook page. Any competent model does this well.

Where it is genuinely weaker:
    backtranslation validation — the point of backtranslating the composed Tamil
    was to check it with an *independent* model. Falling back to the same model
    that wrote it is self-validation, and a model that drops a concept while
    writing may well drop it again while backtranslating. It still catches gross
    failures, and it is better than skipping the check, but it is not the same
    guarantee. Every report says which engine produced it so this never hides.

Preferred order is always IndicTrans2 first; this module is the fallback.
"""

from __future__ import annotations

import logging

from app.tutor.indic import languages
from app.tutor.llm_client import LLMClient, LLMUnavailable, build_client

log = logging.getLogger(__name__)

_client: LLMClient | None = None

TRANSLATE_SYSTEM = """You are a translator. Translate the user's text into English.

RULES:
1. Output ONLY the English translation. No preamble, no notes, no alternatives.
2. Translate meaning, not word by word. The input may be a mix of {language} and
   English (code-mixed), or {language} written in Roman letters — handle both.
3. Keep technical and scientific terms as they are if they are already English.
4. If the text is already entirely English, return it unchanged.
"""


def _get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = build_client()
    return _client


def to_english(text: str, language_code: str = "ta", client: LLMClient | None = None) -> str | None:
    """Translate regional / romanised text to English. `None` if no LLM is available."""
    if not (text or "").strip():
        return None

    lang = languages.get(language_code)
    try:
        result = (client or _get_client()).complete_text(
            system=TRANSLATE_SYSTEM.format(language=lang.english_name),
            user=text,
            max_tokens=1000,
        )
    except LLMUnavailable as exc:
        log.info("LLM translation fallback unavailable: %s", exc)
        return None

    return result.strip() or None
