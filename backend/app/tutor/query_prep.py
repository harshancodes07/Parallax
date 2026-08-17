"""Turn whatever the student typed into something retrieval can actually match.

The premise of the problem statement is that Indian students read **English-medium
textbooks** while thinking in their mother tongue. So the ingested page is English,
and a question asked in Tamil — or in Tanglish — has to reach English before it can
retrieve anything.

That means two different strings come out of one question:

    "thavaram epdi saapdum"
        │
        ├─[IndicXlit]──────────► "தாவரம் எப்படி சாப்பிடும்"   → what the teacher sees
        │                                   │
        └───────────────────────[IndicTrans2 indic→en]──────► "how does a plant eat"
                                                              → what retrieval searches

Getting this wrong is subtle and expensive: transliterating a question to Tamil
script and then searching an English page with it matches nothing, so the tutor
refuses a perfectly in-scope question. That is a worse failure than not
transliterating at all, because it looks like the grounding guardrail working.

Both models degrade independently. If IndicXlit is missing the question is used
as typed; if IndicTrans2 is missing the native-script question is used for
retrieval, which is the pre-existing behaviour rather than a regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.tutor import llm_translate
from app.tutor.llm_client import LLMRateLimited, LLMUnavailable
from app.tutor.indic import translate, transliterate

log = logging.getLogger(__name__)


@dataclass
class PreparedQuery:
    """One question, in the forms the different stages need."""

    original: str | None
    for_retrieval: str | None
    """English where possible — this is what the RAG layer searches with."""

    for_teaching: str | None
    """Native script where possible — this is what the teacher prompt sees, so the
    lesson answers the question the student actually asked."""

    transliterated: bool = False
    translated: bool = False
    models_used: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    """Set when translation was needed but transiently refused (quota, overload).

    The question could not be understood, so any "out of scope" verdict that
    follows is untrustworthy and the router turns it into a 503 instead."""


def prepare(query: str | None, language_code: str = "ta") -> PreparedQuery:
    if not (query or "").strip():
        return PreparedQuery(original=query, for_retrieval=query, for_teaching=query)

    native, transliterated = _transliterate(query, language_code)
    models: list[str] = []
    blocked: str | None = None
    if transliterated:
        models.append("IndicXlit roman→native")

    for_teaching = native
    for_retrieval = native
    translated = False

    # Only translate when the question is actually in the regional language. A
    # question typed in English is already in the corpus language — running it
    # through a translator would only add noise.
    #
    # The romanised case is the one that bit us: with IndicXlit unavailable the
    # text stays in Roman letters, so checking only "was transliterated" or "is
    # native script" skipped translation entirely and the question was refused as
    # out of scope. Tanglish needs translating whether or not we converted the
    # script first — and both translators accept it romanised.
    if transliterated or _is_native_script(native) or transliterate.is_romanised_regional(native):
        english = translate.to_english(native, language_code)
        if english:
            for_retrieval = english
            translated = True
            models.append("IndicTrans2 indic→en (query)")
        else:
            # IndicTrans2 can't run here (no Windows wheels). The LLM handles this
            # hop well, and getting the question into English matters more than
            # which engine does it.
            try:
                english = llm_translate.to_english(native, language_code)
            except LLMRateLimited as exc:
                # We could not work out what the student asked. Retrieval will
                # miss, and reporting that as "not in this chapter" would be a
                # lie about their question rather than the truth about our quota.
                log.warning("query translation rate-limited: %s", exc)
                blocked = str(exc)
                english = None
            except LLMUnavailable as exc:
                log.info("no LLM translator: %s", exc)
                english = None

            if english:
                for_retrieval = english
                translated = True
                models.append("LLM query translation (IndicTrans2 unavailable)")
            elif not blocked:
                log.info("no translator at all; retrieving with the question as typed")

    return PreparedQuery(
        original=query,
        for_retrieval=for_retrieval,
        for_teaching=for_teaching,
        transliterated=transliterated,
        translated=translated,
        models_used=models,
        blocked_reason=blocked,
    )


def _transliterate(query: str, language_code: str) -> tuple[str, bool]:
    try:
        return transliterate.normalise_query(query, language_code)
    except Exception as exc:  # noqa: BLE001 - never fail a lesson over script handling
        log.warning("query transliteration skipped: %s", exc)
        return query, False


def _is_native_script(text: str) -> bool:
    return bool(transliterate._NATIVE_SCRIPT.search(text or ""))
