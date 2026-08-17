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


def prepare(query: str | None, language_code: str = "ta") -> PreparedQuery:
    if not (query or "").strip():
        return PreparedQuery(original=query, for_retrieval=query, for_teaching=query)

    native, transliterated = _transliterate(query, language_code)
    models: list[str] = []
    if transliterated:
        models.append("IndicXlit roman→native")

    for_teaching = native
    for_retrieval = native
    translated = False

    # Only translate when the question is actually in the regional language. A
    # question typed in English is already in the corpus language — running it
    # through a translator would only add noise.
    if transliterated or _is_native_script(native):
        english = translate.to_english(native, language_code)
        if english:
            for_retrieval = english
            translated = True
            models.append("IndicTrans2 indic→en (query)")
        else:
            # IndicTrans2 can't run here (no Windows wheels). The LLM handles this
            # hop well, and getting the question into English matters more than
            # which engine does it.
            english = llm_translate.to_english(native, language_code)
            if english:
                for_retrieval = english
                translated = True
                models.append("LLM query translation (IndicTrans2 unavailable)")
            else:
                log.info("no translator at all; retrieving with the question as typed")

    return PreparedQuery(
        original=query,
        for_retrieval=for_retrieval,
        for_teaching=for_teaching,
        transliterated=transliterated,
        translated=translated,
        models_used=models,
    )


def _transliterate(query: str, language_code: str) -> tuple[str, bool]:
    try:
        return transliterate.normalise_query(query, language_code)
    except Exception as exc:  # noqa: BLE001 - never fail a lesson over script handling
        log.warning("query transliteration skipped: %s", exc)
        return query, False


def _is_native_script(text: str) -> bool:
    return bool(transliterate._NATIVE_SCRIPT.search(text or ""))
