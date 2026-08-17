"""AI4Bharat IndicXlit — Roman/Tanglish ↔ native script.

The problem it solves: a 14-year-old on a phone types "thavaram epdi saapdum"
or "prakasha samsleshanam", not "தாவரம் எப்படி சாப்பிடும்". That romanized text
is a bad query for a Tamil-script textbook page and a bad prompt for a Tamil
teacher persona. IndicXlit turns it into the native script first.

Also runs the other direction: native → Roman, which is what makes the Tanglish
lesson readable to a student who speaks Tamil but never learned to read it —
a real segment, and one nobody at this hackathon will have thought about.

Install: `pip install ai4bharat-transliteration`
"""

from __future__ import annotations

import logging
import re

from app.tutor.indic import languages
from app.tutor.indic.runtime import LazyComponent

log = logging.getLogger(__name__)

BEAM_WIDTH = 10

# Anything with a native-script character in it is already transliterated.
_NATIVE_SCRIPT = re.compile(
    r"[ऀ-ॿ஀-௿ఀ-౿ಀ-೿ഀ-ൿ]"
)
_LATIN_WORD = re.compile(r"[A-Za-z]")


def _load_roman_to_native():
    from ai4bharat.transliteration import XlitEngine

    return XlitEngine(beam_width=BEAM_WIDTH, rescore=True, src_script_type="roman")


def _load_native_to_roman():
    from ai4bharat.transliteration import XlitEngine

    return XlitEngine(beam_width=BEAM_WIDTH, rescore=False, src_script_type="indic")


ROMAN_TO_NATIVE = LazyComponent("IndicXlit roman→native", _load_roman_to_native, "TUTOR_XLIT")
NATIVE_TO_ROMAN = LazyComponent("IndicXlit native→roman", _load_native_to_roman, "TUTOR_XLIT")


def looks_romanised(text: str) -> bool:
    """Worth transliterating? Latin letters present, native script absent."""
    return bool(_LATIN_WORD.search(text or "")) and not _NATIVE_SCRIPT.search(text or "")


def to_native(text: str, language_code: str) -> str | None:
    """Tanglish → Tamil script. `None` if IndicXlit is unavailable."""
    if not (text or "").strip():
        return text
    engine = ROMAN_TO_NATIVE.get()
    if engine is None:
        return None
    lang = languages.get(language_code)
    try:
        return engine.translit_sentence(text, lang.xlit)
    except Exception as exc:  # noqa: BLE001
        log.warning("transliteration to %s failed: %s", lang.xlit, exc)
        return None


def to_roman(text: str, language_code: str) -> str | None:
    """Tamil script → Roman, for students who speak the language but don't read it."""
    if not (text or "").strip():
        return text
    engine = NATIVE_TO_ROMAN.get()
    if engine is None:
        return None
    lang = languages.get(language_code)
    try:
        return engine.translit_sentence(text, lang.xlit)
    except Exception as exc:  # noqa: BLE001
        log.warning("transliteration from %s failed: %s", lang.xlit, exc)
        return None


def normalise_query(text: str, language_code: str) -> tuple[str, bool]:
    """Prepare a student's typed question for retrieval.

    Returns `(query, was_transliterated)`. Only fires when the text really looks
    romanised — an English question like "how do plants make food" must be left
    exactly alone, because English queries are legitimate and transliterating one
    produces nonsense.
    """
    if not looks_romanised(text) or _is_probably_english(text):
        return text, False
    native = to_native(text, language_code)
    if not native:
        return text, False
    return native, True


# Short stop-word list: if the query is mostly these, it is English, not Tanglish.
_ENGLISH_MARKERS = {
    "the", "what", "why", "how", "is", "are", "does", "do", "explain", "of", "in",
    "and", "this", "that", "a", "an", "to", "for", "can", "you", "me", "tell",
}


def _is_probably_english(text: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z']+", text.casefold()) if w]
    if not words:
        return False
    hits = sum(1 for w in words if w in _ENGLISH_MARKERS)
    return hits / len(words) >= 0.34


def status() -> dict[str, str]:
    return {
        "roman_to_native": "loaded" if ROMAN_TO_NATIVE.available else (
            ROMAN_TO_NATIVE.reason or "not loaded yet"
        ),
        "native_to_roman": "loaded" if NATIVE_TO_ROMAN.available else (
            NATIVE_TO_ROMAN.reason or "not loaded yet"
        ),
    }
