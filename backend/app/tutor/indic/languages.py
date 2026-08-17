"""The five languages this branch supports, and every code each AI4Bharat model wants.

One registry so a language is added in one place. IndicTrans2 uses FLORES-200
codes (`tam_Taml`), IndicConformer uses ISO-639-1 (`ta`), IndicXlit uses its own
short codes (which happen to match ISO-639-1 for these five).

Tamil is the demo language — the one "done excellently" per the problem
statement. The other four exist so the En→Indic and Indic→Indic models have
somewhere real to go, not because we claim equal quality in all five.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    """ISO-639-1. Our canonical key everywhere in the app."""

    name: str
    """Endonym, for the UI."""

    english_name: str
    flores: str
    """IndicTrans2 / FLORES-200 code."""

    asr: str
    """IndicConformer language code."""

    xlit: str
    """IndicXlit language code."""

    tts_voice: str
    """Speaker description for Indic Parler-TTS. Parler is prompted in English
    even when the speech is not — the description drives voice, pace and clarity."""

    local_analogy_hint: str
    """Everyday scenes a student in this region actually knows. Fed to the
    teacher prompt so the analogy is not a foreign example."""


ENGLISH_FLORES = "eng_Latn"

TAMIL = Language(
    code="ta",
    name="தமிழ்",
    english_name="Tamil",
    flores="tam_Taml",
    asr="ta",
    xlit="ta",
    tts_voice=(
        "Anushka speaks in a clear, warm and encouraging tone at a slightly slow pace, "
        "like a schoolteacher explaining to a class. Very clear audio, no background noise."
    ),
    local_analogy_hint="the kitchen (அடுப்பு, குக்கர், தோசைக்கல்), farming (வயல், நெல்), the tea shop, the school ground",
)

HINDI = Language(
    code="hi",
    name="हिन्दी",
    english_name="Hindi",
    flores="hin_Deva",
    asr="hi",
    xlit="hi",
    tts_voice=(
        "Divya speaks in a clear, warm and encouraging tone at a slightly slow pace, "
        "like a schoolteacher explaining to a class. Very clear audio, no background noise."
    ),
    local_analogy_hint="the kitchen (चूल्हा, कुकर, तवा), farming (खेत, गेहूँ), the chai stall, the school ground",
)

TELUGU = Language(
    code="te",
    name="తెలుగు",
    english_name="Telugu",
    flores="tel_Telu",
    asr="te",
    xlit="te",
    tts_voice=(
        "Prakash speaks in a clear, warm and encouraging tone at a slightly slow pace, "
        "like a schoolteacher explaining to a class. Very clear audio, no background noise."
    ),
    local_analogy_hint="the kitchen (పొయ్యి, కుక్కర్), farming (పొలం, వరి), the tea stall, the school ground",
)

KANNADA = Language(
    code="kn",
    name="ಕನ್ನಡ",
    english_name="Kannada",
    flores="kan_Knda",
    asr="kn",
    xlit="kn",
    tts_voice=(
        "Suresh speaks in a clear, warm and encouraging tone at a slightly slow pace, "
        "like a schoolteacher explaining to a class. Very clear audio, no background noise."
    ),
    local_analogy_hint="the kitchen (ಒಲೆ, ಕುಕ್ಕರ್), farming (ಹೊಲ, ಭತ್ತ), the tea stall, the school ground",
)

MALAYALAM = Language(
    code="ml",
    name="മലയാളം",
    english_name="Malayalam",
    flores="mal_Mlym",
    asr="ml",
    xlit="ml",
    tts_voice=(
        "Anjali speaks in a clear, warm and encouraging tone at a slightly slow pace, "
        "like a schoolteacher explaining to a class. Very clear audio, no background noise."
    ),
    local_analogy_hint="the kitchen (അടുപ്പ്, കുക്കർ), farming (പാടം, നെല്ല്), the tea shop, the school ground",
)

LANGUAGES: dict[str, Language] = {
    lang.code: lang for lang in (TAMIL, HINDI, TELUGU, KANNADA, MALAYALAM)
}

DEFAULT_LANGUAGE = TAMIL


def get(code: str | None) -> Language:
    """Look up a language, falling back to Tamil rather than raising.

    A bad `language` in a request should still teach the student something.
    """
    if not code:
        return DEFAULT_LANGUAGE
    return LANGUAGES.get(code.strip().casefold(), DEFAULT_LANGUAGE)


def supported() -> list[dict[str, str]]:
    """For the frontend's language picker."""
    return [
        {"code": lang.code, "name": lang.name, "english_name": lang.english_name}
        for lang in LANGUAGES.values()
    ]
