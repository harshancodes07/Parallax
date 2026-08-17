"""AI4Bharat model layer.

Every model here is lazily loaded, individually kill-switchable, and returns
`None` rather than raising when it cannot load. The tutor works without any of
them — they add quality and modality, they are not load-bearing.

| Module            | Model                                        | Role |
|-------------------|----------------------------------------------|------|
| `translate`       | IndicTrans2 (indic→en, en→indic, indic→indic) | validation, fallback, cross-language |
| `transliterate`   | IndicXlit                                     | Tanglish ↔ native script |
| `asr`             | IndicConformer 600M                           | voice questions |
| `tts`             | Indic Parler-TTS                              | lesson read aloud |
"""

from app.tutor.indic import asr, languages, transliterate, translate, tts  # noqa: F401


def status() -> dict[str, dict[str, str]]:
    """What is actually loaded right now. Powers /api/tutor/capabilities."""
    return {
        "translate": translate.status(),
        "transliterate": transliterate.status(),
        "asr": asr.status(),
        "tts": tts.status(),
    }
