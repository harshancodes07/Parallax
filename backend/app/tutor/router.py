"""FastAPI surface for the tutor branch.

    POST /api/tutor/explain        text in  -> TutorLesson
    POST /api/tutor/listen         audio in -> transcript (IndicConformer)
    POST /api/tutor/ask            audio in -> TutorLesson (listen + explain, one call)
    POST /api/tutor/speak          text in  -> WAV        (Indic Parler-TTS)
    POST /api/tutor/transliterate  Tanglish -> native script (IndicXlit)
    GET  /api/tutor/languages      the five supported languages
    GET  /api/tutor/capabilities   which AI4Bharat models actually loaded

Retrieval is injected through `set_grounding_provider` so that at merge time the
`feat/rag-grounding` owner swaps one function and nothing else in this file moves.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.shared.schemas import GroundingResult
from app.tutor import lesson_cache, lesson_generator, mock_rag_service, query_prep
from app.tutor.indic import asr, languages, transliterate, tts
from app.tutor.indic import status as indic_status
from app.tutor.schemas import TutorLesson

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tutor", tags=["tutor"])

GroundingProvider = Callable[[str, "str | None"], GroundingResult]

# Default = the mock. `main` (or the RAG branch) calls set_grounding_provider() at startup.
_grounding_provider: GroundingProvider = mock_rag_service.fetch_grounding


def set_grounding_provider(provider: GroundingProvider) -> None:
    """Integration seam. Pass the real retriever here; signature must match the mock."""
    global _grounding_provider
    _grounding_provider = provider


class ExplainRequest(BaseModel):
    page_id: str = Field(..., description="Identifier of the ingested textbook page")
    query: str | None = Field(
        None, description="What the student asked. Null = 'teach me this page'."
    )
    language: str = Field("ta", description="Composed regional language: ta | hi | te | kn | ml")
    translate_to: list[str] = Field(
        default_factory=list,
        description="Extra languages via IndicTrans2. Translated, not composed.",
    )


class TransliterateRequest(BaseModel):
    text: str
    language: str = "ta"
    direction: str = Field("to_native", description="to_native | to_roman")


class SpeakRequest(BaseModel):
    text: str
    language: str = "ta"
    description: str | None = Field(
        None, description="Override the Parler-TTS voice description"
    )


@router.post("/explain", response_model=TutorLesson)
def explain(
    request: ExplainRequest,
    debug: bool = Query(
        False, description="Include the grounding trace (source text, checks, backtranslation)"
    ),
) -> TutorLesson:
    # Checked before any model call: re-asking the same question during a demo
    # must not spend quota (a Gemini free tier is ~5 lessons' worth).
    cache_key = lesson_cache.key(
        request.page_id, request.query, request.language, request.translate_to
    )
    cached = lesson_cache.get(cache_key)
    if cached is not None:
        if not debug:
            cached.trace = None
        return cached

    prepared = query_prep.prepare(request.query, request.language)

    # Retrieval searches an English page; the teacher answers the question as asked.
    grounding = _grounding_provider(request.page_id, prepared.for_retrieval)
    lesson = lesson_generator.generate(
        grounding,
        prepared.for_teaching,
        language=request.language,
        translate_to=request.translate_to,
    )

    _guard_untrustworthy_refusal(lesson, prepared)
    lesson_cache.put(cache_key, lesson)
    _record_query_prep(lesson, prepared)
    if not debug:
        lesson.trace = None
    return lesson


@router.post("/explain/preview")
def explain_preview(request: ExplainRequest) -> dict[str, str]:
    """Same lesson, rendered in the demo's on-screen format. Convenience for the frontend."""
    prepared = query_prep.prepare(request.query, request.language)
    grounding = _grounding_provider(request.page_id, prepared.for_retrieval)
    lesson = lesson_generator.generate(
        grounding, prepared.for_teaching, language=request.language
    )
    return {"display": lesson_generator.render(lesson)}


# ------------------------------------------------------------------ voice in


@router.post("/listen")
async def listen(
    audio: UploadFile = File(..., description="Recorded question (wav/mp3/flac/ogg)"),
    language: str = Form("ta"),
) -> dict[str, str]:
    """IndicConformer speech → text. The first half of the hands-free loop."""
    transcript = asr.transcribe(await audio.read(), language)
    if transcript is None:
        raise HTTPException(
            status_code=503,
            detail="Speech recognition is unavailable — install torchaudio and let "
            "IndicConformer download, or type the question instead.",
        )
    return {"transcript": transcript, "language": languages.get(language).code}


@router.post("/ask", response_model=TutorLesson)
async def ask(
    audio: UploadFile = File(...),
    page_id: str = Form(...),
    language: str = Form("ta"),
    debug: bool = Query(False),
) -> TutorLesson:
    """Voice question in, full lesson out — ASR and explanation in one round trip.

    Separate from `/listen` on purpose: the frontend usually wants to show the
    transcript for confirmation first. This endpoint is for the hands-free path
    where nobody is looking at the screen.
    """
    transcript = asr.transcribe(await audio.read(), language)
    if transcript is None:
        raise HTTPException(status_code=503, detail="Speech recognition is unavailable")

    # The transcript comes back in native script, so it needs the same journey to
    # English as a typed regional question before it can retrieve anything.
    prepared = query_prep.prepare(transcript, language)
    grounding = _grounding_provider(page_id, prepared.for_retrieval)
    lesson = lesson_generator.generate(
        grounding, prepared.for_teaching, language=language
    )

    if lesson.trace:
        lesson.trace.ai4bharat_models_used.append("IndicConformer ASR")
    _record_query_prep(lesson, prepared)
    if not debug:
        lesson.trace = None
    return lesson


# ------------------------------------------------------------------ voice out


@router.post("/speak")
def speak(request: SpeakRequest) -> Response:
    """Indic Parler-TTS text → speech. Returns WAV bytes."""
    audio = tts.speak(request.text, request.language, request.description)
    if audio is None:
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech is unavailable — install parler-tts and soundfile.",
        )
    return Response(content=audio, media_type="audio/wav")


@router.post("/explain/speak")
def explain_and_speak(request: ExplainRequest) -> Response:
    """The whole hands-free output path: lesson composed, then read aloud."""
    prepared = query_prep.prepare(request.query, request.language)
    grounding = _grounding_provider(request.page_id, prepared.for_retrieval)
    lesson = lesson_generator.generate(
        grounding, prepared.for_teaching, language=request.language
    )

    audio = tts.speak(lesson.tamil_explanation, request.language)
    if audio is None:
        raise HTTPException(status_code=503, detail="Text-to-speech is unavailable")
    return Response(content=audio, media_type="audio/wav")


# ------------------------------------------------------------------ script handling


@router.post("/transliterate")
def transliterate_text(request: TransliterateRequest) -> dict[str, str]:
    """IndicXlit both ways.

    `to_native`: a student types "thavaram enna" — make it Tamil script.
    `to_roman`:  show the Tamil lesson in Roman letters, for students who speak
                 the language but never learned to read it.
    """
    if request.direction == "to_roman":
        result = transliterate.to_roman(request.text, request.language)
    else:
        result = transliterate.to_native(request.text, request.language)

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Transliteration is unavailable — pip install ai4bharat-transliteration",
        )
    return {"text": result, "direction": request.direction, "language": request.language}


# ------------------------------------------------------------------ introspection


@router.get("/languages")
def list_languages() -> dict[str, object]:
    return {"default": "ta", "languages": languages.supported()}


@router.post("/cache/clear")
def clear_cache() -> dict[str, object]:
    """Force the next question to hit the model again. Useful after a prompt change."""
    lesson_cache.clear()
    return lesson_cache.stats()


@router.get("/capabilities")
def capabilities() -> dict[str, object]:
    """Which AI4Bharat models are actually loaded right now.

    Worth having on screen during the demo: it distinguishes "wired up" from
    "running", and it is the honest answer when a judge asks what is real.
    """
    return {
        "ai4bharat": indic_status(),
        "languages": languages.supported(),
        "lesson_cache": lesson_cache.stats(),
    }


# ------------------------------------------------------------------ helpers


def _guard_untrustworthy_refusal(
    lesson: TutorLesson, prepared: query_prep.PreparedQuery
) -> None:
    """Never let a quota failure be reported to a student as "not in this chapter".

    If the question needed translating and the translator was rate-limited, we
    never worked out what was asked — so retrieval missed for our reasons, not
    theirs. Saying "that isn't in this chapter" would be a false statement about
    the textbook, and it is the one kind of wrong answer this branch exists to
    prevent. A 503 is the honest reply: come back in a moment.
    """
    if lesson.grounded or not prepared.blocked_reason:
        return
    log.warning("suppressing refusal: %s", prepared.blocked_reason)
    raise HTTPException(
        status_code=503,
        detail=(
            "I couldn't read your question just now — the language service is busy. "
            "Please try again in a moment. (This is not a judgement about your "
            "textbook page.)"
        ),
    )


def _record_query_prep(lesson: TutorLesson, prepared: query_prep.PreparedQuery) -> None:
    """Put the query rewriting in the trace so the demo can show what was searched."""
    if not lesson.trace:
        return
    if prepared.transliterated:
        lesson.trace.transliterated_query = prepared.for_teaching
    if prepared.translated:
        lesson.trace.notes.append(
            f"retrieved with the translated query: {prepared.for_retrieval!r}"
        )
    for model in prepared.models_used:
        if model not in lesson.trace.ai4bharat_models_used:
            lesson.trace.ai4bharat_models_used.append(model)
