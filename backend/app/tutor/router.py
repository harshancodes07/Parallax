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
from app.tutor import lesson_generator, mock_rag_service
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
    query, transliterated = _normalise_query(request.query, request.language)

    grounding = _grounding_provider(request.page_id, query)
    lesson = lesson_generator.generate(
        grounding,
        query,
        language=request.language,
        translate_to=request.translate_to,
    )

    if transliterated and lesson.trace:
        lesson.trace.transliterated_query = query
        if "IndicXlit roman→native" not in lesson.trace.ai4bharat_models_used:
            lesson.trace.ai4bharat_models_used.append("IndicXlit roman→native")

    if not debug:
        lesson.trace = None
    return lesson


@router.post("/explain/preview")
def explain_preview(request: ExplainRequest) -> dict[str, str]:
    """Same lesson, rendered in the demo's on-screen format. Convenience for the frontend."""
    query, _ = _normalise_query(request.query, request.language)
    grounding = _grounding_provider(request.page_id, query)
    lesson = lesson_generator.generate(grounding, query, language=request.language)
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

    grounding = _grounding_provider(page_id, transcript)
    lesson = lesson_generator.generate(grounding, transcript, language=language)
    if lesson.trace:
        lesson.trace.ai4bharat_models_used.append("IndicConformer ASR")
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
    query, _ = _normalise_query(request.query, request.language)
    grounding = _grounding_provider(request.page_id, query)
    lesson = lesson_generator.generate(grounding, query, language=request.language)

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


@router.get("/capabilities")
def capabilities() -> dict[str, object]:
    """Which AI4Bharat models are actually loaded right now.

    Worth having on screen during the demo: it distinguishes "wired up" from
    "running", and it is the honest answer when a judge asks what is real.
    """
    return {"ai4bharat": indic_status(), "languages": languages.supported()}


# ------------------------------------------------------------------ helpers


def _normalise_query(query: str | None, language: str) -> tuple[str | None, bool]:
    """Run IndicXlit on a romanised question before it reaches retrieval.

    English questions are left alone — transliterating "how do plants make food"
    produces nonsense, so `normalise_query` checks for English markers first.
    """
    if not query:
        return query, False
    try:
        return transliterate.normalise_query(query, language)
    except Exception as exc:  # noqa: BLE001 - never fail a lesson over script handling
        log.warning("query normalisation skipped: %s", exc)
        return query, False
