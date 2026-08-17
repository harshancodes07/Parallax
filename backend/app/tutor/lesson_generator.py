"""Orchestration: check grounding -> generate -> validate -> return.

`generate()` is the only function the rest of the app should call. Whoever wires
`main` calls one thing, and every guardrail sits behind it.

Control flow, in order:

    is_in_scope == False ─────────────────────────► refusal (NO LLM call is made)
    English lesson  ── fails checks ── retry once ── still failing ──► template
    Regional lesson ── concepts lost ── retry once ── still lost ────► kept + flagged
                    ── model unavailable ─────────► IndicTrans2 en→indic ─► template
    extra languages ─────────────────────────────► IndicTrans2 indic→indic

The regional explanation is **composed**, not translated. IndicTrans2 en→indic
exists here only as a fallback and for languages beyond the composed one — if it
ever becomes the primary path, the thing that makes this branch score is gone.
"""

from __future__ import annotations

import logging
import re
import uuid

from app.shared.schemas import GroundingResult, RetrievedChunk
from app.tutor import grounding_check, prompts, tamil_quality
from app.tutor.indic import languages, translate
from app.tutor.llm_client import LESSON_SCHEMA, LLMClient, LLMUnavailable
from app.tutor.schemas import (
    ConceptMapping,
    ExplanationOrigin,
    LessonSource,
    LessonTrace,
    LocalizedExplanation,
    TutorLesson,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

REFUSAL_TEXT = (
    "That isn't in this chapter. I can only teach from the page you uploaded — "
    "try asking about something on it, or upload the page that covers this."
)
REFUSAL_TAMIL = (
    "இது நீங்க upload பண்ண paragraph-ல இல்லை. நான் அந்த page-ல இருக்கறத மட்டும் தான் "
    "சொல்ல முடியும். அந்த page-ல இருக்கற ஏதாவது கேளுங்க, இல்லைன்னா சரியான page-ஐ upload பண்ணுங்க."
)

_default_client = LLMClient()


def generate(
    grounding: GroundingResult | RetrievedChunk,
    query: str | None = None,
    *,
    language: str = "ta",
    translate_to: list[str] | None = None,
    client: LLMClient | None = None,
) -> TutorLesson:
    """Turn a grounded retrieval into a `TutorLesson`.

    Accepts a `GroundingResult` (the real upstream contract) or a bare
    `RetrievedChunk` for convenience during development.

    `language` picks the composed regional language; `translate_to` asks for
    additional languages via IndicTrans2.
    """
    client = client or _default_client
    lang = languages.get(language)
    grounding = _as_grounding(grounding, query)
    request_id = uuid.uuid4().hex[:8]

    # ---------------------------------------------------------------- hard guardrail
    # This branch runs BEFORE any LLM object is touched. Out of scope means we do not
    # generate at all — not "we generate and hope the prompt holds".
    if not grounding.is_in_scope or not grounding.chunks:
        log.info("[%s] refusing: out of scope (query=%r)", request_id, grounding.query)
        return _refusal(lang)

    chunks = sorted(grounding.chunks, key=lambda c: c.similarity_score, reverse=True)
    source_text = "\n\n".join(c.text for c in chunks)
    concepts = _unique([c for chunk in chunks for c in chunk.concepts])
    pages = _unique([chunk.page_number for chunk in chunks])
    primary = chunks[0]
    effective_query = grounding.query or primary.chapter_title or "Explain this page"

    trace = LessonTrace(
        chunk_ids=[c.chunk_id for c in chunks],
        chunk_text_sent_to_llm=source_text,
        concepts_sent_to_llm=concepts,
    )
    # The demo shows this line next to the answer: "generated from Page 42".
    log.info(
        "[%s] grounded generation | lang=%s pages=%s chunks=%s concepts=%s"
        "\n--- source sent to LLM ---\n%s\n---",
        request_id,
        lang.code,
        pages,
        trace.chunk_ids,
        concepts,
        source_text,
    )

    lesson_json, source_kind, problems = _generate_english(
        client, primary, source_text, concepts, effective_query, trace
    )

    if lesson_json is None:
        log.warning("[%s] falling back to template: %s", request_id, problems)
        trace.notes.append(f"template fallback: {'; '.join(problems) or 'generation unavailable'}")
        lesson = _template_lesson(primary, concepts, pages, lang)
        lesson.trace = trace
        return lesson

    regional_text, origin, report = _generate_regional(
        client, primary, concepts, effective_query, lang, lesson_json, trace
    )
    trace.tamil_backtranslation = report.backtranslation
    trace.tamil_missing_concepts = report.missing
    if report.note:
        trace.notes.append(report.note)
    if report.backtranslation_available:
        _record_model(trace, "IndicTrans2 indic→en (validation)")

    analogy_map = [
        ConceptMapping(concept=m["concept"], analogy_component=m["analogy_component"])
        for m in lesson_json.get("analogy_map", [])
        if m.get("concept") and m.get("analogy_component")
    ]

    lesson = TutorLesson(
        topic=lesson_json.get("topic") or (primary.chapter_title or effective_query),
        simple_explanation=lesson_json.get("simple_explanation", ""),
        analogy=lesson_json.get("analogy", ""),
        analogy_map=analogy_map,
        textbook_excerpt=lesson_json.get("textbook_excerpt", ""),
        source_pages=pages,
        tamil_explanation=regional_text,
        language=lang.code,
        language_name=lang.english_name,
        regional_origin=origin,
        grounded=True,
        refusal_reason=None,
        source=source_kind,
        trace=trace,
    )

    lesson.translations = _translate_lesson(lesson, translate_to, trace)
    return lesson


# ------------------------------------------------------------------ refusal


def _refusal(lang) -> TutorLesson:
    """Static refusal. No LLM call, and no model *load* either.

    If IndicTrans2 en→indic happens to be warm already we localise the refusal
    for free; we never trigger a load on this path, because the refusal has to be
    instant — that responsiveness is part of why it reads as a deliberate rule
    rather than a failure.
    """
    if lang.code == "ta":
        regional = REFUSAL_TAMIL
    elif translate.EN_INDIC.available:
        regional = translate.from_english(REFUSAL_TEXT, lang.code) or REFUSAL_TEXT
    else:
        regional = REFUSAL_TEXT

    return TutorLesson(
        topic="Outside this chapter",
        grounded=False,
        refusal_reason=REFUSAL_TEXT,
        tamil_explanation=regional,
        language=lang.code,
        language_name=lang.english_name,
        regional_origin=ExplanationOrigin.TEMPLATE,
        source=LessonSource.REFUSED,
        trace=LessonTrace(notes=["out of scope — no LLM call made"]),
    )


# ------------------------------------------------------------------ English generation


def _generate_english(
    client: LLMClient,
    primary: RetrievedChunk,
    source_text: str,
    concepts: list[str],
    query: str,
    trace: LessonTrace,
) -> tuple[dict | None, LessonSource, list[str]]:
    """Generate, validate, retry once. Returns `(lesson_json | None, source, problems)`."""
    user_prompt = prompts.ENGLISH_LESSON_USER.format(
        page_number=primary.page_number,
        chapter_suffix=prompts.chapter_suffix(primary.chapter_title),
        chunk_text=source_text,
        concepts=prompts.format_concepts(concepts),
        query=query,
    )

    problems: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        trace.attempts = attempt
        prompt = user_prompt
        if problems:
            prompt += prompts.REGENERATION_SUFFIX.format(
                problems="\n".join(f"- {p}" for p in problems)
            )

        try:
            lesson_json = client.complete_json(
                system=prompts.ENGLISH_LESSON_SYSTEM, user=prompt, schema=LESSON_SCHEMA
            )
        except LLMUnavailable as exc:
            return None, LessonSource.TEMPLATE_FALLBACK, [str(exc)]

        problems = _validate(client, lesson_json, source_text, concepts, trace)
        if not problems:
            return (
                lesson_json,
                LessonSource.GENERATED if attempt == 1 else LessonSource.REGENERATED,
                [],
            )
        log.info("attempt %s rejected: %s", attempt, problems)

    return None, LessonSource.TEMPLATE_FALLBACK, problems


def _validate(
    client: LLMClient,
    lesson_json: dict,
    source_text: str,
    concepts: list[str],
    trace: LessonTrace,
) -> list[str]:
    """Deterministic checks first (free), then the LLM claim audit (costs a call)."""
    problems: list[str] = []

    missing = grounding_check.check_analogy_coverage(
        concepts, lesson_json.get("analogy", ""), lesson_json.get("analogy_map", [])
    )
    trace.missing_concepts = missing
    if missing:
        problems.append(
            "the analogy does not map these concepts to a concrete component: "
            + ", ".join(missing)
        )

    excerpt = lesson_json.get("textbook_excerpt", "")
    if not grounding_check.check_excerpt(excerpt, source_text):
        problems.append(
            "textbook_excerpt is not a verbatim quote from the provided page — copy it exactly"
        )

    if problems:
        return problems  # don't spend an audit call on output we already know is bad

    explanation = "\n".join(
        [lesson_json.get("simple_explanation", ""), lesson_json.get("analogy", "")]
    )
    unsupported = grounding_check.check_claims(client, source_text, explanation)
    trace.unsupported_claims = unsupported
    if unsupported:
        problems.append("these claims are not in the source: " + "; ".join(unsupported))
    return problems


# ------------------------------------------------------------------ regional generation


def _generate_regional(
    client: LLMClient,
    primary: RetrievedChunk,
    concepts: list[str],
    query: str,
    lang,
    lesson_json: dict,
    trace: LessonTrace,
) -> tuple[str, ExplanationOrigin, tamil_quality.TamilQualityReport]:
    """Compose in the regional language, validate with IndicTrans2, retry once.

    Only if composition is impossible do we fall back to en→indic translation —
    the flat, obviously-machine-translated path we are otherwise trying to beat.
    """
    user_prompt = prompts.teacher_user(lang).format(
        page_number=primary.page_number,
        chapter_suffix=prompts.chapter_suffix(primary.chapter_title),
        chunk_text=primary.text,
        concepts=prompts.format_concepts(concepts),
        query=query,
    )
    system = prompts.teacher_system(lang)

    try:
        regional_text = client.complete_text(system=system, user=user_prompt)
    except LLMUnavailable as exc:
        log.warning("regional composition unavailable: %s", exc)
        return _translated_fallback(lesson_json, concepts, lang, trace, reason=str(exc))

    report = tamil_quality.evaluate(regional_text, concepts, lang.code)
    if report.passed:
        return regional_text, ExplanationOrigin.COMPOSED, report

    log.info("regional regeneration triggered, missing=%s", report.missing)
    trace.tamil_regenerated = True
    retry_prompt = user_prompt + prompts.TAMIL_RETRY_SUFFIX.format(
        missing=", ".join(report.missing)
    )
    try:
        retry_text = client.complete_text(system=system, user=retry_prompt)
    except LLMUnavailable as exc:
        log.warning("regional regeneration failed: %s", exc)
        return regional_text, ExplanationOrigin.COMPOSED, report

    retry_report = tamil_quality.evaluate(retry_text, concepts, lang.code)
    # Keep whichever attempt lost fewer concepts.
    if len(retry_report.missing) <= len(report.missing):
        return retry_text, ExplanationOrigin.COMPOSED, retry_report
    return regional_text, ExplanationOrigin.COMPOSED, report


def _translated_fallback(
    lesson_json: dict, concepts: list[str], lang, trace: LessonTrace, reason: str
) -> tuple[str, ExplanationOrigin, tamil_quality.TamilQualityReport]:
    """IndicTrans2 en→indic on the English lesson. Better than a stub, worse than composed."""
    english = " ".join(
        part
        for part in (lesson_json.get("simple_explanation"), lesson_json.get("analogy"))
        if part
    )
    translated = translate.from_english(english, lang.code)

    if translated:
        _record_model(trace, "IndicTrans2 en→indic (fallback)")
        trace.notes.append(f"regional text machine-translated ({reason})")
        return (
            translated,
            ExplanationOrigin.TRANSLATED_FROM_ENGLISH,
            tamil_quality.TamilQualityReport(note="machine-translated, not composed"),
        )

    trace.notes.append(f"regional template used ({reason})")
    return (
        _regional_template(concepts, lang),
        ExplanationOrigin.TEMPLATE,
        tamil_quality.TamilQualityReport(note="regional template used"),
    )


def _translate_lesson(
    lesson: TutorLesson, targets: list[str] | None, trace: LessonTrace
) -> list[LocalizedExplanation]:
    """Additional languages via IndicTrans2.

    Prefers indic→indic from the composed text: the composed explanation is warm
    spoken teacher talk, the English `simple_explanation` is flat and factual, so
    translating from the former keeps far more of what makes this branch score.
    """
    out: list[LocalizedExplanation] = []
    for code in targets or []:
        target = languages.get(code)
        if target.code == lesson.language:
            continue

        text = None
        origin = ExplanationOrigin.TRANSLATED_FROM_REGIONAL

        if lesson.regional_origin is ExplanationOrigin.COMPOSED and lesson.tamil_explanation:
            text = translate.between_indic(
                lesson.tamil_explanation, source_code=lesson.language, target_code=target.code
            )
            if text:
                _record_model(trace, "IndicTrans2 indic→indic")

        if not text:
            english = " ".join(p for p in (lesson.simple_explanation, lesson.analogy) if p)
            text = translate.from_english(english, target.code)
            origin = ExplanationOrigin.TRANSLATED_FROM_ENGLISH
            if text:
                _record_model(trace, "IndicTrans2 en→indic")

        if not text:
            trace.notes.append(f"could not produce {target.english_name}: IndicTrans2 unavailable")
            continue

        out.append(
            LocalizedExplanation(
                language=target.code,
                language_name=target.english_name,
                text=text,
                origin=origin,
            )
        )
    return out


# ------------------------------------------------------------------ template fallback


def _template_lesson(
    primary: RetrievedChunk, concepts: list[str], pages: list[int], lang
) -> TutorLesson:
    """Last resort: assembled from the page and the concept list. No free generation.

    Everything here is either copied from the textbook or a fixed string, so it
    cannot hallucinate. It is plainer than a generated lesson — that is the trade.
    """
    sentences = _sentences(primary.text)
    simple = " ".join(sentences[:2]) if sentences else primary.text[:300]
    listed = ", ".join(concepts) if concepts else "the ideas on this page"

    mapping = [
        ConceptMapping(
            concept=concept,
            analogy_component=f"step {i} in the same recipe, on page {primary.page_number}",
        )
        for i, concept in enumerate(concepts, start=1)
    ]
    analogy = (
        f"Think of this page as one recipe with {len(concepts) or 'a few'} steps: "
        f"{listed}. Each one is a step that has to happen for the next to work — "
        "read them in order, like a recipe, not as separate facts."
    )

    return TutorLesson(
        topic=primary.chapter_title or f"Page {primary.page_number}",
        simple_explanation=simple,
        analogy=analogy,
        analogy_map=mapping,
        textbook_excerpt=sentences[0] if sentences else primary.text[:200],
        source_pages=pages,
        tamil_explanation=_regional_template(concepts, lang, primary.page_number),
        language=lang.code,
        language_name=lang.english_name,
        regional_origin=ExplanationOrigin.TEMPLATE,
        grounded=True,
        source=LessonSource.TEMPLATE_FALLBACK,
    )


def _regional_template(concepts: list[str], lang, page_number: int | None = None) -> str:
    """Fixed-string fallback. Tamil is hand-written; others translate if a model is warm."""
    listed = ", ".join(concepts) if concepts else ""
    if lang.code == "ta":
        page = f" ({page_number})" if page_number else ""
        return (
            f"இந்த page-ல{page} நாம படிக்கறது: {listed or 'இந்த பாடம்'}. "
            "இந்த வார்த்தைகள் ஒவ்வொன்னும் ஒரு step. புத்தகத்துல இருக்கற வரிகளை அப்படியே "
            "படிச்சு, ஒவ்வொரு step-ஆ யோசிங்க."
        )

    english = f"On this page we are learning about: {listed}. Read the lines in your textbook and think about each one in order."
    if translate.EN_INDIC.available:
        return translate.from_english(english, lang.code) or english
    return english


# ------------------------------------------------------------------ display helper


def render(lesson: TutorLesson) -> str:
    """The demo format from the spec. Handy for CLI checks and the judges' screen."""
    if not lesson.grounded:
        return f"🚫 {lesson.refusal_reason}\n{lesson.tamil_explanation}"
    pages = ", ".join(str(p) for p in lesson.source_pages)
    return (
        f"🌱 {lesson.topic}\n"
        f"Simple explanation: {lesson.simple_explanation}\n"
        f"Think of it like this: {lesson.analogy}\n"
        f'From your textbook: "{lesson.textbook_excerpt}" — Page {pages}\n'
        f"{lesson.language_name} explanation: {lesson.tamil_explanation}"
    )


# ------------------------------------------------------------------ small helpers


def _record_model(trace: LessonTrace, name: str) -> None:
    if name not in trace.ai4bharat_models_used:
        trace.ai4bharat_models_used.append(name)


def _as_grounding(
    grounding: GroundingResult | RetrievedChunk, query: str | None
) -> GroundingResult:
    if isinstance(grounding, RetrievedChunk):
        return GroundingResult(
            query=query or grounding.chapter_title or "",
            chunks=[grounding],
            is_in_scope=True,
        )
    if query:
        grounding = grounding.model_copy(update={"query": query})
    return grounding


def _unique(items: list) -> list:
    seen: set = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
