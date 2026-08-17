"""Output contract for the tutor-explanation branch.

`TutorLesson` is what practice-generation and the frontend consume. Fields are
additive-only from here on — if you need a new one, add it with a default so
downstream branches keep parsing.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ConceptMapping(BaseModel):
    """One textbook concept tied to one concrete part of the analogy.

    Requirement 4: every concept in `RetrievedChunk.concepts` must appear here.
    The frontend can use this to highlight concept → analogy pairs side by side.
    """

    concept: str
    analogy_component: str


class LessonSource(str, Enum):
    """How the lesson was produced — surfaced for the demo and for judging."""

    GENERATED = "generated"
    """Free LLM generation that passed the grounding + analogy checks."""

    REGENERATED = "regenerated"
    """First attempt failed a check; the second attempt passed."""

    TEMPLATE_FALLBACK = "template_fallback"
    """Both attempts failed. Built directly from concepts — no free generation."""

    REFUSED = "refused"
    """Out of scope. No LLM call was made at all."""


class ExplanationOrigin(str, Enum):
    """How a regional-language explanation came to exist. Judges ask this."""

    COMPOSED = "composed"
    """Written directly in the language by the lesson model. The good path."""

    TRANSLATED_FROM_REGIONAL = "translated_from_regional"
    """IndicTrans2 indic→indic from the composed explanation — keeps the teacher voice."""

    TRANSLATED_FROM_ENGLISH = "translated_from_english"
    """IndicTrans2 en→indic from the English lesson. Flatter; the fallback."""

    TEMPLATE = "template"
    """Assembled from concepts. No generation, no translation."""


class LocalizedExplanation(BaseModel):
    """The lesson in one additional language."""

    language: str
    language_name: str
    text: str
    origin: ExplanationOrigin = ExplanationOrigin.TRANSLATED_FROM_REGIONAL


class LessonTrace(BaseModel):
    """Provenance for the live demo: what went in, what came back, what we checked.

    Optional and default-`None` so it never breaks a downstream consumer. The
    API only fills it when `?debug=true`.
    """

    chunk_ids: list[str] = Field(default_factory=list)
    chunk_text_sent_to_llm: str = ""
    concepts_sent_to_llm: list[str] = Field(default_factory=list)
    attempts: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    tamil_backtranslation: str | None = None
    tamil_missing_concepts: list[str] = Field(default_factory=list)
    tamil_regenerated: bool = False
    transliterated_query: str | None = None
    """Set when IndicXlit rewrote a romanised question into native script."""

    ai4bharat_models_used: list[str] = Field(default_factory=list)
    """Which AI4Bharat models actually ran for this lesson — not which are wired up."""

    notes: list[str] = Field(default_factory=list)


class TutorLesson(BaseModel):
    topic: str
    simple_explanation: str = ""
    """1-2 plain English sentences."""

    analogy: str = ""
    """Prose analogy in which every listed concept is mapped to something concrete."""

    analogy_map: list[ConceptMapping] = Field(default_factory=list)
    """The same analogy, decomposed concept-by-concept for validation and UI."""

    textbook_excerpt: str = ""
    """Near-verbatim quote from the source chunk."""

    source_pages: list[int] = Field(default_factory=list)
    tamil_explanation: str = ""
    """The primary regional explanation, independently composed — not a translation.

    Field name kept for the frontend + practice-generation contract. It holds
    whichever language `language` names; Tamil is the default and the tuned one.
    """

    language: str = "ta"
    language_name: str = "Tamil"
    regional_origin: ExplanationOrigin = ExplanationOrigin.COMPOSED
    """How `tamil_explanation` was produced. `composed` is the one that scores."""

    translations: list[LocalizedExplanation] = Field(default_factory=list)
    """Additional languages, produced by IndicTrans2 on request."""

    grounded: bool = True
    """False triggers the "outside your textbook" UX. No LLM call was made."""

    refusal_reason: str | None = None
    source: LessonSource = LessonSource.GENERATED
    trace: LessonTrace | None = None
