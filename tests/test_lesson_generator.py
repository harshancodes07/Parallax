"""Merge gate #2: the happy path, the guardrails, and the Tamil concept check."""

from __future__ import annotations

import copy

import pytest

from app.tutor import grounding_check, lesson_generator, mock_rag_service, tamil_quality
from app.tutor.schemas import LessonSource
from tests.conftest import GOOD_LESSON, GOOD_TAMIL, FakeClient

CHUNK = mock_rag_service.PHOTOSYNTHESIS


@pytest.fixture
def grounding():
    return mock_rag_service.fetch_grounding("pg42", "how do plants make their food")


# ------------------------------------------------------------------ happy path


def test_generates_a_complete_lesson(fake_client):
    lesson = lesson_generator.generate(CHUNK, "how do plants make food", client=fake_client)

    assert lesson.grounded is True
    assert lesson.source is LessonSource.GENERATED
    assert lesson.refusal_reason is None
    assert lesson.topic == "Photosynthesis"
    assert lesson.source_pages == [42]
    assert lesson.tamil_explanation == GOOD_TAMIL
    assert len(fake_client.json_calls) == 1
    assert len(fake_client.tamil_calls) == 1


def test_tamil_is_generated_from_the_textbook_not_from_the_english_explanation(fake_client):
    """Requirement 2: the Tamil call must not see the English output.

    If the English explanation ever leaks into the Tamil prompt, the model will
    translate it, and translation is exactly what loses the 25% language score.
    """
    lesson_generator.generate(CHUNK, "how do plants make food", client=fake_client)

    tamil_prompt = fake_client.tamil_calls[0]["user"]
    assert CHUNK.text in tamil_prompt
    assert GOOD_LESSON["simple_explanation"] not in tamil_prompt
    assert GOOD_LESSON["analogy"] not in tamil_prompt


def test_every_concept_is_mapped_to_an_analogy_component(fake_client):
    lesson = lesson_generator.generate(CHUNK, None, client=fake_client)

    mapped = {m.concept for m in lesson.analogy_map}
    assert set(CHUNK.concepts) <= mapped
    assert all(m.analogy_component.strip() for m in lesson.analogy_map)


def test_trace_carries_the_exact_source_sent_to_the_llm(fake_client):
    lesson = lesson_generator.generate(CHUNK, None, client=fake_client)

    assert lesson.trace is not None
    assert lesson.trace.chunk_text_sent_to_llm == CHUNK.text
    assert lesson.trace.chunk_ids == [CHUNK.chunk_id]
    assert lesson.trace.concepts_sent_to_llm == CHUNK.concepts


def test_render_produces_the_demo_format(fake_client):
    lesson = lesson_generator.generate(CHUNK, None, client=fake_client)

    display = lesson_generator.render(lesson)
    assert display.startswith("🌱 Photosynthesis")
    assert "Simple explanation:" in display
    assert "Think of it like this:" in display
    assert "— Page 42" in display
    assert "Tamil explanation:" in display


# ------------------------------------------------------------------ analogy guardrail


def test_incomplete_analogy_triggers_one_regeneration():
    bad = copy.deepcopy(GOOD_LESSON)
    bad["analogy"] = "It is like cooking food in a kitchen."
    bad["analogy_map"] = [{"concept": "chlorophyll", "analogy_component": "the cook"}]

    client = FakeClient(json_responses=[bad, GOOD_LESSON], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert len(client.json_calls) == 2
    assert lesson.source is LessonSource.REGENERATED
    # The retry prompt must name what was wrong, not just say "try again".
    assert "oxygen" in client.json_calls[1]["user"]


def test_two_bad_analogies_fall_back_to_a_template():
    bad = copy.deepcopy(GOOD_LESSON)
    bad["analogy"] = "It is like a process."
    bad["analogy_map"] = []

    client = FakeClient(json_responses=[bad, bad], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert lesson.source is LessonSource.TEMPLATE_FALLBACK
    assert lesson.grounded is True
    # The fallback is built from concepts, so coverage still holds.
    assert {m.concept for m in lesson.analogy_map} == set(CHUNK.concepts)
    # And its excerpt is copied from the page, so it cannot be invented.
    assert lesson.textbook_excerpt in CHUNK.text


# ------------------------------------------------------------------ grounding guardrail


def test_fabricated_excerpt_is_rejected():
    bad = copy.deepcopy(GOOD_LESSON)
    bad["textbook_excerpt"] = "Plants were first studied by Jan Baptist van Helmont in 1648."

    client = FakeClient(json_responses=[bad, GOOD_LESSON], tamil_responses=[GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert len(client.json_calls) == 2
    assert lesson.source is LessonSource.REGENERATED
    assert lesson.textbook_excerpt == GOOD_LESSON["textbook_excerpt"]


def test_unsupported_claim_flagged_by_the_self_check_triggers_regeneration():
    client = FakeClient(
        json_responses=[GOOD_LESSON, GOOD_LESSON],
        tamil_responses=[GOOD_TAMIL],
        claim_responses=[
            "- Photosynthesis was discovered in 1779 by Jan Ingenhousz.",
            "NONE",
        ],
    )

    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert len(client.json_calls) == 2
    assert len(client.claim_calls) == 2
    assert lesson.source is LessonSource.REGENERATED


def test_self_check_saying_none_is_accepted_first_time():
    client = FakeClient(
        json_responses=[GOOD_LESSON], tamil_responses=[GOOD_TAMIL], claim_responses=["NONE"]
    )

    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert lesson.source is LessonSource.GENERATED
    assert lesson.trace.unsupported_claims == []


def test_llm_unavailable_falls_back_to_template_rather_than_failing():
    client = FakeClient(json_responses=[], tamil_responses=[])

    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert lesson.source is LessonSource.TEMPLATE_FALLBACK
    assert lesson.grounded is True
    assert lesson.simple_explanation
    assert lesson.tamil_explanation


# ------------------------------------------------------------------ Tamil quality


BACKTRANSLATION = (
    "Look, think of the leaf as a kitchen. The chlorophyll inside it is the cook. "
    "Sunlight is the stove fire. The carbon dioxide coming from the air and the "
    "waters rising from the roots are our ingredients. From these the plant prepares "
    "glucose. The leftover oxygen goes out. That is what we breathe."
)


def test_backtranslated_tamil_concepts_fuzzy_match_the_chunk_concepts(monkeypatch):
    """Spec test #2: round-trip the Tamil through IndicTrans2 and check the concepts survived."""
    monkeypatch.setattr(
        tamil_quality.translate, "to_english", lambda text, lang: BACKTRANSLATION
    )

    report = tamil_quality.evaluate(GOOD_TAMIL, CHUNK.concepts, "ta")

    assert report.backtranslation_available is True
    assert report.passed, f"concepts lost in backtranslation: {report.missing}"


def test_tamil_missing_a_concept_triggers_exactly_one_regeneration(monkeypatch):
    thin_tamil = "செடி sunlight-ஐ பயன்படுத்தி சாப்பாடு தயார் பண்ணுது."
    monkeypatch.setattr(
        tamil_quality.translate,
        "to_english",
        lambda text, lang: (
            "The plant prepares food using sunlight."
            if text == thin_tamil
            else BACKTRANSLATION
        ),
    )

    client = FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[thin_tamil, GOOD_TAMIL])
    lesson = lesson_generator.generate(CHUNK, None, client=client)

    assert len(client.tamil_calls) == 2
    assert lesson.trace.tamil_regenerated is True
    # The retry must name the concepts that went missing.
    assert "chlorophyll" in client.tamil_calls[1]["user"]
    assert lesson.tamil_explanation == GOOD_TAMIL
    assert "IndicTrans2 indic→en (validation)" in lesson.trace.ai4bharat_models_used


def test_tamil_check_degrades_gracefully_without_indictrans2(monkeypatch):
    monkeypatch.setattr(tamil_quality.translate, "to_english", lambda text, lang: None)

    report = tamil_quality.evaluate(GOOD_TAMIL, CHUNK.concepts, "ta")

    assert report.backtranslation_available is False
    # Tanglish keeps the technical terms in English, so they still match directly.
    assert report.passed


# ------------------------------------------------------------------ matcher unit tests


@pytest.mark.parametrize(
    ("concept", "text", "expected"),
    [
        ("chlorophyll", "the chlorophyll in the leaf", True),
        ("carbon dioxide", "carbon dioxides are absorbed", True),
        ("water", "waters rising from the roots", True),
        ("oxygen", "the plant makes sugar", False),
        ("glucose", "glucose is stored as starch", True),
    ],
)
def test_concept_matcher(concept, text, expected):
    assert grounding_check.concept_present(concept, text) is expected


def test_excerpt_checker_accepts_verbatim_and_rejects_invention():
    assert grounding_check.check_excerpt(GOOD_LESSON["textbook_excerpt"], CHUNK.text)
    assert not grounding_check.check_excerpt("Chlorophyll was isolated in 1817.", CHUNK.text)


def test_generate_accepts_a_full_grounding_result(grounding, fake_client):
    lesson = lesson_generator.generate(grounding, client=fake_client)

    assert lesson.grounded is True
    assert lesson.source_pages == [42, 43]
