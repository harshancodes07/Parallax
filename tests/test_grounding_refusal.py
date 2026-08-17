"""Merge gate #1: out-of-scope input must refuse WITHOUT calling the LLM.

Grounding is 25% of the score and hallucination is the heavy penalty, so this
file asserts the absence of a call, not just the shape of the answer.
"""

from __future__ import annotations

import pytest

from app.shared.schemas import GroundingResult, RetrievedChunk
from app.tutor import lesson_generator, mock_rag_service
from app.tutor.schemas import LessonSource

UNRELATED_CHUNK = RetrievedChunk(
    chunk_id="pg99-c1",
    page_number=99,
    text="The Indus Valley Civilisation had well planned cities with covered drains.",
    chapter_title="Chapter 9: Early Cities",
    concepts=["cities", "drains"],
    similarity_score=0.11,
)


def test_out_of_scope_refuses_and_never_calls_the_llm(exploding_client):
    grounding = GroundingResult(
        query="explain quantum entanglement", chunks=[UNRELATED_CHUNK], is_in_scope=False
    )

    # ExplodingClient raises AssertionError on any call — reaching the end proves
    # the hard `if` short-circuited before generation.
    lesson = lesson_generator.generate(grounding, client=exploding_client)

    assert lesson.grounded is False
    assert lesson.source is LessonSource.REFUSED
    assert lesson.refusal_reason == lesson_generator.REFUSAL_TEXT
    assert lesson.tamil_explanation == lesson_generator.REFUSAL_TAMIL
    assert lesson.simple_explanation == ""
    assert lesson.analogy == ""
    assert lesson.textbook_excerpt == ""


def test_empty_chunks_refuses_even_if_flagged_in_scope(exploding_client):
    """Defence in depth: `is_in_scope=True` with no chunks is still nothing to teach from."""
    grounding = GroundingResult(query="anything", chunks=[], is_in_scope=True)

    lesson = lesson_generator.generate(grounding, client=exploding_client)

    assert lesson.grounded is False
    assert lesson.source is LessonSource.REFUSED


def test_refusal_records_that_no_call_was_made(exploding_client):
    grounding = GroundingResult(query="who is the prime minister", chunks=[], is_in_scope=False)

    lesson = lesson_generator.generate(grounding, client=exploding_client)

    assert lesson.trace is not None
    assert "no LLM call" in " ".join(lesson.trace.notes)


@pytest.mark.parametrize("query", mock_rag_service.OUT_OF_SCOPE_QUERIES)
def test_mock_rag_marks_known_out_of_scope_questions(query):
    grounding = mock_rag_service.fetch_grounding("pg42", query)

    assert grounding.is_in_scope is False
    assert grounding.chunks == []


def test_mock_rag_marks_unknown_page_out_of_scope():
    grounding = mock_rag_service.fetch_grounding("pg-does-not-exist", "photosynthesis")

    assert grounding.is_in_scope is False


def test_mock_rag_returns_the_photosynthesis_page_in_scope():
    grounding = mock_rag_service.fetch_grounding("pg42", "how do plants make food")

    assert grounding.is_in_scope is True
    assert grounding.chunks[0].page_number == 42


def test_api_refuses_out_of_scope_question(monkeypatch):
    """Route-level check: the refusal reaches the frontend in the right shape."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from app.tutor import router as tutor_router

    monkeypatch.setattr(
        lesson_generator,
        "_default_client",
        _Exploder(),
        raising=True,
    )

    from app.main import app

    client = fastapi_testclient.TestClient(app)
    response = client.post(
        "/api/tutor/explain", json={"page_id": "pg42", "query": "what is the capital of france"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["refusal_reason"]
    assert body["trace"] is None  # trace only on ?debug=true
    assert tutor_router.router.prefix == "/api/tutor"


class _Exploder:
    def complete_json(self, **kwargs):  # noqa: ANN003
        raise AssertionError("LLM called on an out-of-scope request")

    def complete_text(self, **kwargs):  # noqa: ANN003
        raise AssertionError("LLM called on an out-of-scope request")
