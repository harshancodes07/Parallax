"""RAG → tutor, end to end, with nothing faked but the LLM.

Real chunking, real embeddings, a real Chroma collection, the real adapter and
the tutor's real lesson generator. The only stand-in is the language model,
because there is no API key in CI.

This is the test that would have caught every integration bug worth catching:
a renamed field, a page number that does not survive the hop, an excerpt the
tutor cannot verify against the page, or a refusal that still costs a call.
"""

import pytest

from app.rag import store, tutor_provider
from app.rag.tutor_provider import fetch_grounding
from app.tutor import grounding_check, lesson_generator
from app.tutor.schemas import LessonSource

pytestmark = pytest.mark.slow

CHAPTER = {
    "doc_id": "itest",
    "chapter_title": "Chapter 4: Nutrition in Plants",
    "pages": [
        {
            "page": 42,
            "text": (
                "Photosynthesis is the process by which green plants make their own "
                "food. It takes place mainly in the leaves, inside tiny structures "
                "called chloroplasts. Chloroplasts contain a green pigment called "
                "chlorophyll, which absorbs sunlight. The plant takes in carbon "
                "dioxide from the air through small pores on the leaf called stomata, "
                "and absorbs water from the soil through its roots. Using the energy "
                "of sunlight, the plant converts carbon dioxide and water into "
                "glucose, and releases oxygen as a by-product."
            ),
        },
        {
            "page": 43,
            "text": (
                "The overall word equation for photosynthesis is: carbon dioxide plus "
                "water, in the presence of sunlight and chlorophyll, produces glucose "
                "plus oxygen. Without chlorophyll a plant cannot trap sunlight, which "
                "is why plants kept in the dark turn pale and eventually die."
            ),
        },
    ],
}


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    """One real ingest into a throwaway store, shared by the whole module."""
    path = tmp_path_factory.mktemp("itest-store")
    original = store.VECTOR_STORE_PATH
    store.VECTOR_STORE_PATH = str(path)
    store.get_client.cache_clear()
    try:
        result = store.ingest(CHAPTER)
        assert result["n_chunks"] > 0
        yield result
    finally:
        store.VECTOR_STORE_PATH = original
        store.get_client.cache_clear()


@pytest.fixture(autouse=True)
def no_scope_check(monkeypatch):
    """The scope check needs an API key. Individual tests opt back in."""
    monkeypatch.setattr(tutor_provider, "SCOPE_CHECK", False)


class TestContractSatisfied:
    """Every field the tutor reads, populated by real retrieval."""

    def test_in_scope_question_returns_usable_grounding(self, indexed):
        result = fetch_grounding("itest", "What is photosynthesis?")

        assert result.is_in_scope is True
        assert result.chunks

        chunk = result.chunks[0]
        assert chunk.page_number in (42, 43)
        assert 0.0 < chunk.similarity_score <= 1.0, "must be real, not a constant"
        assert chunk.text.strip()
        assert chunk.chunk_id

    def test_concepts_are_populated(self, indexed):
        """The field the tutor flagged as most likely to be under-delivered.
        Empty here means its analogy and Tamil guardrails pass everything."""
        result = fetch_grounding("itest", "What is photosynthesis?")
        assert any(c.concepts for c in result.chunks)

    def test_chapter_title_survives_ingest(self, indexed):
        result = fetch_grounding("itest", "What is photosynthesis?")
        assert result.chunks[0].chapter_title == "Chapter 4: Nutrition in Plants"

    def test_similarity_scores_actually_discriminate(self, indexed):
        """The tutor sorts by this and teaches from chunks[0]. A constant would
        make the primary chunk arbitrary."""
        close = fetch_grounding("itest", "What is chlorophyll?")
        assert close.is_in_scope
        scores = [c.similarity_score for c in close.chunks]
        assert len(set(scores)) > 1 or len(scores) == 1


class TestTextIsVerbatim:
    """The tutor quotes `text` near-verbatim and then verifies the quote against
    the page. Summarised or cleaned-up text fails its excerpt check."""

    def test_retrieved_text_matches_the_source_page(self, indexed):
        result = fetch_grounding("itest", "What is photosynthesis?")
        pages = {p["page"]: p["text"] for p in CHAPTER["pages"]}

        for chunk in result.chunks:
            assert chunk.text in pages[chunk.page_number], (
                "retrieval altered the page text; the tutor's excerpt check "
                "compares against exactly this"
            )

    def test_tutors_excerpt_check_passes_on_our_text(self, indexed):
        """Run their real validator against our real chunk."""
        result = fetch_grounding("itest", "What is photosynthesis?")
        chunk = result.chunks[0]
        quote = " ".join(chunk.text.split()[:12])
        assert grounding_check.check_excerpt(quote, chunk.text) is True


class TestRefusalPath:
    def test_unrelated_question_refuses(self, indexed):
        result = fetch_grounding("itest", "Who was the first prime minister of India?")
        assert result.is_in_scope is False
        assert result.chunks == []

    def test_refusal_costs_no_llm_call(self, indexed, exploding_client):
        """The whole point of refusing in the RAG layer: the tutor short-circuits
        before generation. ExplodingClient raises on any call."""
        result = fetch_grounding("itest", "What is the capital of France?")
        assert result.is_in_scope is False

        lesson = lesson_generator.generate(result, client=exploding_client)
        assert lesson.grounded is False
        assert lesson.source is LessonSource.REFUSED

    def test_scope_check_rejects_an_adjacent_topic(self, indexed, monkeypatch):
        """Similarity admits 'respiration' against a photosynthesis page. The
        scope check is what turns that into a refusal."""
        monkeypatch.setattr(tutor_provider, "SCOPE_CHECK", True)
        monkeypatch.setattr(tutor_provider, "is_answerable", lambda q, c: False)

        result = fetch_grounding("itest", "What is respiration in plants?")
        assert result.is_in_scope is False

    def test_without_the_scope_check_that_question_gets_through(self, indexed):
        """Documents why the check exists. If this ever starts refusing on
        similarity alone, the embedder improved and the check could come out."""
        result = fetch_grounding("itest", "What is respiration in plants?")
        assert result.is_in_scope is True


class TestTeachThisPage:
    def test_page_scoped_request_returns_only_that_page(self, indexed):
        result = fetch_grounding("itest#42", None)
        assert result.is_in_scope is True
        assert {c.page_number for c in result.chunks} == {42}

    def test_whole_document_request_spans_pages(self, indexed):
        result = fetch_grounding("itest", None)
        assert {c.page_number for c in result.chunks} == {42, 43}

    def test_unknown_document_refuses(self, indexed):
        assert fetch_grounding("no-such-doc", None).is_in_scope is False


class TestLessonGeneration:
    """The full hop: real retrieval into the tutor's real generator.

    Their `fake_client` fixture carries a single canned lesson. Real concepts
    come from the page rather than from that canned text, so the analogy
    coverage check can legitimately ask for one regeneration — these build a
    client with enough responses to let that happen.
    """

    @staticmethod
    def _client():
        from tests.conftest import GOOD_LESSON, GOOD_TAMIL, FakeClient

        return FakeClient(
            json_responses=[GOOD_LESSON] * 3, tamil_responses=[GOOD_TAMIL] * 3
        )

    def test_grounded_lesson_is_produced_from_real_retrieval(self, indexed):
        result = fetch_grounding("itest#42", "What is photosynthesis?")
        assert result.is_in_scope

        lesson = lesson_generator.generate(result, client=self._client())

        assert lesson.grounded is True
        assert lesson.source is not LessonSource.REFUSED
        assert lesson.source_pages == [42], "the '— Page 42' citation in the demo"

    def test_page_number_reaches_the_lesson_unchanged(self, indexed):
        """A wrong number here is a citation that does not match the page the
        student is looking at."""
        result = fetch_grounding("itest#43", "What happens without chlorophyll?")
        if not result.is_in_scope:
            pytest.skip("page 43 did not match this question closely enough")
        lesson = lesson_generator.generate(result, client=self._client())
        assert lesson.source_pages == [43]

    def test_concepts_from_retrieval_reach_the_generator(self, indexed):
        """End of the chain the tutor cares about: page text -> concepts ->
        the list its analogy check runs on."""
        result = fetch_grounding("itest#42", "What is photosynthesis?")
        retrieved = {c for chunk in result.chunks for c in chunk.concepts}
        assert retrieved

        lesson = lesson_generator.generate(result, client=self._client())
        assert lesson.trace is not None, "generate always attaches a trace; the router strips it"
        assert set(lesson.trace.concepts_sent_to_llm) == retrieved
