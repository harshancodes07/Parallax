"""The RAG → tutor seam.

The tutor treats `is_in_scope=False` as final and refuses without calling an
LLM, so everything this returns is load-bearing. No model and no vector store
here — both are stubbed so the mapping and the decision logic are pinned
exactly.
"""

import pytest

from app.rag import tutor_provider
from app.rag.tutor_provider import fetch_grounding, parse_page_id


def raw(page=1, score=0.9, concepts=("chlorophyll", "glucose"), chunk_id=None, title="Chapter 4"):
    return {
        "chunk_id": chunk_id or f"d-p{page}-c0",
        "page": page,
        "text": "Photosynthesis is how green plants make food.",
        "score": score,
        "concepts": list(concepts),
        "chapter_title": title,
    }


@pytest.fixture
def rag(monkeypatch):
    """Control what the RAG layer hands the adapter."""

    calls: dict = {"retrieve": [], "answerable": []}

    def install(*, retrieval=None, page_chunks=None, answerable=True):
        def _retrieve(doc_id, query, k=4, translate=None):
            calls["retrieve"].append(
                {"doc_id": doc_id, "query": query, "translate": translate}
            )
            return retrieval

        def _is_answerable(question, chunks):
            calls["answerable"].append(question)
            if isinstance(answerable, Exception):
                raise answerable
            return answerable

        monkeypatch.setattr(tutor_provider, "retrieve", _retrieve)
        monkeypatch.setattr(tutor_provider, "is_answerable", _is_answerable)
        monkeypatch.setattr(
            tutor_provider.store, "fetch_page", lambda d, p=None: page_chunks or []
        )
        monkeypatch.setattr(tutor_provider, "SCOPE_CHECK", True)
        return calls

    return install


def in_scope(*chunks, score=0.9):
    return {"in_scope": True, "top_score": score, "chunks": list(chunks), "search_text": "q"}


def out_of_scope(reason="below_threshold", score=0.31):
    return {"in_scope": False, "top_score": score, "reason": reason, "chunks": []}


class TestPageId:
    def test_bare_id_is_the_whole_document(self):
        assert parse_page_id("a3f9c1") == ("a3f9c1", None)

    def test_hash_selects_a_page(self):
        assert parse_page_id("a3f9c1#42") == ("a3f9c1", 42)

    def test_non_numeric_page_degrades_to_the_document(self):
        """An id that came from a URL should not 500 the request."""
        assert parse_page_id("a3f9c1#cover") == ("a3f9c1", None)

    def test_empty_page_part_degrades_to_the_document(self):
        assert parse_page_id("a3f9c1#") == ("a3f9c1", None)


class TestShapeMapping:
    def test_fields_land_where_the_tutor_expects_them(self, rag):
        rag(retrieval=in_scope(raw(page=42, score=0.88)))

        result = fetch_grounding("d", "What is photosynthesis?")

        assert result.is_in_scope is True
        assert result.query == "What is photosynthesis?"
        chunk = result.chunks[0]
        assert chunk.page_number == 42, "tutor renders this as the '— Page 42' citation"
        assert chunk.similarity_score == 0.88, "tutor sorts by this to pick the primary"
        assert chunk.concepts == ["chlorophyll", "glucose"]
        assert chunk.chapter_title == "Chapter 4"

    def test_text_is_passed_through_verbatim(self, rag):
        """The tutor quotes this near-verbatim and then verifies the quote
        against the page, so any cleanup here fails its excerpt check."""
        original = raw()
        rag(retrieval=in_scope(original))
        assert fetch_grounding("d", "q").chunks[0].text == original["text"]

    def test_missing_concepts_become_an_empty_list_not_none(self, rag):
        rag(retrieval=in_scope({**raw(), "concepts": None}))
        assert fetch_grounding("d", "q").chunks[0].concepts == []

    def test_chunk_order_is_preserved(self, rag):
        rag(retrieval=in_scope(raw(score=0.9, chunk_id="a"), raw(score=0.7, chunk_id="b")))
        assert [c.chunk_id for c in fetch_grounding("d", "q").chunks] == ["a", "b"]


class TestRefusal:
    def test_below_threshold_refuses(self, rag):
        rag(retrieval=out_of_scope())
        result = fetch_grounding("d", "What is the capital of France?")
        assert result.is_in_scope is False
        assert result.chunks == []

    def test_empty_index_refuses(self, rag):
        rag(retrieval=out_of_scope(reason="empty_index", score=0.0))
        assert fetch_grounding("d", "q").is_in_scope is False

    def test_refusal_never_runs_the_scope_check(self, rag):
        """Layer 1 refusing is what makes it free — don't spend a call after it."""
        calls = rag(retrieval=out_of_scope())
        fetch_grounding("d", "q")
        assert calls["answerable"] == []

    def test_refusal_keeps_the_query_for_the_tutors_message(self, rag):
        rag(retrieval=out_of_scope())
        assert fetch_grounding("d", "why is the sky blue?").query == "why is the sky blue?"


class TestScopeCheck:
    """The reason this adapter exists rather than a plain field rename.

    Similarity admits same-subject questions the page does not answer
    ("respiration" scores 0.844 against a photosynthesis page; the worst
    genuine in-scope question scores 0.846).
    """

    def test_admitted_by_similarity_but_unanswerable_is_refused(self, rag):
        rag(retrieval=in_scope(raw(score=0.844)), answerable=False)
        result = fetch_grounding("d", "What is respiration in plants?")
        assert result.is_in_scope is False
        assert result.chunks == []

    def test_answerable_question_passes_through(self, rag):
        rag(retrieval=in_scope(raw(score=0.909)), answerable=True)
        assert fetch_grounding("d", "What is photosynthesis?").is_in_scope is True

    def test_scope_check_sees_the_filtered_chunks(self, rag):
        calls = rag(retrieval=in_scope(raw(score=0.9)), answerable=True)
        fetch_grounding("d", "What is photosynthesis?")
        assert calls["answerable"] == ["What is photosynthesis?"]

    def test_can_be_switched_off(self, rag, monkeypatch):
        calls = rag(retrieval=in_scope(raw()), answerable=False)
        monkeypatch.setattr(tutor_provider, "SCOPE_CHECK", False)
        assert fetch_grounding("d", "q").is_in_scope is True
        assert calls["answerable"] == [], "disabled means not called at all"


class TestNoDoubleTranslation:
    def test_retrieval_is_told_the_query_is_already_english(self, rag):
        """query_prep already translated. Doing it again spends a second model
        call to paraphrase English into English, and the drift lands in the
        text that gets embedded."""
        calls = rag(retrieval=in_scope(raw()))
        fetch_grounding("d", "how does a plant eat")
        assert calls["retrieve"][0]["translate"] is False


class TestTeachThisPage:
    """`query=None` — "teach me this page". Nothing to rank against."""

    def test_returns_the_whole_page_in_scope(self, rag):
        rag(page_chunks=[raw(page=42), raw(page=42, chunk_id="d-p42-c1")])
        result = fetch_grounding("d#42", None)
        assert result.is_in_scope is True
        assert len(result.chunks) == 2

    def test_scores_are_full_confidence(self, rag):
        """The student asked for this page by name — there is no similarity
        judgement to make, and a low score would mislead the tutor's sort."""
        rag(page_chunks=[raw(page=42, score=1.0)])
        assert fetch_grounding("d#42", None).chunks[0].similarity_score == 1.0

    def test_never_runs_retrieval_or_the_scope_check(self, rag):
        calls = rag(page_chunks=[raw()])
        fetch_grounding("d", None)
        assert calls["retrieve"] == []
        assert calls["answerable"] == []

    def test_unknown_page_refuses(self, rag):
        rag(page_chunks=[])
        assert fetch_grounding("d#999", None).is_in_scope is False

    def test_blank_query_is_treated_as_no_query(self, rag):
        rag(page_chunks=[raw()])
        assert fetch_grounding("d", "   ").is_in_scope is True

    def test_query_falls_back_to_the_chapter_title(self, rag):
        rag(page_chunks=[raw(title="Chapter 4: Nutrition in Plants")])
        assert fetch_grounding("d", None).query == "Chapter 4: Nutrition in Plants"


class TestPageScoping:
    def test_chunks_from_other_pages_are_dropped(self, rag):
        """The tutor teaches one page. Neighbouring pages broaden the lesson
        past what the student is actually looking at."""
        rag(retrieval=in_scope(raw(page=42), raw(page=43, chunk_id="d-p43-c0")))
        result = fetch_grounding("d#42", "What is photosynthesis?")
        assert [c.page_number for c in result.chunks] == [42]

    def test_no_page_filter_keeps_everything(self, rag):
        rag(retrieval=in_scope(raw(page=42), raw(page=43, chunk_id="d-p43-c0")))
        assert len(fetch_grounding("d", "q").chunks) == 2

    def test_refuses_when_the_page_has_none_of_the_matches(self, rag):
        rag(retrieval=in_scope(raw(page=43)))
        assert fetch_grounding("d#42", "q").is_in_scope is False


class TestConceptWarning:
    def test_warns_when_no_chunk_carries_concepts(self, rag, caplog):
        """Empty concepts disable the tutor's analogy and language guardrails
        without failing anything — it has to be loud here or nowhere."""
        rag(retrieval=in_scope({**raw(), "concepts": []}))
        with caplog.at_level("WARNING"):
            result = fetch_grounding("d", "q")
        assert result.is_in_scope is True, "still usable, just unguarded"
        assert any("concept" in r.message.lower() for r in caplog.records)

    def test_silent_when_concepts_are_present(self, rag, caplog):
        rag(retrieval=in_scope(raw()))
        with caplog.at_level("WARNING"):
            fetch_grounding("d", "q")
        assert not [r for r in caplog.records if "concept" in r.message.lower()]
