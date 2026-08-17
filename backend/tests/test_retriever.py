"""Layer 1: the similarity gate.

The store and the translator are stubbed so these pin the *gate logic* exactly
— real scores are covered in test_grounding.py.
"""

import pytest

from app.rag import retriever
from app.rag.retriever import format_context, retrieve


@pytest.fixture
def store(monkeypatch):
    """Control what the vector store returns, and skip translation."""
    monkeypatch.setattr(retriever, "to_english", lambda q: q)
    monkeypatch.setattr(retriever, "TRANSLATE_QUERIES", False)

    def install(*scores):
        chunks = [
            {"chunk_id": f"d-p1-c{i}", "page": 1, "text": f"chunk {i}", "score": s}
            for i, s in enumerate(scores)
        ]
        monkeypatch.setattr(retriever, "query", lambda *a, **k: chunks)
        return chunks

    return install


@pytest.fixture
def threshold(monkeypatch):
    def install(value):
        monkeypatch.setattr(retriever, "SCORE_THRESHOLD", value)

    return install


class TestRefusal:
    def test_empty_index_refuses(self, store):
        store()  # no chunks at all
        out = retrieve("d", "anything")
        assert out["in_scope"] is False
        assert out["reason"] == "empty_index"
        assert out["top_score"] == 0.0
        assert out["chunks"] == []

    def test_below_threshold_refuses(self, store, threshold):
        threshold(0.81)
        store(0.42)
        out = retrieve("d", "capital of France")
        assert out["in_scope"] is False
        assert out["reason"] == "below_threshold"
        assert out["top_score"] == 0.42

    def test_refusal_leaks_no_chunks(self, store, threshold):
        """The whole point: a refusal must hand back nothing to answer from."""
        threshold(0.81)
        store(0.80, 0.79, 0.70)
        assert retrieve("d", "q")["chunks"] == []

    def test_just_below_threshold_still_refuses(self, store, threshold):
        threshold(0.81)
        store(0.8099)
        assert retrieve("d", "q")["in_scope"] is False


class TestAdmission:
    def test_above_threshold_admits_with_chunks(self, store, threshold):
        threshold(0.81)
        chunks = store(0.91, 0.85)
        out = retrieve("d", "photosynthesis")
        assert out["in_scope"] is True
        assert out["top_score"] == 0.91
        assert out["chunks"] == chunks
        assert "reason" not in out, "an in-scope result carries no refusal reason"

    def test_threshold_is_inclusive(self, store, threshold):
        """`score < THRESHOLD` refuses, so a score equal to it is admitted.
        Pinned because flipping this silently shifts every borderline question."""
        threshold(0.81)
        store(0.81)
        assert retrieve("d", "q")["in_scope"] is True

    def test_top_score_comes_from_the_best_chunk(self, store, threshold):
        threshold(0.5)
        store(0.95, 0.60, 0.55)
        assert retrieve("d", "q")["top_score"] == 0.95


class TestSearchText:
    def test_reports_what_was_actually_embedded(self, monkeypatch, store, threshold):
        threshold(0.5)
        store(0.9)
        monkeypatch.setattr(retriever, "TRANSLATE_QUERIES", True)
        monkeypatch.setattr(retriever, "to_english", lambda q: "What is photosynthesis?")

        out = retrieve("d", "ஒளிச்சேர்க்கை என்றால் என்ன?")
        assert out["search_text"] == "What is photosynthesis?"

    def test_untranslated_when_disabled(self, store, threshold):
        threshold(0.5)
        store(0.9)
        assert retrieve("d", "original wording")["search_text"] == "original wording"

    def test_present_on_refusals_too(self, store):
        """Debugging a wrong refusal means seeing what was searched."""
        store()
        assert "search_text" in retrieve("d", "q")


class TestFormatContext:
    def test_tags_every_chunk_with_its_page(self):
        chunks = [
            {"page": 1, "text": "alpha"},
            {"page": 4, "text": "beta"},
        ]
        out = format_context(chunks)
        assert "[p.1] alpha" in out
        assert "[p.4] beta" in out

    def test_empty_chunks_produce_empty_context(self):
        assert format_context([]) == ""

    def test_pages_are_what_the_model_cites_from(self):
        """The model can only cite [p.N] for an N it was shown."""
        out = format_context([{"page": 7, "text": "x"}])
        assert "[p.7]" in out and "[p.1]" not in out
