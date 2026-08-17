"""Chunking. Pure logic — no model, no store, no network."""

import pytest

from app.rag import chunker
from app.rag.chunker import chunk_document, chunk_page


@pytest.fixture
def small_chunks(monkeypatch):
    """10-word chunks with 3 words of overlap, so cases fit on screen."""
    monkeypatch.setattr(chunker, "CHUNK_WORDS", 10)
    monkeypatch.setattr(chunker, "CHUNK_OVERLAP_WORDS", 3)


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


class TestEmptyInput:
    def test_empty_string_yields_nothing(self):
        assert chunk_page("d", 1, "") == []

    def test_whitespace_only_yields_nothing(self):
        assert chunk_page("d", 1, "   \n\t  ") == []

    def test_empty_page_contributes_nothing_but_does_not_raise(self):
        # The OCR contract allows empty pages; they must not break ingestion.
        doc = {"doc_id": "d", "pages": [{"page": 1, "text": ""}, {"page": 2, "text": "hello"}]}
        chunks = chunk_document(doc)
        assert [c.page for c in chunks] == [2]

    def test_missing_text_key_defaults_to_empty(self):
        doc = {"doc_id": "d", "pages": [{"page": 1}]}
        assert chunk_document(doc) == []


class TestSinglePage:
    def test_text_shorter_than_a_chunk_is_one_chunk(self, small_chunks):
        chunks = chunk_page("d", 1, words(4))
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "d-p1-c0"
        assert chunks[0].text == words(4)

    def test_text_exactly_one_chunk_long_is_one_chunk(self, small_chunks):
        chunks = chunk_page("d", 1, words(10))
        assert len(chunks) == 1, "a full-width window must not spawn an empty successor"

    def test_page_number_rides_on_every_chunk(self, small_chunks):
        chunks = chunk_page("d", 7, words(40))
        assert len(chunks) > 1
        assert {c.page for c in chunks} == {7}

    def test_chunk_ids_are_unique_and_sequential(self, small_chunks):
        chunks = chunk_page("abc", 2, words(40))
        ids = [c.chunk_id for c in chunks]
        assert ids == [f"abc-p2-c{i}" for i in range(len(chunks))]
        assert len(set(ids)) == len(ids)

    def test_consecutive_chunks_overlap_by_the_configured_amount(self, small_chunks):
        chunks = chunk_page("d", 1, words(30))
        assert len(chunks) >= 2
        for a, b in zip(chunks, chunks[1:]):
            tail = a.text.split()[-3:]
            head = b.text.split()[:3]
            assert tail == head, "overlap keeps a concept from being split at a boundary"

    def test_no_text_is_dropped_between_chunks(self, small_chunks):
        # Chunk i starts OVERLAP words before the previous chunk ended, so
        # dropping OVERLAP words from every chunk after the first must
        # reconstruct the page exactly. Anything missing is text the retriever
        # could never surface.
        original = words(37)
        chunks = chunk_page("d", 1, original)
        assert len(chunks) > 1

        rebuilt = chunks[0].text.split()
        for c in chunks[1:]:
            rebuilt += c.text.split()[3:]
        assert rebuilt == original.split()

    def test_final_chunk_is_not_a_near_duplicate_tail(self, small_chunks):
        # 11 words with step 7: without the early break the loop emits a second
        # window of one word, a near-duplicate of the first chunk's tail.
        chunks = chunk_page("d", 1, words(11))
        assert len(chunks) == 2
        assert len(chunks[-1].text.split()) > 1


class TestMultiPage:
    def test_pages_are_chunked_independently(self, small_chunks):
        doc = {
            "doc_id": "x",
            "pages": [{"page": 1, "text": words(25)}, {"page": 2, "text": words(25)}],
        }
        chunks = chunk_document(doc)
        p1 = [c for c in chunks if c.page == 1]
        p2 = [c for c in chunks if c.page == 2]
        assert p1 and p2
        # Counters restart per page, so ids stay unique via the page component.
        assert p1[0].chunk_id == "x-p1-c0"
        assert p2[0].chunk_id == "x-p2-c0"
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_doc_id_is_carried_onto_every_chunk(self, small_chunks):
        doc = {"doc_id": "zz", "pages": [{"page": 1, "text": words(25)}]}
        assert {c.doc_id for c in chunk_document(doc)} == {"zz"}

    def test_real_sample_chapter_chunks(self, sample_doc):
        chunks = chunk_document(sample_doc)
        assert len(chunks) == 2, "sample pages are short enough to be one chunk each"
        assert [c.page for c in chunks] == [1, 2]


class TestConfigGuard:
    def test_overlap_not_smaller_than_chunk_size_is_rejected(self, monkeypatch):
        """Overlap >= chunk size makes step <= 0.

        A negative step silently yields ZERO chunks, which means the document
        indexes empty and every question about it is reported out of scope --
        a misconfiguration that looks exactly like a working refusal.
        """
        monkeypatch.setattr(chunker, "CHUNK_WORDS", 10)
        monkeypatch.setattr(chunker, "CHUNK_OVERLAP_WORDS", 12)
        with pytest.raises((ValueError, AssertionError)):
            chunk_page("d", 1, words(50))
