"""Query translation.

The contract here is degradation, not correctness: translation runs on the
latency path before every retrieval, and a failure must never turn a valid
question into a refusal.
"""

import pytest
from tests.rag.conftest import fake_response

from app.rag.translate import to_english


@pytest.fixture(autouse=True)
def clear_cache():
    """to_english is lru_cached at module scope, so results leak between tests."""
    to_english.cache_clear()
    yield
    to_english.cache_clear()


class TestHappyPath:
    def test_returns_the_translation(self, stub_client):
        stub_client(fake_response("What is photosynthesis?"))
        assert to_english("ஒளிச்சேர்க்கை என்றால் என்ன?") == "What is photosynthesis?"

    def test_strips_surrounding_whitespace(self, stub_client):
        stub_client(fake_response("  What is photosynthesis?\n"))
        assert to_english("q") == "What is photosynthesis?"


class TestDegradation:
    """Every branch here must return something searchable, never raise."""

    def test_api_failure_falls_back_to_the_original(self, stub_client):
        stub_client(None, raises=RuntimeError("connection reset"))
        assert to_english("What is photosynthesis?") == "What is photosynthesis?"

    def test_missing_api_key_falls_back(self, stub_client):
        """The most likely failure in practice: nobody filled in .env."""
        stub_client(None, raises=Exception("Could not resolve authentication method"))
        assert to_english("What is photosynthesis?") == "What is photosynthesis?"

    def test_empty_response_falls_back(self, stub_client):
        stub_client(fake_response(None))
        assert to_english("original question") == "original question"

    def test_whitespace_response_falls_back(self, stub_client):
        stub_client(fake_response("   "))
        assert to_english("original question") == "original question"

    def test_unexpected_exception_types_are_caught_too(self, stub_client):
        """The handler is deliberately broad — any API-side surprise degrades
        to searching the original text rather than failing the lookup."""

        class SomethingNobodyAnticipated(Exception):
            pass

        stub_client(None, raises=SomethingNobodyAnticipated("?"))
        assert to_english("q") == "q"

    def test_keyboard_interrupt_still_propagates(self, stub_client):
        """`except Exception` is correct here, not a gap: Ctrl-C must not be
        swallowed by a best-effort translation."""
        stub_client(None, raises=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            to_english("q")


class TestCaching:
    def test_repeated_question_hits_the_cache(self, stub_client):
        calls = stub_client(fake_response("What is photosynthesis?"))
        for _ in range(3):
            to_english("ஒளிச்சேர்க்கை என்றால் என்ன?")
        assert len(calls) == 1, "translation is a per-question cost, not per-request"

    def test_different_questions_are_translated_separately(self, stub_client):
        calls = stub_client(fake_response("translated"))
        to_english("question one")
        to_english("question two")
        assert len(calls) == 2

    def test_cache_is_byte_exact(self, stub_client):
        """Trailing whitespace misses the cache. Documented, not desired --
        normalizing the key would make the cache meaningfully more effective."""
        calls = stub_client(fake_response("translated"))
        to_english("same question")
        to_english("same question ")
        assert len(calls) == 2
