"""Layer 2: grounded generation and the refusal sentinel.

Calibration showed the similarity gate cannot separate same-subject questions
that are not in the chapter ("respiration" scores 0.844 against a
photosynthesis page; the worst genuine in-scope question scores 0.846). So this
layer is the primary gate for that band, and these tests are the contract for
it. Every path that returns `grounded: True` is a path that can hallucinate.
"""

from tests.rag.conftest import fake_response

from app.rag.config import REFUSAL_SENTINEL
from app.rag.guardrail import answer

IN_SCOPE = {
    "in_scope": True,
    "top_score": 0.86,
    "chunks": [{"chunk_id": "d-p1-c0", "page": 1, "text": "Photosynthesis is...", "score": 0.86}],
}


class TestLayerOnePassthrough:
    def test_below_threshold_never_calls_the_model(self, stub_retrieval, stub_client):
        stub_retrieval({"in_scope": False, "top_score": 0.31, "reason": "below_threshold"})
        calls = stub_client(fake_response("should never be reached"))

        out = answer("d", "What is the capital of France?")

        assert out == {
            "grounded": False,
            "text": None,
            "reason": "below_threshold",
            "top_score": 0.31,
        }
        assert calls == [], "refusing here is what makes layer 1 free"

    def test_empty_index_passes_its_reason_through(self, stub_retrieval, stub_client):
        stub_retrieval({"in_scope": False, "top_score": 0.0, "reason": "empty_index"})
        stub_client(fake_response("x"))
        assert answer("d", "q")["reason"] == "empty_index"


class TestSentinel:
    def test_sentinel_alone_is_a_refusal(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response(REFUSAL_SENTINEL))

        out = answer("d", "What is respiration in plants?")
        assert out["grounded"] is False
        assert out["reason"] == "model_refused"
        assert out["text"] is None

    def test_sentinel_embedded_in_prose_still_refuses(self, stub_retrieval, stub_client):
        """The model sometimes wraps the marker in a sentence. Detection is a
        substring check precisely so that still counts as a refusal."""
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response(f"I'm sorry, {REFUSAL_SENTINEL}."))
        assert answer("d", "q")["grounded"] is False

    def test_refusal_carries_the_score_for_debugging(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response(REFUSAL_SENTINEL))
        assert answer("d", "q")["top_score"] == 0.86


class TestEmptyResponse:
    """Regression: an empty response used to fall through as a grounded answer.

    An empty string does not contain the sentinel, so the substring check
    passed and the function returned grounded=True with text="" — the exact
    failure the module exists to prevent.
    """

    def test_no_text_blocks_is_a_refusal(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response(None))

        out = answer("d", "q")
        assert out["grounded"] is False
        assert out["reason"] == "no_answer"
        assert out["text"] is None

    def test_whitespace_only_is_a_refusal(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response("   \n  "))
        assert answer("d", "q")["grounded"] is False

    def test_safety_refusal_stop_reason_is_a_refusal(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response(None, stop_reason="refusal"))
        assert answer("d", "q")["reason"] == "no_answer"

    def test_truncated_response_with_no_text_is_a_refusal(self, stub_retrieval, stub_client):
        """Thinking can consume max_tokens before any answer is written."""
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response(None, stop_reason="max_tokens"))
        assert answer("d", "q")["grounded"] is False


class TestGroundedAnswer:
    def test_returns_text_and_parses_citations(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response("Plants make food in the leaves [p.1]."))

        out = answer("d", "What is photosynthesis?")
        assert out["grounded"] is True
        assert out["text"] == "Plants make food in the leaves [p.1]."
        assert out["citations"] == [1]
        assert out["top_score"] == 0.86

    def test_citations_are_sorted_and_deduplicated(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response("A [p.3] B [p.1] C [p.3] D [p.2]"))
        assert answer("d", "q")["citations"] == [1, 2, 3]

    def test_multi_digit_pages_parse(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response("See [p.12] and [p.107]."))
        assert answer("d", "q")["citations"] == [12, 107]

    def test_text_is_stripped(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        stub_client(fake_response("  answer [p.1]  \n"))
        assert answer("d", "q")["text"] == "answer [p.1]"


class TestRequestShape:
    def test_context_reaches_the_model_with_page_markers(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        calls = stub_client(fake_response("ok [p.1]"))

        answer("d", "What is photosynthesis?")

        content = calls[0]["messages"][0]["content"]
        assert "[p.1] Photosynthesis is..." in content
        assert "What is photosynthesis?" in content

    def test_system_prompt_defines_the_sentinel(self, stub_retrieval, stub_client):
        stub_retrieval(IN_SCOPE)
        calls = stub_client(fake_response("ok"))
        answer("d", "q")
        assert REFUSAL_SENTINEL in calls[0]["system"]

    def test_runs_at_high_effort(self, stub_retrieval, stub_client):
        """This call is the primary gate for same-subject questions, so it is
        deliberately not run at low effort. Pinned so a cost trim is a
        conscious decision rather than a silent one."""
        stub_retrieval(IN_SCOPE)
        calls = stub_client(fake_response("ok"))
        answer("d", "q")
        assert calls[0]["output_config"]["effort"] == "high"

    def test_student_question_is_sent_verbatim(self, stub_retrieval, stub_client):
        """Only the *search* text is translated; the model must see the
        student's own wording so it can reply in their language."""
        stub_retrieval(IN_SCOPE)
        calls = stub_client(fake_response("ok"))
        tamil = "ஒளிச்சேர்க்கை என்றால் என்ன?"
        answer("d", tamil)
        assert tamil in calls[0]["messages"][0]["content"]


class TestIsAnswerable:
    """The scope check the tutor seam depends on.

    This is the gate that actually rejects a same-subject question the page
    does not answer — similarity cannot, and the tutor treats the verdict as
    final.
    """

    CHUNKS = [{"page": 1, "text": "Photosynthesis happens in the leaves.", "score": 0.84}]

    def test_in_chapter_reply_admits(self, stub_client):
        from app.rag.guardrail import is_answerable

        stub_client(fake_response("IN_CHAPTER"))
        assert is_answerable("What is photosynthesis?", self.CHUNKS) is True

    def test_sentinel_reply_rejects(self, stub_client):
        from app.rag.guardrail import is_answerable

        stub_client(fake_response(REFUSAL_SENTINEL))
        assert is_answerable("What is respiration in plants?", self.CHUNKS) is False

    def test_no_chunks_is_never_answerable(self, stub_client):
        from app.rag.guardrail import is_answerable

        calls = stub_client(fake_response("IN_CHAPTER"))
        assert is_answerable("q", []) is False
        assert calls == [], "nothing to check against, so nothing to spend"

    def test_safety_refusal_rejects(self, stub_client):
        from app.rag.guardrail import is_answerable

        stub_client(fake_response(None, stop_reason="refusal"))
        assert is_answerable("q", self.CHUNKS) is False

    def test_outage_fails_open(self, stub_client):
        """Failing closed would answer 'that isn't in this chapter' to every
        question during an outage — indistinguishable from a working guardrail,
        and far harder to notice than an over-permissive one."""
        from app.rag.guardrail import is_answerable

        stub_client(None, raises=RuntimeError("connection reset"))
        assert is_answerable("What is photosynthesis?", self.CHUNKS) is True

    def test_empty_reply_fails_open(self, stub_client):
        from app.rag.guardrail import is_answerable

        stub_client(fake_response(None))
        assert is_answerable("q", self.CHUNKS) is True

    def test_excerpts_and_question_both_reach_the_model(self, stub_client):
        from app.rag.guardrail import is_answerable

        calls = stub_client(fake_response("IN_CHAPTER"))
        is_answerable("What is photosynthesis?", self.CHUNKS)
        content = calls[0]["messages"][0]["content"]
        assert "[p.1] Photosynthesis happens in the leaves." in content
        assert "What is photosynthesis?" in content
