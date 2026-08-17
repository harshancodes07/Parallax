"""The grounding regression guard — real embedding model, real scores.

Marked slow: downloads/loads ~1GB on first run. Everything here is what
backend/scripts/calibrate.py measures, frozen as assertions so a change to the
embedder, the chunk size, or the threshold cannot quietly move the boundary.

Run with:  pytest backend/tests/test_grounding.py
Skip with: pytest backend/tests -m "not slow"
"""

import numpy as np
import pytest

from app.rag.chunker import chunk_document
from app.rag.config import PASSAGE_PREFIX, QUERY_PREFIX, SCORE_THRESHOLD

pytestmark = pytest.mark.slow

# Same subject as the chapter, genuinely not answerable from it. The chapter
# covers photosynthesis only -- never respiration, transpiration, stem
# transport, or cell structure.
ADJACENT_TOPIC = [
    "What is respiration in plants?",
    "How do plants transport water through the stem?",
    "What is transpiration?",
    "Describe the structure of a plant cell.",
]


@pytest.fixture(scope="module")
def score(sample_doc):
    """Best cosine similarity of a question against the sample chapter.

    Embeds directly rather than going through Chroma so the test never touches
    the persistent store in ./data.
    """
    from app.rag.store import get_model

    model = get_model()
    chunks = chunk_document(sample_doc)
    passages = model.encode(
        [PASSAGE_PREFIX + c.text for c in chunks], normalize_embeddings=True
    )

    def best(question: str) -> float:
        vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
        return float(np.max(passages @ vec))

    return best


class TestInScope:
    def test_every_in_scope_question_is_admitted(self, score, sample_questions):
        failures = [
            (q, round(score(q), 3))
            for q in sample_questions["in_scope"]
            if score(q) < SCORE_THRESHOLD
        ]
        assert not failures, (
            f"in-scope questions fell below SCORE_THRESHOLD={SCORE_THRESHOLD}: {failures}. "
            "The tutor would tell a student their own chapter isn't in the chapter."
        )


class TestUnrelatedSubjects:
    """What the similarity gate genuinely does well."""

    UNRELATED = [
        "Who was the first prime minister of India?",
        "What is the Pythagoras theorem?",
        "Explain Newton's second law of motion.",
        "What is the capital of France?",
        "How do I solve a quadratic equation?",
        "What is the human digestive system made of?",
    ]

    def test_every_unrelated_subject_is_rejected(self, score):
        leaks = [
            (q, round(score(q), 3)) for q in self.UNRELATED if score(q) >= SCORE_THRESHOLD
        ]
        assert not leaks, f"unrelated subjects passed the gate: {leaks}"

    def test_unrelated_band_clears_the_threshold_by_a_real_margin(self, score):
        best_unrelated = max(score(q) for q in self.UNRELATED)
        assert SCORE_THRESHOLD - best_unrelated >= 0.005, (
            f"best unrelated question scores {best_unrelated:.3f} against a "
            f"threshold of {SCORE_THRESHOLD} — too tight to survive rephrasing"
        )


class TestAdjacentTopicIsNotSeparable:
    """Pins the calibration finding as an executable fact.

    These questions are NOT in the chapter, and the similarity gate admits them
    anyway. That is expected: cosine similarity measures topical relatedness,
    not whether the page answers the question. They are caught downstream by
    the sentinel in guardrail.py.

    If these ever start failing, the embedder got better at answerability and
    the guardrail's effort setting can be revisited. That is a good failure.
    """

    def test_adjacent_topics_score_in_the_in_scope_band(self, score):
        admitted = [q for q in ADJACENT_TOPIC if score(q) >= SCORE_THRESHOLD]
        assert admitted, (
            "adjacent-topic questions are now being rejected by similarity alone — "
            "re-run calibrate.py, the threshold story has changed for the better"
        )

    def test_margin_against_in_scope_is_noise(self, score, sample_questions):
        worst_in = min(score(q) for q in sample_questions["in_scope"])
        best_adjacent = max(score(q) for q in ADJACENT_TOPIC)
        assert worst_in - best_adjacent < 0.02, (
            f"worst in-scope {worst_in:.3f} vs best adjacent {best_adjacent:.3f} — "
            "these now separate; the threshold could do this job alone"
        )


class TestChunkSize:
    def test_smaller_chunks_do_not_help(self, sample_doc, sample_questions):
        """Documents the measured result behind calibrate.py's advice: shrinking
        chunks lets an off-topic question match a fragment, so separation gets
        worse, not better."""
        from app.rag import chunker
        from app.rag.store import get_model

        model = get_model()

        def margin(chunk_words: int, overlap: int) -> float:
            original = (chunker.CHUNK_WORDS, chunker.CHUNK_OVERLAP_WORDS)
            chunker.CHUNK_WORDS, chunker.CHUNK_OVERLAP_WORDS = chunk_words, overlap
            try:
                chunks = chunk_document(sample_doc)
                passages = model.encode(
                    [PASSAGE_PREFIX + c.text for c in chunks], normalize_embeddings=True
                )

                def best(q):
                    v = model.encode(QUERY_PREFIX + q, normalize_embeddings=True)
                    return float(np.max(passages @ v))

                worst_in = min(best(q) for q in sample_questions["in_scope"])
                best_out = max(best(q) for q in sample_questions["out_of_scope"])
                return worst_in - best_out
            finally:
                chunker.CHUNK_WORDS, chunker.CHUNK_OVERLAP_WORDS = original

        assert margin(30, 8) < margin(220, 40), (
            "30-word chunks now separate better than 220 — the guidance in "
            "calibrate.py needs revisiting"
        )
