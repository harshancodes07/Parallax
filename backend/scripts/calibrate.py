"""Find the grounding threshold from real data instead of guessing it.

Usage:
    PYTHONPATH=backend .venv/bin/python backend/scripts/calibrate.py \
        backend/scripts/sample_chapter.json backend/scripts/sample_questions.json

Prints the similarity score of every question, then the widest gap between the
worst in-scope score and the best out-of-scope score. Put the threshold in that
gap and set RAG_SCORE_THRESHOLD in .env.

Run this before the demo. It is also the evidence for the "what stops it from
inventing?" question the judges will ask.

Write the questions in ENGLISH. This measures the embedder directly, and
retrieve() translates to English before it embeds, so English questions are what
the threshold actually applies to.
"""

import json
import sys

from app.rag.store import ingest, query

# A threshold is only trustworthy if in-scope and out-of-scope scores separate by
# more than the noise from rephrasing a question. Anything tighter than this is a
# coin flip dressed up as a number.
MIN_MARGIN = 0.02


def main(doc_path: str, questions_path: str) -> int:
    doc = json.loads(open(doc_path).read())
    questions = json.loads(open(questions_path).read())

    print(f"ingesting {doc_path} ...")
    print(ingest(doc), "\n")

    def score(q: str) -> float:
        hits = query(doc["doc_id"], q, k=1)
        return hits[0]["score"] if hits else 0.0

    in_scores = [(q, score(q)) for q in questions["in_scope"]]
    out_scores = [(q, score(q)) for q in questions["out_of_scope"]]

    print("IN SCOPE (should stay above the threshold)")
    for q, s in sorted(in_scores, key=lambda x: x[1]):
        print(f"  {s:.3f}  {q}")

    print("\nOUT OF SCOPE (should stay below the threshold)")
    for q, s in sorted(out_scores, key=lambda x: -x[1]):
        print(f"  {s:.3f}  {q}")

    worst_in = min(s for _, s in in_scores)
    best_out = max(s for _, s in out_scores)
    margin = worst_in - best_out
    print(f"\nworst in-scope : {worst_in:.3f}")
    print(f"best out-of-scope: {best_out:.3f}")
    print(f"margin           : {margin:+.3f}")

    if margin >= MIN_MARGIN:
        print(f"\nclean separation — set RAG_SCORE_THRESHOLD={(worst_in + best_out) / 2:.2f}")
        return 0

    if margin > 0:
        print(
            f"\nSEPARATION TOO NARROW ({margin:.3f} < {MIN_MARGIN}). A threshold exists but\n"
            f"only just: set RAG_SCORE_THRESHOLD={(worst_in + best_out) / 2:.2f} and expect it to\n"
            "misclassify on rephrasing. The similarity gate is NOT doing this job on\n"
            "its own — the refusal sentinel in guardrail.py is load-bearing, not a\n"
            "backstop. Treat it accordingly."
        )
    else:
        print(
            "\nNO CLEAN SEPARATION. No single threshold separates these questions."
        )

    print(
        "\nWhich fix depends on WHICH questions are leaking through:\n"
        "  - unrelated subjects (history, algebra) scoring high => retrieval is weak.\n"
        "    Try a stronger multilingual model (BAAI/bge-m3) via EMBED_MODEL.\n"
        "  - same-subject, adjacent-topic questions scoring high => expected, and no\n"
        "    threshold fixes it. Cosine similarity measures topical relatedness, not\n"
        "    answerability, and 'respiration' really is close to a photosynthesis page.\n"
        "    Smaller chunks make this WORSE, not better (measured): a short chunk lets\n"
        "    an off-topic question match a fragment. Set the threshold just above the\n"
        "    unrelated-subject band and let the guardrail sentinel judge the rest."
    )
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
