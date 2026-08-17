"""Find the grounding threshold from real data instead of guessing it.

Usage:
    PYTHONPATH=backend .venv/bin/python backend/scripts/calibrate.py \
        backend/scripts/sample_chapter.json backend/scripts/sample_questions.json

Prints the similarity score of every question, then the widest gap between the
worst in-scope score and the best out-of-scope score. Put the threshold in that
gap and set RAG_SCORE_THRESHOLD in .env.

Run this before the demo. It is also the evidence for the "what stops it from
inventing?" question the judges will ask.
"""

import json
import sys

from app.rag.store import ingest, query


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
    print(f"\nworst in-scope : {worst_in:.3f}")
    print(f"best out-of-scope: {best_out:.3f}")

    if worst_in > best_out:
        print(f"\nclean separation — set RAG_SCORE_THRESHOLD={(worst_in + best_out) / 2:.2f}")
        return 0

    print(
        "\nNO CLEAN SEPARATION. No single threshold separates these questions.\n"
        "Fix the retrieval before tuning the number: try smaller chunks, or a\n"
        "stronger multilingual model (BAAI/bge-m3) via EMBED_MODEL."
    )
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
