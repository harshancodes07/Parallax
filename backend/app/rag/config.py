"""Tunable knobs for the RAG layer. Keep every magic number here."""

import os

# Embedding model. Multilingual matters: a student asks in their mother tongue
# about an English textbook page, so the query and the passage are in different
# languages and still have to match.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

# e5 models are trained with these prefixes. Dropping them measurably hurts
# retrieval, so they are applied in store.py / retriever.py, not by the caller.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Chunking, in words (a rough stand-in for tokens — good enough here).
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# Retrieval
TOP_K = 4

# Grounding cutoff: cosine similarity below this means "not in this chapter".
# 0.81 is measured, not guessed — see backend/scripts/calibrate.py. Re-run it
# against a real chapter before the demo and move this if the gap shifts.
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.81"))

# Translate the query to English before embedding. Cross-lingual embedding does
# not separate in-scope from out-of-scope questions well enough to ground on —
# see translate.py for the numbers. Costs one model call per question.
TRANSLATE_QUERIES = os.getenv("TRANSLATE_QUERIES", "1") == "1"
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "claude-opus-5")

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vectorstore")

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-opus-5")

# Machine-checkable refusal marker. Detecting refusal from prose is unreliable;
# a sentinel is not.
REFUSAL_SENTINEL = "NOT_IN_CHAPTER"
