"""Grounded retrieval over one uploaded textbook chapter.

Public surface (see docs/contract.md):
    ingest(doc)                     -> {"doc_id", "n_chunks"}
    retrieve(doc_id, question, k=4) -> Retrieval
    answer(doc_id, question)        -> Answer
"""

from .guardrail import answer
from .retriever import retrieve
from .store import ingest

__all__ = ["ingest", "retrieve", "answer"]
