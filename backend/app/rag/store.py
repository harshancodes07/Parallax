"""Embedding model + vector store.

One Chroma collection per doc_id, so a question about chapter A can never
retrieve a chunk from chapter B. That isolation is half of the grounding story.
"""

import functools

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .chunker import Chunk, chunk_document
from .config import EMBED_MODEL, PASSAGE_PREFIX, QUERY_PREFIX, VECTOR_STORE_PATH


@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Loaded once per process. Loading per request costs seconds each time."""
    return SentenceTransformer(EMBED_MODEL)


@functools.lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=VECTOR_STORE_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [PASSAGE_PREFIX + t for t in texts]
    vecs = get_model().encode(prefixed, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    vec = get_model().encode(QUERY_PREFIX + text, normalize_embeddings=True)
    return vec.tolist()


def _collection(doc_id: str):
    # Cosine space so distance maps cleanly onto similarity = 1 - distance.
    # Chroma's default is L2, which would make the threshold meaningless.
    return get_client().get_or_create_collection(
        name=f"doc_{doc_id}",
        metadata={"hnsw:space": "cosine"},
    )


def ingest(doc: dict) -> dict:
    """Chunk, embed, and index one uploaded document."""
    chunks: list[Chunk] = chunk_document(doc)
    if not chunks:
        return {"doc_id": doc["doc_id"], "n_chunks": 0}

    collection = _collection(doc["doc_id"])
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed_passages([c.text for c in chunks]),
        metadatas=[{"page": c.page, "doc_id": c.doc_id} for c in chunks],
    )
    return {"doc_id": doc["doc_id"], "n_chunks": len(chunks)}


def query(doc_id: str, question: str, k: int) -> list[dict]:
    """Nearest chunks, best first. Scores are cosine similarity in [0, 1]."""
    collection = _collection(doc_id)
    if collection.count() == 0:
        return []

    res = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": chunk_id,
            "page": meta["page"],
            "text": text,
            "score": 1.0 - distance,
        }
        for chunk_id, text, meta, distance in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
