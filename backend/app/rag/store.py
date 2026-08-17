"""Embedding model + vector store.

One Chroma collection per doc_id, so a question about chapter A can never
retrieve a chunk from chapter B. That isolation is half of the grounding story.
"""

import functools

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .chunker import Chunk, chunk_document
from .concepts import extract
from .config import EMBED_MODEL, PASSAGE_PREFIX, QUERY_PREFIX, VECTOR_STORE_PATH

CONCEPT_SEP = "|"


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
        metadatas=[
            {
                "page": c.page,
                "doc_id": c.doc_id,
                # Chroma metadata values must be scalars, so the term list is
                # stored joined. `|` because a concept can legitimately contain
                # a comma-free space ("carbon dioxide") but never a pipe.
                "concepts": CONCEPT_SEP.join(extract(c.text)),
                "chapter_title": doc.get("chapter_title") or "",
            }
            for c in chunks
        ],
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
            "concepts": _split_concepts(meta.get("concepts")),
            "chapter_title": meta.get("chapter_title") or None,
        }
        for chunk_id, text, meta, distance in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def fetch_page(doc_id: str, page: int | None = None) -> list[dict]:
    """Every chunk of a document, or of one page of it — no question involved.

    The tutor's "teach me this page" request carries no query, so there is
    nothing to rank by similarity against. Scores come back as 1.0: the student
    asked for this page by name, which is as in-scope as a request gets.
    """
    collection = _collection(doc_id)
    if collection.count() == 0:
        return []

    where = {"page": page} if page is not None else None
    res = collection.get(where=where, include=["documents", "metadatas"])

    chunks = [
        {
            "chunk_id": chunk_id,
            "page": meta["page"],
            "text": text,
            "score": 1.0,
            "concepts": _split_concepts(meta.get("concepts")),
            "chapter_title": meta.get("chapter_title") or None,
        }
        for chunk_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    # `get` returns insertion order, which is only page order by luck.
    chunks.sort(key=lambda c: (c["page"], c["chunk_id"]))
    return chunks


def _split_concepts(raw: str | None) -> list[str]:
    """Chunks indexed before concepts existed have no such metadata key, so this
    has to cope with `None` rather than assuming a re-ingest has happened."""
    if not raw:
        return []
    return [term for term in raw.split(CONCEPT_SEP) if term]
