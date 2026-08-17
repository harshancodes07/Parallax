"""Stub upstream data so this branch runs standalone before `feat/rag-grounding` lands.

Delete this module's *use* at integration time (swap `fetch_grounding` for the
real retriever); keep the module itself — the tests depend on it.
"""

from __future__ import annotations

from app.shared.schemas import GroundingResult, RetrievedChunk

PHOTOSYNTHESIS = RetrievedChunk(
    chunk_id="pg42-c1",
    page_number=42,
    text=(
        "Photosynthesis is the process by which green plants prepare their own food. "
        "The green pigment chlorophyll present in the leaves absorbs sunlight. "
        "Using this energy, the plant combines carbon dioxide from the air with water "
        "drawn up from the roots to form glucose. Oxygen is released into the air as a "
        "by-product. The glucose formed is stored in the plant as starch."
    ),
    chapter_title="Chapter 4: Nutrition in Plants",
    concepts=["chlorophyll", "sunlight", "carbon dioxide", "water", "glucose", "oxygen"],
    similarity_score=0.91,
)

RESPIRATION = RetrievedChunk(
    chunk_id="pg43-c1",
    page_number=43,
    text=(
        "Respiration in plants takes place both day and night. During respiration, "
        "the glucose stored in the plant is broken down using oxygen to release energy. "
        "Carbon dioxide and water vapour are given out. Unlike photosynthesis, "
        "respiration does not need sunlight."
    ),
    chapter_title="Chapter 4: Nutrition in Plants",
    concepts=["glucose", "oxygen", "energy", "carbon dioxide"],
    similarity_score=0.78,
)

WATER_CYCLE = RetrievedChunk(
    chunk_id="pg58-c2",
    page_number=58,
    text=(
        "Water from rivers, lakes and oceans evaporates due to the heat of the sun and "
        "rises into the air as water vapour. High above the ground the vapour cools and "
        "condenses into tiny droplets, forming clouds. When the droplets grow heavy they "
        "fall back to the earth as rain. This is called the water cycle."
    ),
    chapter_title="Chapter 7: Water",
    concepts=["evaporation", "water vapour", "condensation", "clouds", "rain"],
    similarity_score=0.88,
)

_PAGES: dict[str, list[RetrievedChunk]] = {
    "pg42": [PHOTOSYNTHESIS, RESPIRATION],
    "pg43": [RESPIRATION],
    "pg58": [WATER_CYCLE],
}

# Judges *will* try these live. Anything here must produce a refusal, never an answer.
OUT_OF_SCOPE_QUERIES = (
    "who is the prime minister of india",
    "explain quantum entanglement",
    "what is the capital of france",
)


def fetch_grounding(page_id: str, query: str | None = None) -> GroundingResult:
    """Pretend to be the RAG layer.

    Returns `is_in_scope=False` for an unknown page or a question that clearly
    isn't about the ingested content — the case the refusal path must handle.
    """
    chunks = _PAGES.get(page_id, [])
    effective_query = query or (chunks[0].chapter_title if chunks else "") or page_id

    if not chunks:
        return GroundingResult(query=effective_query, chunks=[], is_in_scope=False)

    if query and _looks_out_of_scope(query, chunks):
        return GroundingResult(query=query, chunks=[], is_in_scope=False)

    return GroundingResult(query=effective_query, chunks=chunks, is_in_scope=True)


def _looks_out_of_scope(query: str, chunks: list[RetrievedChunk]) -> bool:
    """Crude stand-in for a similarity threshold. The real check lives in the RAG branch."""
    normalised = query.casefold().strip()
    if normalised in OUT_OF_SCOPE_QUERIES:
        return True

    haystack = " ".join(c.text for c in chunks).casefold()
    words = [w for w in _tokenise(normalised) if len(w) > 3]
    if not words:
        return False
    return not any(w in haystack for w in words)


def _tokenise(text: str) -> list[str]:
    return [w.strip(".,?!'\"") for w in text.split()]
