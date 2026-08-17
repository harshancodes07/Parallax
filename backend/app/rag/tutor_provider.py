"""Real retrieval behind the tutor's `set_grounding_provider` seam.

    from app.rag.tutor_provider import fetch_grounding
    from app.tutor.router import set_grounding_provider
    set_grounding_provider(fetch_grounding)

Same signature as `app.tutor.mock_rag_service.fetch_grounding`, so nothing on
the tutor side moves.

Three things this does that a plain shape-mapping would not:

1. **Owns the whole in-scope decision.** The tutor treats `is_in_scope=False` as
   final and refuses without calling an LLM, so this side has to be right.
   Similarity alone cannot reject a same-subject question the page does not
   answer, so an admitted result is confirmed by `guardrail.is_answerable`
   before it goes back. Disable with `RAG_SCOPE_CHECK=0`.

2. **Does not translate twice.** `query_prep` has already turned the student's
   Tamil into English by the time it reaches us. Running the RAG layer's own
   translation over that would spend a second model call to paraphrase English
   into English, and any drift lands in the search text.

3. **Scopes to a page.** `page_id` may be a bare document id, or `doc#page` for
   one page of it. The tutor teaches "this page", so pulling neighbouring pages
   into the lesson broadens it past what the student is looking at.
"""

from __future__ import annotations

import logging

from app.shared.schemas import GroundingResult, RetrievedChunk

from . import store
from .config import SCOPE_CHECK, TOP_K
from .guardrail import is_answerable
from .retriever import retrieve

log = logging.getLogger(__name__)

PAGE_SEP = "#"


def parse_page_id(page_id: str) -> tuple[str, int | None]:
    """`"a3f9c1"` -> whole document. `"a3f9c1#42"` -> page 42 of it.

    The OCR contract keys an upload by `doc_id` and numbers pages inside it,
    while the tutor addresses "a page". This is the join between the two. A
    non-numeric page part is ignored rather than raising — an id that came from
    a URL should degrade to the whole document, not 500.
    """
    doc_id, sep, page_part = page_id.partition(PAGE_SEP)
    if not sep:
        return page_id, None
    try:
        return doc_id, int(page_part)
    except ValueError:
        log.warning("page_id %r has a non-numeric page part; using the whole doc", page_id)
        return doc_id, None


def _to_chunk(raw: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=raw["chunk_id"],
        page_number=raw["page"],
        text=raw["text"],
        chapter_title=raw.get("chapter_title"),
        concepts=raw.get("concepts") or [],
        similarity_score=raw["score"],
    )


def _refuse(query: str) -> GroundingResult:
    return GroundingResult(query=query, chunks=[], is_in_scope=False)


def fetch_grounding(page_id: str, query: str | None = None) -> GroundingResult:
    """Retrieve grounding for one tutor request.

    `query` is already English — `query_prep` translated it. `None` means
    "teach me this page", which has nothing to rank against, so the page is
    returned whole.
    """
    doc_id, page = parse_page_id(page_id)

    if query is None or not query.strip():
        chunks = store.fetch_page(doc_id, page)
        if not chunks:
            log.info("no content indexed for page_id=%r", page_id)
            return _refuse("")
        return GroundingResult(
            query=chunks[0].get("chapter_title") or "",
            chunks=[_to_chunk(c) for c in chunks],
            is_in_scope=True,
        )

    # Translation is off here on purpose — see the module docstring.
    hit = retrieve(doc_id, query, k=TOP_K, translate=False)

    if not hit["in_scope"]:
        log.info(
            "refused at the similarity gate | page_id=%s score=%.3f reason=%s",
            page_id,
            hit["top_score"],
            hit["reason"],
        )
        return _refuse(query)

    chunks = hit["chunks"]
    if page is not None:
        chunks = [c for c in chunks if c["page"] == page]
        if not chunks:
            log.info(
                "top matches were on other pages of %s; nothing on page %s", doc_id, page
            )
            return _refuse(query)

    if SCOPE_CHECK and not is_answerable(query, chunks):
        log.info(
            "similarity admitted %.3f but the page does not answer it | page_id=%s q=%r",
            hit["top_score"],
            page_id,
            query,
        )
        return _refuse(query)

    result = GroundingResult(
        query=query,
        chunks=[_to_chunk(c) for c in chunks],
        is_in_scope=True,
    )
    if not any(c.concepts for c in result.chunks):
        # The tutor's analogy-coverage and regional-language checks both run off
        # this list; with it empty they pass everything and the grounding score
        # is lost without any visible failure.
        log.warning(
            "no concepts on any chunk of %s — the tutor's analogy and language "
            "guardrails will pass everything. Re-ingest to populate them.",
            page_id,
        )
    return result
