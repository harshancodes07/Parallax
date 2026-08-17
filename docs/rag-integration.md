# RAG → tutor integration

Status: **done and tested.** `app.rag.tutor_provider.fetch_grounding` implements
the seam; `main.py` installs it at startup. This document answers the questions
raised on the tutor side and records what changed as a result.

## 1. The contract holds

`RetrievedChunk` and `GroundingResult` in `backend/app/shared/schemas.py` are
unchanged — no edits were needed. Every field is populated:

| Field | What RAG puts there |
| --- | --- |
| `chunk_id` | `{doc_id}-p{page}-c{n}`, unique per chunk |
| `page_number` | The page the text came from, carried from OCR through chunking |
| `text` | **Verbatim page text.** Never summarised or cleaned — sliced on word boundaries only, so your excerpt check compares against exactly what was indexed |
| `chapter_title` | From `IngestedDoc.chapter_title` when OCR supplies one, else `None` |
| `concepts` | Key terms, extracted at ingest — see below |
| `similarity_score` | Real cosine similarity in [0, 1], never a constant. `1.0` for a `query=None` page fetch, since the student asked for that page by name and there is nothing to rank |

`GroundingResult.query` echoes the question that was searched; on a refusal it
still carries the query so your message can quote it.

## 2. `concepts` is populated

Extracted once per upload and stored on the chunk, so retrieval stays a pure
vector lookup. `CONCEPT_MAX=6` by default, matching the size your mock used.

The LLM path is preferred; a deterministic fallback runs when no API key is
available, so ingestion never produces an empty list just because a key is
missing. The fallback is blunter — it ranks frequency and known multi-word
science terms — and it logs when it is used.

If a result ever does come back with no concepts on any chunk, the adapter logs
a warning naming the page, because the failure is otherwise invisible: both your
analogy-coverage and Tamil checks would pass everything and the grounding score
would go quietly.

## 3. Queries: English confirmed, and we now translate only once

Confirmed — retrieval indexes English and expects English. The embedding model is
still multilingual, but that is not what the grounding decision rests on.

One thing to know: the RAG layer had **its own** query translation, added for the
case where it is called directly rather than through the tutor. Running both
would have spent a second model call turning your already-English string into
English again, with any paraphrase drift landing in the text that gets embedded.
`retrieve()` now takes `translate=False` and the adapter passes it. Nothing to do
on your side.

## 4. `is_in_scope` now costs one model call — here is why

You said RAG owns the entire out-of-scope decision and the tutor never
second-guesses it. Taking that seriously changed the design, because **the
similarity threshold cannot make that decision alone.**

Measured against a real photosynthesis chapter:

| Question | Score | In the chapter? |
| --- | --- | --- |
| What is the role of chlorophyll? | 0.846 | yes — and it is the *worst* in-scope score |
| **What is respiration in plants?** | **0.844** | **no** |
| How do plants transport water through the stem? | 0.837 | no |
| How do I solve a quadratic equation? | 0.798 | no |

A 0.002 margin is noise. Cosine similarity measures topical relatedness, not
whether the page answers the question, and smaller chunks make it measurably
worse rather than better. So the threshold's real job is rejecting *unrelated
subjects*, and a second check asks the model whether the retrieved page actually
answers the question before the adapter reports it in scope.

Consequences for you:

- An in-scope verdict costs one extra model call. It also *saves* a whole lesson
  generation whenever it refuses, so on out-of-scope traffic it is cheaper.
- It **fails open**. If the model is unreachable, the question is admitted and
  your grounding checks carry the load. Failing closed would answer "that isn't
  in this chapter" to every question during an outage — indistinguishable from a
  working guardrail, and far harder to notice.
- `RAG_SCOPE_CHECK=0` turns it off if you want layer 1 only while iterating.

## 5. The `pg42` widening you flagged

Fixed by page scoping. `page_id` accepts either form:

| `page_id` | Behaviour |
| --- | --- |
| `a3f9c1` | The whole upload. Top-k may span pages. |
| `a3f9c1#42` | Page 42 only. Matches on other pages are dropped. |

Use the `#page` form when the student is looking at one page and the lesson
should not widen past it. A non-numeric page part degrades to the whole document
rather than erroring, so an id that came from a URL cannot 500 the request.

`query=None` ("teach me this page") skips retrieval entirely and returns that
page's chunks whole, in page order, at score 1.0.

## 6. Wiring

Already done in `backend/app/main.py`:

```python
from app.rag.tutor_provider import fetch_grounding
from app.tutor.router import set_grounding_provider

set_grounding_provider(fetch_grounding)
```

It is wrapped in a `try` on purpose: importing the RAG stack pulls in
sentence-transformers and Chroma, and someone working on the tutor alone should
get a running app rather than an `ImportError` at startup. The mock stays in
place if it fails, and the chosen provider is logged at INFO.

`mock_rag_service.py` is untouched and still the reference implementation.

## Tests

`pytest` — 219 tests, both suites, green together.

- `tests/rag/test_tutor_provider.py` — the seam: shape mapping, refusal paths,
  page scoping, the scope check, no double translation.
- `tests/rag/test_integration_tutor.py` — real chunking, real embeddings, a real
  Chroma collection, the real adapter and your real `lesson_generator`. Only the
  LLM is faked. It asserts that our `text` passes *your* `check_excerpt`, that
  page numbers survive the hop, and that a refusal reaches
  `LessonSource.REFUSED` without any model call.

Two merge-level problems were fixed on the way, both of which produce a green
run that proves less than it appears to:

- `pytest.ini` collected `tests` only, silently skipping the entire RAG suite.
- `backend/tests/` shadowed your top-level `tests` package via
  `pythonpath = backend`, breaking `from tests.conftest import ...`.

Both resolved by moving the RAG suite to `tests/rag/`. One tests tree now.

## Still open

- **The scope check has never run against a live model.** There is no API key in
  this environment, so its decision logic is tested against scripted replies. Whether
  Opus 5 actually emits the refusal marker for "What is respiration in plants?"
  on a photosynthesis page is the single most valuable thing left to verify, and
  it needs one run with a real key.
- `chapter_title` is `None` until OCR supplies one; your code already handles that.
- The deterministic concept fallback is noticeably blunter than the LLM path
  (it can surface a verb like "absorb"). With a key at ingest time this does not
  apply.
