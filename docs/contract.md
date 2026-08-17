# Module contracts

Everyone codes against these shapes. If you need to change one, say so in the group
chat **before** you change it — someone else is already building against it.

## OCR → RAG

`ocr.extract(file) -> IngestedDoc`

```python
{
  "doc_id": "a3f9c1",              # stable id for this upload
  "pages": [
    {"page": 1, "text": "Photosynthesis is the process by which..."},
    {"page": 2, "text": "..."}
  ]
}
```

Rules: `page` is 1-indexed. `text` is plain text, no markup. Empty pages are allowed
but must still appear in the list.

## RAG → tutor / teachback / practice

`rag.retrieve(doc_id, question, k=4) -> Retrieval`

Ask in whatever language the student typed — the question is translated to
English internally before embedding, and `search_text` reports what was
actually searched.

In scope:

```python
{
  "in_scope": True,
  "top_score": 0.82,
  "search_text": "What is photosynthesis?",
  "chunks": [
    {"chunk_id": "a3f9c1-p1-c0", "page": 1, "text": "...", "score": 0.82},
    {"chunk_id": "a3f9c1-p1-c1", "page": 1, "text": "...", "score": 0.71}
  ]
}
```

Out of scope:

```python
{
  "in_scope": False,
  "top_score": 0.31,
  "reason": "below_threshold",     # or "empty_index"
  "search_text": "What is the capital of France?",
  "chunks": []
}
```

`rag.answer(doc_id, question, language) -> Answer`

```python
{"grounded": True,  "text": "Photosynthesis is... [p.1]", "citations": [1], "top_score": 0.82}
{"grounded": False, "text": None, "reason": "below_threshold", "top_score": 0.31}
```

When `grounded` is `False`, the **UI** shows the "not in this chapter" message — the
RAG layer never invents a fallback answer. This is the single most important rule in
the project; grounding is 25% of the score and hallucination is a heavy penalty.

## Ingestion

`rag.ingest(doc) -> {"doc_id": str, "n_chunks": int}`

One vector-store collection per `doc_id`, so an upload can never retrieve another
upload's content.

## The refusal shape (tutor, teachback, practice)

Every module downstream of RAG inherits the refusal. None of them may answer when
`rag.retrieve` says out of scope, and none of them invents a fallback:

```python
{"grounded": False, "reason": "below_threshold", "top_score": 0.31}
```

`reason` is one of `below_threshold`, `empty_index`, `model_refused`, or
`no_answer` (the model returned nothing — safety refusal, or it ran out of
tokens), passed straight through from RAG. On refusal every other key in the success shape is
absent — callers check `grounded` first and never read past it. The UI owns the
"that isn't in this chapter" message; no module below the UI writes that sentence.

`language` is an ISO 639-1 code (`"ta"`, `"hi"`, `"bn"`). The student's language,
not the book's — the page is English, the teaching is not.

## RAG → tutor

`tutor.explain(doc_id, concept, language) -> Explanation`

```python
{
  "grounded": True,
  "language": "ta",
  "text": "ஒளிச்சேர்க்கை என்பது... [p.1]",
  "citations": [1],
  "top_score": 0.86
}
```

Rules: `text` is in `language`, pitched at a 14-year-old, and explains through
analogy rather than translating the page literally — a literal translation scores
nothing. Every factual claim carries a `[p.N]` marker, and `citations` is the
sorted set of pages those markers reference. An analogy is allowed to reach
outside the page; a *fact* is not.

## RAG → teachback

`teachback.evaluate(doc_id, concept, student_answer, language) -> Evaluation`

```python
{
  "grounded": True,
  "verdict": "partial",          # "correct" | "partial" | "incorrect"
  "score": 0.6,                  # 0.0-1.0
  "misconception": "Thinks the plant takes in oxygen rather than releasing it.",
  "followup": "எந்த வாயுவை... ?",
  "citations": [1],
  "top_score": 0.86
}
```

Rules: `misconception` names the *specific* wrong belief, or is `None` when
`verdict` is `"correct"` — "student is confused" is not a misconception and
scores nothing. `followup` is one question in `language` targeting that
misconception. Grade only against the retrieved chunks: if the student says
something true that the page doesn't cover, that is not a mistake.

## RAG → practice

`practice.generate(doc_id, language, difficulty="medium") -> PracticeSet`

```python
{
  "grounded": True,
  "difficulty": "medium",        # "easy" | "medium" | "hard"
  "mcqs": [
    {
      "question": "...",
      "options": ["...", "...", "...", "..."],   # exactly 4
      "answer_index": 2,                         # 0-3
      "rationales": ["why A is wrong", "...", "why C is right", "..."],
      "page": 1
    }
  ],                             # exactly 5
  "short_answers": [
    {"question": "...", "model_answer": "... [p.2]", "page": 2}
  ],                             # exactly 2
  "top_score": 0.86
}
```

Rules: 5 MCQs and 2 short answers, every one answerable from the retrieved chunks
alone. `rationales` has one entry per option, including the correct one —
distractor rationales are explicitly marked and a distractor must be *plausibly*
wrong (a real misconception), not obviously absurd. `difficulty` is chosen by the
caller from teach-back results; the module does not track student state.
