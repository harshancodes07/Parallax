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

In scope:

```python
{
  "in_scope": True,
  "top_score": 0.82,
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
