# Bodhi — AI Textbook Tutor in Your Mother Tongue

PS-S01. Upload any page of your own textbook (photo/PDF/text) → the tutor teaches it in one regional
language done excellently → then verifies understanding through a teach-back loop and adaptive practice.

## Core rule of this project

**Everything the tutor says must be grounded in the uploaded page.** If a question falls outside the
ingested content, the answer is "that isn't in this chapter" — never an invented one. Grounding is 25%
of the score and hallucination is a heavy penalty.

## Repo layout

```
backend/
  app/
    ocr/         # photo/PDF -> text (Tesseract / PaddleOCR / vision-model OCR)
    rag/         # chunking, embeddings, FAISS or Chroma store, retrieval + out-of-scope detection
    tutor/       # regional-language explanation generation (analogy-rich, not literal translation)
    teachback/   # scores free-text student answers, pinpoints the specific misconception
    practice/    # 5 MCQs + 2 short answers from the page, adaptive difficulty, distractor rationales
frontend/        # web app — must be usable by a 14-year-old
docs/            # architecture notes, prompt designs, demo script
```

## Feature ownership (one owner, one branch, one PR)

| Area | Branch prefix | Owner |
| --- | --- | --- |
| OCR ingestion | `feat/ocr-*` | _TBD_ |
| RAG + grounding guardrail | `feat/rag-*` | _TBD_ |
| Regional-language teaching | `feat/tutor-*` | _TBD_ |
| Teach-back evaluation | `feat/teachback-*` | _TBD_ |
| Practice generation | `feat/practice-*` | _TBD_ |
| Frontend / UX | `feat/ui-*` | _TBD_ |

Fill in the owners before you start — it's what keeps two people out of the same file.

## Getting started

```bash
git clone <repo-url>
cd Parallax
cp .env.example .env      # add your own API keys; .env is gitignored and never committed
```

Backend:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

Frontend setup goes here once the UI PR lands.

## Tests

```bash
pytest -m "not slow"   # fast: no model download, no API key
pytest                 # adds the grounding + integration tests (~1GB model on first run)
```

The `slow` tests load the real embedding model and assert on real similarity
scores — they are the regression guard for the grounding threshold. The rest
stub the store and the model, so they need no API key.

## Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: never commit to `main`, branch per feature, open a PR,
one teammate reviews, squash merge.
