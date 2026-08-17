# `feat/tutor-explanation` — regional-language teaching

**Scope:** given an already-retrieved, already-grounded textbook chunk, produce a
structured lesson. This branch does no OCR, no embeddings, no vector search, no
teach-back scoring, no MCQ generation. It *consumes* a `GroundingResult` and
*produces* a `TutorLesson`.

## Layout

```
backend/app/shared/schemas.py       # SHARED contract — coordinate with feat/rag-grounding
backend/app/tutor/
  schemas.py            # TutorLesson, ConceptMapping, LessonTrace
  mock_rag_service.py   # 3 fake chunks + an out-of-scope case, for standalone dev
  prompts.py            # all prompt text, kept separate for iteration
  llm_client.py         # thin Claude wrapper + the structured-output schema
  grounding_check.py    # claim self-check, analogy coverage, excerpt verification
  tamil_quality.py      # IndicTrans2 backtranslation check
  lesson_generator.py   # orchestration — the ONE public function
  router.py             # HTTP surface
  indic/                # AI4Bharat model layer
    languages.py        #   5 languages, every code each model wants
    runtime.py          #   lazy loading, kill switches, graceful failure
    translate.py        #   IndicTrans2 ×3 directions
    transliterate.py    #   IndicXlit
    asr.py              #   IndicConformer
    tts.py              #   Indic Parler-TTS
tests/
  test_lesson_generator.py
  test_grounding_refusal.py
  test_indic_models.py
```

> The prompt spec said `backend/tutor_explanation/`; the repo README and the
> pre-existing `.gitkeep` dirs say `backend/app/tutor/`. Went with the repo so the
> merge is clean. Module names and the single-entry-point rule are unchanged.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; `source .venv/bin/activate` elsewhere
pip install -r backend/requirements.txt
cp .env.example .env                                 # add GEMINI_API_KEY (free) or ANTHROPIC_API_KEY
uvicorn app.main:app --reload --app-dir backend
pytest                                               # 62 tests, no network, no API key needed
```

```bash
curl -X POST "http://127.0.0.1:8000/api/tutor/explain?debug=true" \
  -H "content-type: application/json" \
  -d '{"page_id": "pg42", "query": "how do plants make their food"}'
```

`?debug=true` adds the `trace` object — source text sent to the model, which checks
ran, the backtranslation, and which AI4Bharat models actually ran. That is the
demo's "generated from Page 42" panel.

## Endpoints

| Endpoint | Does | Model |
|---|---|---|
| `POST /api/tutor/explain` | Lesson from a page | Claude + IndicTrans2 |
| `POST /api/tutor/explain/preview` | Same, rendered in the demo format | — |
| `POST /api/tutor/listen` | Audio → transcript | IndicConformer |
| `POST /api/tutor/ask` | Audio → full lesson, one round trip | IndicConformer + Claude |
| `POST /api/tutor/speak` | Text → WAV | Indic Parler-TTS |
| `POST /api/tutor/explain/speak` | Page → spoken lesson | Claude + Parler-TTS |
| `POST /api/tutor/transliterate` | Tanglish ↔ native script | IndicXlit |
| `GET /api/tutor/languages` | The five languages | — |
| `GET /api/tutor/capabilities` | **Which models actually loaded** | — |

`/capabilities` is worth having on screen during judging: it distinguishes "wired
up" from "running", which is the honest answer when someone asks what is real.

## AI4Bharat models

All five requested models are wired. Every one is lazily loaded, individually
kill-switchable, and returns `None` rather than raising — the tutor works with
none of them installed.

| Model | Verified id | Role here |
|---|---|---|
| IndicTrans2 Indic→En 200M | `ai4bharat/indictrans2-indic-en-dist-200M` | **Validation.** Backtranslate the composed explanation, check concepts survived. |
| IndicTrans2 En→Indic 200M | `ai4bharat/indictrans2-en-indic-dist-200M` | **Fallback only.** Used when composition is unavailable, and for languages we can't reach otherwise. |
| IndicTrans2 Indic→Indic 320M | `ai4bharat/indictrans2-indic-indic-dist-320M` | Tamil → the other four **without** passing through English. |
| IndicConformer 600M | `ai4bharat/indic-conformer-600m-multilingual` | Voice questions. CTC by default, one RNNT retry on an empty transcript. |
| Indic Parler-TTS | `ai4bharat/indic-parler-tts` | Lesson read aloud. |
| IndicXlit | pip `ai4bharat-transliteration` | Romanised student input → native script, and back for students who speak but don't read. |

**On En→Indic being a fallback rather than the main path.** Requirement 2 says the
regional explanation must be independently composed, not translated. That is the
25% regional-language score: a translated explanation reads like Google Translate
because it *is* Google Translate. So the pipeline composes in-language and uses
En→Indic only when composition fails — which is still a large upgrade on the old
behaviour there, which was a fixed stub sentence. `ExplanationOrigin` on every
lesson says which path produced it, and `test_regional_explanation_is_composed_not_translated`
fails the build if en→indic ever runs on the happy path. If you want translation as
the default, that's a one-line change in `_generate_regional` — but you'd be
trading away the thing this branch is scored on.

**Why Indic→Indic instead of routing through English.** The composed Tamil is warm,
spoken, Tanglish-flavoured teacher talk. The English `simple_explanation` is flat
and factual. Translating Tamil→Telugu directly keeps far more of the former, which
is exactly what the stitched 320M model exists for.

**Why IndicXlit is on the input path.** A 14-year-old on a phone types
"thavaram epdi saapdum", not native script. That's a bad query for retrieval and a
bad prompt for a Tamil teacher persona. It only fires on text that looks romanised
*and* isn't English — transliterating "how do plants make food" produces nonsense,
so there's an English-marker check before it runs, with tests pinning both.

> ⚠️ **Warm the weights the night before the demo.** First use downloads roughly
> 1GB per IndicTrans2 direction, ~2.4GB for IndicConformer, ~3GB for Parler-TTS.
> Hit `/api/tutor/capabilities` after a warm-up request to confirm `loaded`.

### Environment variables

**Provider is pluggable.** Set `GEMINI_API_KEY` (free tier, no credit card —
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)) or
`ANTHROPIC_API_KEY` (paid). With neither, every lesson degrades to the template
path instead of erroring, which still demos the guardrails.

| Variable | Default | Why |
|---|---|---|
| `GEMINI_API_KEY` | — | Free tier. Auto-selected when present. |
| `ANTHROPIC_API_KEY` / `LLM_API_KEY` | — | Claude. Either name works. |
| `LLM_PROVIDER` | auto | `gemini` \| `anthropic`. Leave blank to pick by whichever key exists. |
| `TUTOR_LESSON_MODEL` | per provider | `gemini-3.7-flash` / `claude-opus-5`. |
| `TUTOR_CHECK_MODEL` | per provider | The grounding audit. On Claude this is Haiku — a cheap model is the right tool for a yes/no check. |
| `TUTOR_EFFORT` | `medium` (Claude only) | Demo-latency choice, not a cost one — at `high` a lesson takes long enough to feel dead on stage. |
| `TUTOR_INDIC_EN` | `1` | IndicTrans2 backtranslation validation. `0` = skip. |
| `TUTOR_EN_INDIC` | `1` | IndicTrans2 en→indic fallback. |
| `TUTOR_INDIC_INDIC` | `1` | IndicTrans2 indic→indic. |
| `TUTOR_ASR` | `1` | IndicConformer. Set `0` on a laptop without the RAM. |
| `TUTOR_TTS` | `1` | Indic Parler-TTS. |
| `TUTOR_XLIT` | `1` | IndicXlit. |

Each `INDICTRANS2_*`, `INDIC_CONFORMER_MODEL` and `INDIC_PARLER_TTS_MODEL` var can
also override the checkpoint id if you want the 1B IndicTrans2 variants instead of
the distilled ones.

## The one function

```python
from app.tutor.lesson_generator import generate

lesson = generate(grounding_result, query)   # -> TutorLesson
```

Accepts a `GroundingResult` or a bare `RetrievedChunk`. Never raises for LLM
failure — it degrades. Whoever wires `main` calls exactly this.

## Integration seam

`router.set_grounding_provider(fn)` swaps the mock for the real retriever.
`fn(page_id: str, query: str | None) -> GroundingResult`. One line at merge time;
nothing else in this branch moves.

## Design decisions worth knowing

**Two independent LLM calls, not one.** The spec's starter prompt returned Tamil in
the same JSON as the English. It is now a separate call with its own system prompt,
and that call **never sees the English explanation** — only the textbook chunk and
the concept list. A model shown an English paragraph translates it; a model shown a
textbook page and told it is a Tamil schoolteacher teaches it. There is a test that
asserts the English output does not appear in the Tamil prompt, because this is
exactly the kind of thing that quietly regresses during a merge.

**The refusal is a code branch, not a prompt instruction.** `is_in_scope == False`
returns before any LLM object is touched. `test_grounding_refusal.py` passes an
`ExplodingClient` that raises on any call, so the test proves the *absence* of a
call rather than the shape of the answer. Same guard fires on `is_in_scope=True`
with zero chunks.

**Validation order is deterministic-first.** Analogy coverage and excerpt
verification are free string checks and run before the claim-audit call — bad output
never costs an API round trip. Only if those pass do we spend the Haiku audit.

**Degradation ladder.** generate → validate → regenerate once → template built
directly from `concepts` and page sentences (no free generation, so it cannot
hallucinate) → for out-of-scope, refusal. Every lesson carries `source` so the UI
and the judges can see which rung it landed on.

**Fuzzy concept matching is deliberately lenient.** A false "present" costs nothing;
a false "missing" burns a regeneration. It tolerates plurals, inflection and word
order, and it checks the Tamil text *and* the backtranslation, because Tanglish
keeps `chlorophyll` in English on purpose and round-tripping it is pointless.

**IndicTrans2 is optional.** ~1GB of weights, lazily loaded, commented out of
`requirements.txt`. If it is missing, concepts are checked against the raw Tamil
text and the report says so. A missing optional dependency must not fail a lesson
on demo day.

## Before merge

- [x] Out-of-scope input → `grounded=False`, correct refusal, LLM never called
- [x] Photosynthesis mock chunk → backtranslated Tamil fuzzy-matches the concepts
- [ ] **Run each AI4Bharat model once for real.** Every one is written-but-unexercised
      — the model ids are verified against Hugging Face, the loading code is not.
      Install, hit `/api/tutor/capabilities`, confirm four `loaded`. Budget an hour;
      this is where version drift in `IndicTransToolkit` / `parler-tts` will bite.
- [ ] **A native Tamil speaker reviews 3–4 real generated outputs** — "sounds like a
      teacher" vs "sounds like Google Translate". This is the only check that
      matters for the 25% and no test can stand in for it. Run it on real API
      output, not the fixtures.
- [ ] Lock `backend/app/shared/schemas.py` with the `feat/rag-grounding` owner

Note for Windows demos: printing Tamil to a `cmd`/PowerShell console needs
`PYTHONIOENCODING=utf-8`. Over HTTP it is fine — JSON is UTF-8.

---

# Ideas worth stealing (ranked by demo payoff per hour)

Ordered by what actually moves the rubric, not by how clever they sound.

### 1. Show the grounding, don't claim it — "Where did this come from?" (grounding, 25%)

The `trace` is already returned. Render it: click any sentence of the explanation
and the source line on the page image highlights. Judges score grounding on whether
they *believe* you; a highlight that lands on the right line in front of them is
worth more than any architecture slide. Half a day of frontend on data that already
exists.

### 2. Refuse out loud, with the near-miss (grounding, 25%)

When you refuse, don't just say "not in this chapter" — say *"that isn't on this
page. The closest thing here is photosynthesis (page 42) — want that instead?"*
Same refusal, but it demonstrates you know what the page contains, which is the
opposite of a model that refuses because it is confused. Turns the out-of-scope
question — the moment judges are trying to catch you out — into your strongest demo
beat. Needs one field from the RAG branch: the top chunk even when below threshold.

### 3. Two Tamil registers, one toggle (regional language, 25%)

Generate the Tamil at two levels and let the student flip: **classroom Tanglish**
(what we do now) and **pure Tamil** for the terms a textbook exam will actually ask
for. Same call, two variants. It directly answers the objection a Tamil-speaking
judge will raise — *"real students need the formal term for the exam"* — and it
shows the Tanglish is a deliberate pedagogical choice rather than the model being
lazy. Cheap: one extra generation, no new infrastructure.

### 4. Analogy fit is a retrieval problem, not a generation problem (analogy quality)

Instead of hoping the model picks a good scene, keep a small hand-written list of
~15 analogy domains a Tamil 14-year-old genuinely knows (kitchen, paddy field, tea
shop, bus stop, cricket, temple festival, autorickshaw, school ground) and pass 3
candidates into the prompt with the instruction to pick the best fit and justify it
in one line. Constrained choice beats open invention, gives you variety across
lessons without randomness, and makes the analogy step auditable. About two hours.

### 5. The analogy stays the same across the whole session (differentiator)

Once a page uses the kitchen for photosynthesis, reuse the *same kitchen* for
respiration on the next page — the cook is still chlorophyll, the dosa is still
glucose. Store the chosen scene per session and pass it into subsequent prompts as
a constraint. That is what a real teacher does, no chatbot does it, and it is
visible in a two-page demo. Small state object, big perceived difference.

### 6. Feed the teach-back branch a misconception map, not just prose (cross-branch, 25%)

You already produce `analogy_map` — one concept, one concrete component. Emit
alongside it the *predictable* wrong answer per concept ("students say the plant
eats soil", "students say oxygen is the food"). Teach-back evaluation can then
pinpoint a named misconception instead of scoring free text, which is exactly the
"pinpoints the specific misconception" wording in the rubric. This branch is where
that data is cheapest to produce, and it makes the teammate on `feat/teachback` much
faster. Talk to them before building it.

### 7. Confidence-shaped answers (guardrail, low effort)

You have three quality signals already: `source` (generated / regenerated /
template), unsupported-claim count, and Tamil concepts recovered. Combine them into
one badge on the lesson card. A tutor that says "I'm sure of this" vs "I'm only
partly covering this page" reads as honest engineering; hiding the template
fallback behind an identical-looking card reads as a system that doesn't know when
it is weak. Twenty minutes, and it makes the whole guardrail story visible.

### 8. Cache the page prefix (cost/latency, invisible but real)

Every call on a page re-sends the same chunk text. Put a cache breakpoint after the
textbook content and you pay ~10% on repeat calls — which matters once teach-back
and practice generation are also hitting the same page, and it takes the third
question on a page from slow to instant during the demo. One parameter, and it
compounds across three branches.

### 9. Voice is wired — now close the loop (stretch goal, mostly done)

`/api/tutor/ask` (speak a question) and `/api/tutor/explain/speak` (hear the lesson)
exist. What's left is the frontend: a hold-to-talk button and an autoplay, which
lands the "whole loop hands-free in the regional language" stretch goal.

Worth knowing why this sounds good rather than robotic: the Tamil is already
spoken-register teacher talk, and TTS on stiff written Tamil is what sounds
mechanical. The composition decision and the voice quality are the same decision —
if anyone switches the Tamil to machine translation to save time, the voice demo
degrades with it.

### 10. Show the same lesson in three languages at once (differentiator)

`translate_to` already returns them. Rendering Tamil, Telugu and Hindi side by side
from *one* grounded page is a 10-second demo beat that no single-language team can
match, and it makes the Indic→Indic model visible rather than a line in a table.

**What I would not build:** a fluency-scoring model for Tamil (the spec is right —
not worth the time; a native speaker spot-check is both cheaper and more credible),
and per-student difficulty adaptation inside this branch (that belongs to
`feat/practice-generation`; duplicating it here creates two sources of truth for
"what does this student know").
