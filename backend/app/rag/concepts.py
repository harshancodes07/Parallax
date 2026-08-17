"""Key terms per chunk, for the tutor's analogy and regional-language checks.

The tutor requires every concept to be mapped to a concrete component of its
analogy, and separately checks that each one survived into the Tamil lesson. So
this list is not decoration — it is what both of those guardrails run on. An
empty list means both silently pass everything, which loses the grounding score
without any visible failure.

Two consequences shape the design:

* Extraction happens at **ingest**, not per query. One cost per upload, stored
  on the chunk, so retrieval stays a pure vector lookup.
* Precision beats recall. A concept the page barely mentions still has to be
  worked into an analogy a 14-year-old understands, and a bad one costs a
  regeneration. Six good terms beat fifteen shaky ones.

The LLM path is preferred. The deterministic fallback exists so ingestion works
with no API key at all — it is noticeably blunter, and says so in the logs.
"""

from __future__ import annotations

import functools
import logging
import re

from .config import CONCEPT_MAX, CONCEPT_MODEL

log = logging.getLogger(__name__)

SYSTEM = """You extract the key concepts a teacher would have to explain from one page of a school textbook.

Return ONLY a comma-separated list, lowercase, no numbering, no commentary.

Rules:
- Between 3 and {max_concepts} terms, fewest that still cover the page.
- Use the page's own wording ("carbon dioxide", not "CO2").
- Concrete nouns and named processes only — the things a diagram would label.
- Skip generic words a student already knows (plant, air, energy, process) unless
  the page is specifically about that thing.
- If the page teaches one named process, include it."""


# Words that carry no teaching weight on their own. Deliberately short: this is a
# fallback, and over-filtering here silently drops real terms.
_STOPWORDS = frozenset("""
a an the this that these those it its is are was were be been being
of in on at to from by for with without into through during above below
and or but not no nor so than then thus also as if when while because
which who whom whose what where how why
can could will would shall should may might must
they them their there here own same other another each every both few more most
some such only very just about over under again further once
plant plants make makes made made use used uses using called call
takes take taken taking gives give given giving forms form formed
one two three first second next also more much many
""".split())

_MULTIWORD = re.compile(
    r"\b(carbon dioxide|water vapour|water vapor|water cycle|"
    r"food chain|solar energy|green pigment|by-product|"
    r"nitrogen cycle|digestive system|circulatory system)\b",
    re.IGNORECASE,
)

_WORD = re.compile(r"[a-z][a-z-]{3,}")


def _singular(word: str) -> str:
    """Crude de-pluralisation that leaves Greek/Latin science nouns alone.

    Naively stripping a trailing "s" turns "photosynthesis" into
    "photosynthesi" — a concept the tutor would then hunt for in the lesson and
    never find, burning a regeneration on a word that was never real.
    """
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "is", "us", "os")):
        return word[:-1]
    return word


def _fallback(text: str, max_concepts: int) -> list[str]:
    """No-LLM extraction: known multi-word terms, then frequency-ranked nouns.

    Blunt by construction. It exists so an upload still produces *something* for
    the tutor's guardrails to check rather than an empty list, which would
    disable them silently.
    """
    lowered = text.lower()
    found: list[str] = []

    for match in _MULTIWORD.finditer(lowered):
        term = match.group(0).lower()
        if term not in found:
            found.append(term)

    # Remove the multi-word hits so their parts don't compete as single words.
    remainder = _MULTIWORD.sub(" ", lowered)

    counts: dict[str, int] = {}
    for word in _WORD.findall(remainder):
        stem = _singular(word)
        if stem in _STOPWORDS or word in _STOPWORDS:
            continue
        counts[stem] = counts.get(stem, 0) + 1

    # Frequent first, then longer words — a longer term is usually the technical
    # one in a sentence that also contains a common one.
    ranked = sorted(counts, key=lambda w: (-counts[w], -len(w), w))
    for word in ranked:
        if len(found) >= max_concepts:
            break
        if word not in found:
            found.append(word)

    return found[:max_concepts]


@functools.lru_cache(maxsize=1024)
def extract(text: str, max_concepts: int = CONCEPT_MAX) -> tuple[str, ...]:
    """Key terms for one chunk, best first.

    Cached because ingestion re-chunks the same page during calibration runs.
    Returns a tuple so the cache stays hashable; callers want `list(...)`.
    """
    if not text or not text.strip():
        return ()

    try:
        from .guardrail import _get_client  # local import avoids a cycle

        resp = _get_client().messages.create(
            model=CONCEPT_MODEL,
            max_tokens=1000,
            output_config={"effort": "low"},
            system=SYSTEM.format(max_concepts=max_concepts),
            messages=[{"role": "user", "content": text}],
        )
        reply = "".join(b.text for b in resp.content if b.type == "text").strip()
        terms = _parse(reply, max_concepts)
        if terms:
            return terms
        log.warning("concept extraction returned nothing usable; using the fallback")
    except Exception as exc:  # noqa: BLE001 - ingestion must not die on this
        log.warning("concept extraction unavailable (%s); using the fallback", exc)

    return tuple(_fallback(text, max_concepts))


def _parse(reply: str, max_concepts: int) -> tuple[str, ...]:
    """Pull a clean term list out of the model's reply.

    Tolerant of the model ignoring 'comma-separated' and returning bullets or
    numbered lines, which it occasionally does on short pages.
    """
    flattened = reply.replace("\n", ",")
    terms: list[str] = []
    for raw in flattened.split(","):
        term = raw.strip().strip("-•*0123456789. \t").lower()
        if not term or len(term) > 40:
            continue
        if term in terms:
            continue
        terms.append(term)
    return tuple(terms[:max_concepts])
