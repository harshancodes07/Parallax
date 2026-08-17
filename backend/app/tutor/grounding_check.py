"""Validation layer: does the generated lesson actually trace back to the page?

Three independent checks, all cheap enough to run on every request:

* `check_claims`   — LLM audit: any claim not present in the source text?
* `check_analogy_coverage` — deterministic: is every concept mapped?
* `check_excerpt`  — deterministic: is the quote really from the page?

The deterministic ones run first so an obvious failure never costs an API call.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata

from app.tutor import prompts
from app.tutor.llm_client import CHECK_MODEL, LLMClient, LLMUnavailable

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

# Above this, a concept token and a text token are "the same word" — covers
# plurals, inflection and OCR-ish noise without matching unrelated words.
_FUZZY_CUTOFF = 0.85


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def concept_present(concept: str, text: str) -> bool:
    """Is `concept` mentioned in `text`? Tolerant of inflection and word order.

    Deliberately lenient: this gate exists to catch a concept being *dropped*,
    not to grade phrasing. A false "present" costs nothing; a false "missing"
    burns a regeneration.
    """
    haystack = normalise(text)
    needle = normalise(concept)
    if not needle:
        return True
    if needle in haystack:
        return True

    hay_tokens = haystack.split()
    for token in needle.split():
        if token in hay_tokens or token in haystack:
            continue
        if not difflib.get_close_matches(token, hay_tokens, n=1, cutoff=_FUZZY_CUTOFF):
            return False
    return True


def missing_concepts(concepts: list[str], *texts: str) -> list[str]:
    """Concepts absent from all of `texts`, in the order they were listed."""
    blob = " \n ".join(t for t in texts if t)
    return [c for c in concepts if not concept_present(c, blob)]


def check_analogy_coverage(
    concepts: list[str],
    analogy: str,
    analogy_map: list[dict] | None = None,
) -> list[str]:
    """Requirement 4: every concept must map to an explicit analogy component.

    A concept counts as covered only if it is named in the analogy prose *or*
    carries a non-empty mapping entry — a mapping with an empty component is
    exactly the vague output we want to reject.
    """
    mapped: list[str] = []
    for entry in analogy_map or []:
        component = (entry.get("analogy_component") or "").strip()
        concept = (entry.get("concept") or "").strip()
        if concept and component:
            mapped.append(concept)

    missing: list[str] = []
    for concept in concepts:
        if concept_present(concept, analogy):
            continue
        if any(concept_present(concept, m) for m in mapped):
            continue
        missing.append(concept)
    return missing


def check_excerpt(excerpt: str, source_text: str) -> bool:
    """Is the quote near-verbatim from the page? Guards against a fabricated citation."""
    quote = normalise(excerpt)
    source = normalise(source_text)
    if not quote:
        return False
    if quote in source:
        return True
    # Allow small drift (a trimmed clause, a fixed typo) but nothing rewritten.
    matcher = difflib.SequenceMatcher(None, quote, source)
    match = matcher.find_longest_match(0, len(quote), 0, len(source))
    return match.size >= 0.85 * len(quote)


def check_claims(client: LLMClient, source_text: str, explanation: str) -> list[str]:
    """Post-generation audit. Returns unsupported claims; empty list means clean.

    If the auditor itself is unavailable we return `[]` rather than failing the
    lesson — the deterministic checks above still applied, and a broken auditor
    should not turn a good lesson into a template.
    """
    try:
        reply = client.complete_text(
            system=prompts.CLAIM_CHECK_SYSTEM,
            user=prompts.CLAIM_CHECK_USER.format(
                source_text=source_text, explanation=explanation
            ),
            model=CHECK_MODEL,
            max_tokens=1000,
        )
    except LLMUnavailable as exc:
        log.warning("grounding self-check skipped: %s", exc)
        return []

    stripped = reply.strip()
    if not stripped or normalise(stripped) in {"none", "none."}:
        return []
    if normalise(stripped).startswith("none"):
        return []

    return [line.strip("-• \t") for line in stripped.splitlines() if line.strip()]
