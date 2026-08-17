"""
Core orchestration for teach-back evaluation.

Public function: evaluate_teachback(...)

Parsing is STRICT by design: if the LLM returns malformed JSON or a
response that doesn't match the expected schema, this raises a clear
error rather than fabricating a partial/fallback result. Any
user-facing fallback behaviour (e.g. "let's try that again") belongs
in the API/orchestration layer that calls this module, not here.
"""

import json

from pydantic import ValidationError

from app.teachback.llm_client import call_llm
from app.teachback.prompts import build_teachback_prompt
from app.teachback.schemas import TeachbackResult

# Deterministic mapping — the LLM decides understanding_level, Python decides next_action.
# Keeping this in code (not the prompt) makes demo behaviour predictable and easy to reason about.
_NEXT_ACTION_BY_LEVEL = {
    "strong": "practice",
    "partial": "retry_teachback",
    "weak": "reteach",
}

# Keys the LLM is expected to return. next_action is deliberately excluded —
# it is computed here, never trusted from the model.
_EXPECTED_LLM_KEYS = {
    "understanding_level",
    "understood",
    "misconceptions",
    "missing_concepts",
    "feedback",
    "needs_reteach",
}


class TeachbackEvaluationError(Exception):
    """Raised when the LLM response cannot be parsed into a valid TeachbackResult."""


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` even when told not to. Strip that, nothing else."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_llm_response(raw: str) -> dict:
    """
    Strictly parse and validate the LLM's raw text response.

    Raises TeachbackEvaluationError with a clear, specific message on any failure:
    invalid JSON, missing keys, unexpected extra keys, or wrong types.
    Never silently substitutes a default value.
    """
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TeachbackEvaluationError(
            f"LLM response was not valid JSON: {exc}. Raw response: {raw!r}"
        ) from exc

    if not isinstance(data, dict):
        raise TeachbackEvaluationError(
            f"Expected a JSON object from the LLM, got {type(data).__name__}: {raw!r}"
        )

    missing_keys = _EXPECTED_LLM_KEYS - data.keys()
    if missing_keys:
        raise TeachbackEvaluationError(
            f"LLM response is missing required keys: {sorted(missing_keys)}. Raw response: {raw!r}"
        )

    extra_keys = data.keys() - _EXPECTED_LLM_KEYS
    if extra_keys:
        raise TeachbackEvaluationError(
            f"LLM response contained unexpected keys: {sorted(extra_keys)}. Raw response: {raw!r}"
        )

    if data["understanding_level"] not in _NEXT_ACTION_BY_LEVEL:
        raise TeachbackEvaluationError(
            "LLM returned an invalid understanding_level: "
            f"{data['understanding_level']!r}. Expected one of {sorted(_NEXT_ACTION_BY_LEVEL)}."
        )

    return data


async def evaluate_teachback(concept: str, tutor_explanation: str, student_answer: str) -> TeachbackResult:
    """
    Evaluate a student's teach-back answer against the concept they were taught.

    Domain-agnostic: works for any `concept` passed in at runtime.

    Raises TeachbackEvaluationError if the LLM response is malformed or violates
    the expected schema. Callers should catch this and decide on user-facing
    fallback behaviour (this module does not fabricate results silently).
    """
    prompt = build_teachback_prompt(concept, tutor_explanation, student_answer)
    raw_response = await call_llm(prompt)
    parsed = _parse_llm_response(raw_response)

    next_action = _NEXT_ACTION_BY_LEVEL[parsed["understanding_level"]]

    try:
        return TeachbackResult(
            understanding_level=parsed["understanding_level"],
            understood=parsed["understood"],
            misconceptions=parsed["misconceptions"],
            missing_concepts=parsed["missing_concepts"],
            feedback=parsed["feedback"],
            needs_reteach=parsed["needs_reteach"],
            next_action=next_action,
        )
    except ValidationError as exc:
        raise TeachbackEvaluationError(
            f"LLM response did not match the expected TeachbackResult schema: {exc}"
        ) from exc
