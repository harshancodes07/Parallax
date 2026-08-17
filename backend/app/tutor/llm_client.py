"""Thin wrapper around the Claude API.

Kept deliberately small so tests can swap it out wholesale. The `anthropic`
import is lazy: nothing in this branch needs the SDK installed until an actual
call is made, which keeps the refusal path and the unit tests dependency-free.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Generation model. Opus 5 runs adaptive thinking by default, so max_tokens has to
# cover thinking + response text.
LESSON_MODEL = os.getenv("TUTOR_LESSON_MODEL", "claude-opus-5")

# The grounding self-check is a small yes/no audit — a cheap model is the right tool,
# and it keeps the extra latency off the demo's critical path.
CHECK_MODEL = os.getenv("TUTOR_CHECK_MODEL", "claude-haiku-4-5")

# `medium` is a deliberate demo-latency choice, not a cost one: at `high` a lesson can
# take long enough to feel dead on stage. Bump to `high` if answer quality slips.
LESSON_EFFORT = os.getenv("TUTOR_EFFORT", "medium")

MAX_TOKENS = int(os.getenv("TUTOR_MAX_TOKENS", "8000"))


class LLMUnavailable(RuntimeError):
    """No API key, SDK missing, or the API refused/failed. Callers fall back to a template."""


class LLMClient:
    """One client for the whole branch. Instantiate once (see `lesson_generator`)."""

    def __init__(self, api_key: str | None = None) -> None:
        self._explicit_key = api_key
        self._client: Any | None = None

    @property
    def _api_key(self) -> str | None:
        """Resolved on use, not at construction.

        This client is instantiated at import time, so reading the key in
        `__init__` would capture the environment before `load_dotenv()` — the
        exact failure where a correct `.env` silently produces template lessons.

        `.env.example` ships `LLM_API_KEY`; the Anthropic SDK reads
        `ANTHROPIC_API_KEY`. Accept either so nobody loses twenty minutes.
        """
        return (
            self._explicit_key
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("LLM_API_KEY")
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise LLMUnavailable("anthropic SDK is not installed (pip install anthropic)") from exc
        if not self._api_key:
            raise LLMUnavailable("no ANTHROPIC_API_KEY / LLM_API_KEY in the environment")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ---------------------------------------------------------------- structured JSON
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Constrained generation. The response is guaranteed to match `schema`."""
        client = self._get_client()
        try:
            response = client.messages.create(
                model=model or LESSON_MODEL,
                max_tokens=max_tokens or MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": LESSON_EFFORT,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except Exception as exc:  # noqa: BLE001 - surface everything as one failure mode
            raise LLMUnavailable(f"lesson generation call failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMUnavailable("model declined the request")

        text = _first_text(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"model returned non-JSON despite schema: {text[:200]}") from exc

    # ---------------------------------------------------------------- free text
    def complete_text(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model or LESSON_MODEL,
            "max_tokens": max_tokens or MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        # Haiku 4.5 rejects `effort`; only send it for the models that accept it.
        chosen_effort = effort or (LESSON_EFFORT if (model or LESSON_MODEL) != CHECK_MODEL else None)
        if chosen_effort:
            kwargs["output_config"] = {"effort": chosen_effort}

        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"text generation call failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMUnavailable("model declined the request")
        return _first_text(response).strip()


def _first_text(response: Any) -> str:
    """Pull the text out of a Messages response, skipping thinking blocks."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise LLMUnavailable("response contained no text block")


# Schema for the English lesson. Structured outputs require `additionalProperties: false`
# and an explicit `required` list on every object.
LESSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "simple_explanation": {"type": "string"},
        "analogy": {"type": "string"},
        "analogy_map": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "analogy_component": {"type": "string"},
                },
                "required": ["concept", "analogy_component"],
                "additionalProperties": False,
            },
        },
        "textbook_excerpt": {"type": "string"},
    },
    "required": ["topic", "simple_explanation", "analogy", "analogy_map", "textbook_excerpt"],
    "additionalProperties": False,
}
