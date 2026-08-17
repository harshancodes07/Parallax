"""LLM access for the tutor branch, with a pluggable provider.

Two providers, one interface. Everything downstream — `lesson_generator`,
`grounding_check` — only knows `complete_json` / `complete_text`, so swapping
providers is an env var, not a refactor.

    LLM_PROVIDER=gemini      + GEMINI_API_KEY      (free tier, no card)
    LLM_PROVIDER=anthropic   + ANTHROPIC_API_KEY

If `LLM_PROVIDER` is unset we pick whichever key is present. With no key at all,
every call raises `LLMUnavailable` and the tutor degrades to template lessons —
which is a working demo of the guardrails, just not of the teaching.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

log = logging.getLogger(__name__)

MAX_TOKENS = int(os.getenv("TUTOR_MAX_TOKENS", "8000"))


class LLMUnavailable(RuntimeError):
    """No API key, SDK missing, or the API refused/failed. Callers fall back to a template."""


class LLMRateLimited(LLMUnavailable):
    """Quota or rate limit. Transient — the request would succeed later.

    Kept distinct because the two must not be handled the same way. A missing
    model can be answered from a template; a rate limit means we do not know
    the answer yet, and telling the student "that isn't in this chapter" would
    be false.
    """


_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "quota",
    "too_many_requests",
    "resource_exhausted",
    "overloaded",
    "503",
    "529",
)


def _classify(exc: Exception, context: str) -> LLMUnavailable:
    message = str(exc)
    haystack = message.casefold()
    if any(marker in haystack for marker in _TRANSIENT_MARKERS):
        return LLMRateLimited(f"{context}: {message}")
    return LLMUnavailable(f"{context}: {message}")


# --------------------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------------------


# Anything speaking the OpenAI chat-completions API is one provider to us; only the
# base URL, key and default model differ. Adding OpenRouter or Together is a row.
OPENAI_COMPATIBLE: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "key_env": "XAI_API_KEY",
        "model": "grok-4.6",
    },
}

_PROVIDER_ALIASES = {
    "gemini": "gemini",
    "google": "gemini",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "groq": "groq",
    "grok": "xai",  # people write "grok" for both; xAI is the one that owns the name
    "xai": "xai",
}


def _detect_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().casefold()
    if explicit in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[explicit]
    # No explicit choice: pick by whichever key is present.
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("XAI_API_KEY"):
        return "xai"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"


PROVIDER = _detect_provider()

if PROVIDER in OPENAI_COMPATIBLE:
    _preset = OPENAI_COMPATIBLE[PROVIDER]
    LESSON_MODEL = os.getenv("TUTOR_LESSON_MODEL") or os.getenv("LLM_MODEL") or _preset["model"]
    CHECK_MODEL = os.getenv("TUTOR_CHECK_MODEL", LESSON_MODEL)
    LESSON_EFFORT = os.getenv("TUTOR_EFFORT", "")
elif PROVIDER == "gemini":
    # One model for both jobs — the free tier is fast enough that a separate
    # cheap model for the grounding audit buys nothing.
    LESSON_MODEL = os.getenv("TUTOR_LESSON_MODEL", "gemini-3.7-flash")
    CHECK_MODEL = os.getenv("TUTOR_CHECK_MODEL", "gemini-3.7-flash")
    LESSON_EFFORT = os.getenv("TUTOR_EFFORT", "")
else:
    # Opus 5 runs adaptive thinking by default, so max_tokens covers thinking + text.
    LESSON_MODEL = os.getenv("TUTOR_LESSON_MODEL", "claude-opus-5")
    # The grounding self-check is a small yes/no audit — a cheap model is the right
    # tool, and it keeps the extra latency off the demo's critical path.
    CHECK_MODEL = os.getenv("TUTOR_CHECK_MODEL", "claude-haiku-4-5")
    # `medium` is a deliberate demo-latency choice, not a cost one: at `high` a lesson
    # can take long enough to feel dead on stage.
    LESSON_EFFORT = os.getenv("TUTOR_EFFORT", "medium")


class LLMClient(Protocol):
    """What the rest of the branch depends on. Structural — fakes need no base class."""

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> str: ...


# --------------------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------------------


class GeminiClient:
    """Google Gemini via the `google-genai` SDK.

    Free tier, no credit card, and strong on Indian languages — which is the
    reason to prefer it here over a faster provider running open models that
    have seen far less Tamil.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._explicit_key = api_key
        self._client: Any | None = None

    @property
    def _api_key(self) -> str | None:
        return (
            self._explicit_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise LLMUnavailable(
                "google-genai is not installed (pip install -U google-genai)"
            ) from exc
        if not self._api_key:
            raise LLMUnavailable("no GEMINI_API_KEY in the environment")
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _create(self, **kwargs: Any) -> Any:
        client = self._get_client()
        try:
            return client.interactions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - one failure mode for every caller
            raise _classify(exc, "gemini call failed") from exc

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        interaction = self._create(
            model=model or LESSON_MODEL,
            system_instruction=system,
            input=user,
            generation_config={"max_output_tokens": max_tokens or MAX_TOKENS},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
        )
        text = (interaction.output_text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"gemini returned non-JSON despite schema: {text[:200]}") from exc

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> str:
        interaction = self._create(
            model=model or LESSON_MODEL,
            system_instruction=system,
            input=user,
            generation_config={"max_output_tokens": max_tokens or MAX_TOKENS},
        )
        return (interaction.output_text or "").strip()


# --------------------------------------------------------------------------------------
# OpenAI-compatible (Groq, xAI/Grok, and anything else with that API)
# --------------------------------------------------------------------------------------


class OpenAICompatibleClient:
    """Any provider speaking the OpenAI chat-completions API.

    Structured output support varies across these providers and across models on
    the same provider, and it is not reliably advertised. So we ask for a strict
    `json_schema` first and, if the provider rejects it, fall back to
    `json_object` with the schema stated in the prompt. The second path is
    weaker — the shape is requested rather than enforced — which is exactly why
    the analogy-coverage and excerpt checks downstream are not optional.
    """

    def __init__(self, provider: str, api_key: str | None = None) -> None:
        preset = OPENAI_COMPATIBLE[provider]
        self.provider = provider
        self.base_url = os.getenv("LLM_BASE_URL") or preset["base_url"]
        self._key_env = preset["key_env"]
        self._explicit_key = api_key
        self._client: Any | None = None
        self._schema_mode: str | None = None  # learned on first use, then reused

    @property
    def _api_key(self) -> str | None:
        return self._explicit_key or os.getenv(self._key_env) or os.getenv("LLM_API_KEY")

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise LLMUnavailable("openai SDK is not installed (pip install openai)") from exc
        if not self._api_key:
            raise LLMUnavailable(f"no {self._key_env} in the environment")
        self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        return self._client

    def _chat(self, *, model: str, system: str, user: str, max_tokens: int, **extra: Any) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc, f"{self.provider} call failed") from exc
        return (response.choices[0].message.content or "").strip()

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        model = model or LESSON_MODEL
        max_tokens = max_tokens or MAX_TOKENS

        if self._schema_mode != "json_object":
            try:
                text = self._chat(
                    model=model,
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "lesson", "schema": schema, "strict": True},
                    },
                )
                self._schema_mode = "json_schema"
                return _parse_json(text)
            except LLMRateLimited:
                raise
            except LLMUnavailable as exc:
                log.info("%s rejected json_schema, falling back to json_object: %s", self.provider, exc)
                self._schema_mode = "json_object"

        text = self._chat(
            model=model,
            system=f"{system}\n\nReturn ONLY JSON matching this schema:\n{json.dumps(schema)}",
            user=user,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return _parse_json(text)

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> str:
        return self._chat(
            model=model or LESSON_MODEL,
            system=system,
            user=user,
            max_tokens=max_tokens or MAX_TOKENS,
        )


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMUnavailable(f"provider returned non-JSON: {text[:200]}") from exc


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------


class AnthropicClient:
    """Claude via the official SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        self._explicit_key = api_key
        self._client: Any | None = None

    @property
    def _api_key(self) -> str | None:
        """Resolved on use, not at construction.

        This client is instantiated at import time, so reading the key in
        `__init__` would capture the environment before `load_dotenv()` — the
        exact failure where a correct `.env` silently produces template lessons.

        `.env.example` ships `LLM_API_KEY`; the SDK reads `ANTHROPIC_API_KEY`.
        Accept either so nobody loses twenty minutes to a variable name.
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

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if LESSON_EFFORT:
            output_config["effort"] = LESSON_EFFORT

        try:
            response = client.messages.create(
                model=model or LESSON_MODEL,
                max_tokens=max_tokens or MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
            )
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc, "lesson generation call failed") from exc

        if response.stop_reason == "refusal":
            raise LLMUnavailable("model declined the request")

        text = _first_text(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"model returned non-JSON despite schema: {text[:200]}") from exc

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
        # Haiku 4.5 rejects `effort`; only send it for models that accept it.
        chosen = effort or (LESSON_EFFORT if (model or LESSON_MODEL) != CHECK_MODEL else None)
        if chosen:
            kwargs["output_config"] = {"effort": chosen}

        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise _classify(exc, "text generation call failed") from exc

        if response.stop_reason == "refusal":
            raise LLMUnavailable("model declined the request")
        return _first_text(response).strip()


def _first_text(response: Any) -> str:
    """Pull the text out of a Messages response, skipping thinking blocks."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise LLMUnavailable("response contained no text block")


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def build_client() -> LLMClient:
    """The client the app runs with, chosen by `LLM_PROVIDER` or by which key exists."""
    if PROVIDER in OPENAI_COMPATIBLE:
        client = OpenAICompatibleClient(PROVIDER)
        log.info("LLM provider: %s (%s) at %s", PROVIDER, LESSON_MODEL, client.base_url)
        return client
    if PROVIDER == "gemini":
        log.info("LLM provider: gemini (%s)", LESSON_MODEL)
        return GeminiClient()
    log.info("LLM provider: anthropic (%s)", LESSON_MODEL)
    return AnthropicClient()


# Schema for the English lesson. Both providers accept this shape; Anthropic's
# structured outputs require `additionalProperties: false` and an explicit
# `required` list on every object, and Gemini tolerates both.
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
