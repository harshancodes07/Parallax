"""Shared test fixtures for the tutor branch.

The fake client is the whole point: every test below runs with zero network
calls, so `pytest` is green on a laptop with no API key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Also set via pytest.ini; kept here so `pytest tests/...` works from any cwd.
BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Kill switches BEFORE the app imports. Without these, any test that reaches a
# LazyComponent tries to import torch + transformers and load real weights —
# which cost 20s on the first miss and would download gigabytes on a machine
# where they'd actually succeed. Tests assert the contract around the models,
# never the models themselves.
for _switch in (
    "TUTOR_INDIC_EN",
    "TUTOR_EN_INDIC",
    "TUTOR_INDIC_INDIC",
    "TUTOR_ASR",
    "TUTOR_TTS",
    "TUTOR_XLIT",
):
    os.environ.setdefault(_switch, "0")

from app.tutor.llm_client import CHECK_MODEL, LLMUnavailable  # noqa: E402


class FakeClient:
    """Stands in for `LLMClient` and records exactly what was asked of it."""

    def __init__(
        self,
        json_responses: list | None = None,
        tamil_responses: list | None = None,
        claim_responses: list | None = None,
    ) -> None:
        self.json_responses = list(json_responses or [])
        self.tamil_responses = list(tamil_responses or [])
        self.claim_responses = list(claim_responses or [])
        self.json_calls: list[dict] = []
        self.tamil_calls: list[dict] = []
        self.claim_calls: list[dict] = []

    # -- LLMClient surface ------------------------------------------------
    def complete_json(self, *, system, user, schema, model=None, max_tokens=None):
        self.json_calls.append({"system": system, "user": user})
        if not self.json_responses:
            raise LLMUnavailable("fake client ran out of json responses")
        nxt = self.json_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def complete_text(self, *, system, user, model=None, max_tokens=None, effort=None):
        if model == CHECK_MODEL:
            self.claim_calls.append({"user": user})
            return self.claim_responses.pop(0) if self.claim_responses else "NONE"
        self.tamil_calls.append({"system": system, "user": user})
        if not self.tamil_responses:
            raise LLMUnavailable("fake client ran out of tamil responses")
        nxt = self.tamil_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    # -- assertions -------------------------------------------------------
    @property
    def total_calls(self) -> int:
        return len(self.json_calls) + len(self.tamil_calls) + len(self.claim_calls)


class ExplodingClient:
    """Any call at all is a test failure. Used to prove the refusal path is LLM-free."""

    def complete_json(self, **kwargs):  # noqa: ANN003
        raise AssertionError("LLM generation was called on an out-of-scope request")

    def complete_text(self, **kwargs):  # noqa: ANN003
        raise AssertionError("LLM generation was called on an out-of-scope request")


GOOD_LESSON = {
    "topic": "Photosynthesis",
    "simple_explanation": (
        "Green plants make their own food using sunlight. "
        "They turn carbon dioxide and water into glucose and give out oxygen."
    ),
    "analogy": (
        "Think of the leaf as your kitchen. The chlorophyll is the cook, sunlight is the "
        "gas flame, carbon dioxide from the air and water from the roots are the two "
        "ingredients, glucose is the dosa that comes out, and oxygen is the steam that "
        "escapes into the room."
    ),
    "analogy_map": [
        {"concept": "chlorophyll", "analogy_component": "the cook in the kitchen"},
        {"concept": "sunlight", "analogy_component": "the gas flame"},
        {"concept": "carbon dioxide", "analogy_component": "the first ingredient"},
        {"concept": "water", "analogy_component": "the second ingredient"},
        {"concept": "glucose", "analogy_component": "the dosa that comes out"},
        {"concept": "oxygen", "analogy_component": "the steam that escapes"},
    ],
    "textbook_excerpt": (
        "The green pigment chlorophyll present in the leaves absorbs sunlight."
    ),
}

GOOD_TAMIL = (
    "பாருங்க, இலை-ய ஒரு சமையலறை மாதிரி நினைச்சுக்கோங்க. அதுல இருக்கற chlorophyll தான் சமையல்காரர். "
    "sunlight தான் அடுப்பு நெருப்பு. காத்துல இருந்து வர்ற carbon dioxide-உம், வேர்ல இருந்து ஏறி வர்ற "
    "water-உம் தான் நம்ம சாமான். இதுல இருந்து செடி glucose-ஐ தயார் பண்ணுது. மிச்சம் வர்ற oxygen "
    "வெளிய போயிடும். அது தான் நாம சுவாசிக்கறோம்."
)


@pytest.fixture(autouse=True)
def no_llm_translation_fallback(monkeypatch):
    """Stop the LLM translation fallback from making real network calls.

    `llm_translate` builds its own client rather than using the injected fake, so
    on a machine that happens to have GEMINI_API_KEY set, every test touching the
    Tamil check would hit the API. Tests that want the fallback stub it themselves.
    """
    from app.tutor import llm_translate

    monkeypatch.setattr(llm_translate, "to_english", lambda *a, **k: None)


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient(json_responses=[GOOD_LESSON], tamil_responses=[GOOD_TAMIL])


@pytest.fixture
def exploding_client() -> ExplodingClient:
    return ExplodingClient()
