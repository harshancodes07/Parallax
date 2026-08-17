"""Shared fixtures.

Two kinds of test live here, and the split matters:

  - Fast tests mock the vector store and the Anthropic client. They pin the
    refusal *logic* and run in milliseconds with no API key and no model
    download. Everything in the grounding contract is covered here.

  - Tests marked `slow` load the real embedding model (~1GB on first run) and
    assert on real similarity scores. They are the regression guard for the
    calibration finding: unrelated subjects must stay out.

Run everything:      pytest backend/tests
Skip the slow ones:  pytest backend/tests -m "not slow"
"""

import json
import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SCRIPTS = BACKEND / "scripts"


@pytest.fixture(scope="session")
def sample_doc() -> dict:
    return json.loads((SCRIPTS / "sample_chapter.json").read_text())


@pytest.fixture(scope="session")
def sample_questions() -> dict:
    return json.loads((SCRIPTS / "sample_questions.json").read_text())


def fake_response(text: str | None = None, *, stop_reason: str = "end_turn"):
    """A stand-in for an Anthropic Message.

    `text=None` produces a response with no text blocks at all, which is what a
    safety refusal or a thinking-truncated response actually looks like.
    """
    blocks = []
    if text is not None:
        blocks.append(types.SimpleNamespace(type="text", text=text))
    return types.SimpleNamespace(content=blocks, stop_reason=stop_reason)


@pytest.fixture
def stub_client(monkeypatch):
    """Replace the Anthropic client with one that returns a canned response.

    Returns a function: call it with the response you want the model to give.
    Also records the kwargs of the last request so tests can assert on how the
    model was called, not just what came back.
    """
    from app.rag import guardrail

    calls: list[dict] = []

    def install(response, *, raises: Exception | None = None):
        def messages_create(**kwargs):
            calls.append(kwargs)
            if raises is not None:
                raise raises
            return response

        client = types.SimpleNamespace(
            messages=types.SimpleNamespace(create=messages_create)
        )
        monkeypatch.setattr(guardrail, "_get_client", lambda: client)
        return calls

    return install


@pytest.fixture
def stub_retrieval(monkeypatch):
    """Force `guardrail.answer` to see a given retrieval result."""
    from app.rag import guardrail

    def install(result):
        monkeypatch.setattr(guardrail, "retrieve", lambda *a, **k: result)

    return install


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: loads the real embedding model and scores real text"
    )
