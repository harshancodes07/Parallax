"""Cache identical lessons so re-asking costs nothing.

Written after exhausting a Gemini free-tier quota (`limit: 20`) during testing.
One lesson costs about four model calls — query translation, English lesson,
grounding audit, regional composition — so roughly five lessons empties the
allowance, and the way an allowance actually gets emptied is asking the *same*
question twenty times while wiring up a demo.

A demo is the best possible case for caching: the same three pages and the same
handful of questions, over and over. Second and later asks are then instant and
free, which also makes the on-stage experience better than the first run.

Deliberately simple:
* in-memory, per process — dies with the server, which is correct for a cache
  holding generated content nobody promised to keep
* only successful grounded lessons are stored; refusals are cheap to recompute
  and a template fallback should get another try at the real thing
* bounded, oldest-out, so a long session cannot grow without limit
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from threading import Lock

from app.tutor.schemas import LessonSource, TutorLesson

log = logging.getLogger(__name__)

MAX_ENTRIES = int(os.getenv("TUTOR_CACHE_SIZE", "128"))
ENABLED = os.getenv("TUTOR_CACHE", "1") not in {"0", "false", "False"}

_store: "OrderedDict[tuple, TutorLesson]" = OrderedDict()
_lock = Lock()


def key(page_id: str, query: str | None, language: str, translate_to: list[str]) -> tuple:
    return (page_id, (query or "").strip().casefold(), language, tuple(sorted(translate_to or [])))


def get(cache_key: tuple) -> TutorLesson | None:
    if not ENABLED:
        return None
    with _lock:
        lesson = _store.get(cache_key)
        if lesson is not None:
            _store.move_to_end(cache_key)
            log.info("lesson cache hit: %s", cache_key[:2])
            # Copy so a caller stripping `trace` for one response does not
            # mutate what the next caller gets.
            return lesson.model_copy(deep=True)
    return None


def put(cache_key: tuple, lesson: TutorLesson) -> None:
    """Store only lessons worth reusing."""
    if not ENABLED:
        return
    if not lesson.grounded or lesson.source is LessonSource.TEMPLATE_FALLBACK:
        return
    with _lock:
        _store[cache_key] = lesson.model_copy(deep=True)
        _store.move_to_end(cache_key)
        while len(_store) > MAX_ENTRIES:
            _store.popitem(last=False)


def clear() -> None:
    with _lock:
        _store.clear()


def stats() -> dict[str, int | bool]:
    return {"enabled": ENABLED, "entries": len(_store), "max_entries": MAX_ENTRIES}
