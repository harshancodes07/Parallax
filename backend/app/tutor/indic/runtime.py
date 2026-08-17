"""Lazy loading for every AI4Bharat model in this branch.

Rules that apply to all of them, enforced here so no individual module gets to
forget one:

* **Nothing loads at import time.** These are gigabytes of weights; the API must
  start in a second and the tests must run with nothing installed.
* **A failure to load is never fatal.** Every caller gets `None` and degrades.
  The failure is logged exactly once, not once per request.
* **Each model has an env kill switch**, so a demo laptop can turn off whatever
  it doesn't have RAM for.
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Callable, Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

_FALSEY = {"0", "false", "False", "no", "off"}


def enabled(env_var: str, default: str = "1") -> bool:
    return os.getenv(env_var, default) not in _FALSEY


def device() -> str:
    """`cuda` when available, else `cpu`. Safe to call without torch installed."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class LazyComponent(Generic[T]):
    """A model loaded on first use, at most once, never fatally."""

    def __init__(self, name: str, loader: Callable[[], T], env_var: str | None = None) -> None:
        self.name = name
        self._loader = loader
        self._env_var = env_var
        self._value: T | None = None
        self._failed = False
        self._reason = ""
        self._lock = Lock()

    @property
    def available(self) -> bool:
        """True only if it is already loaded. Does not trigger a load."""
        return self._value is not None

    @property
    def reason(self) -> str:
        """Why it is unavailable — surfaced in the trace so a silent skip is visible."""
        return self._reason

    def get(self) -> T | None:
        if self._env_var and not enabled(self._env_var):
            self._reason = f"{self.name} disabled via {self._env_var}=0"
            return None
        if self._value is not None:
            return self._value
        if self._failed:
            return None

        with self._lock:
            if self._value is None and not self._failed:
                try:
                    log.info("loading %s (first use — this can take a while)", self.name)
                    self._value = self._loader()
                    log.info("%s ready", self.name)
                except Exception as exc:  # noqa: BLE001 - optional dependency, never fatal
                    self._failed = True
                    self._reason = f"{self.name} unavailable: {exc}"
                    log.warning("%s", self._reason)
        return self._value

    def reset(self) -> None:
        """Drop the loaded model and clear the failure memo. Tests and hot reload."""
        with self._lock:
            self._value = None
            self._failed = False
            self._reason = ""


def split_sentences(text: str) -> list[str]:
    """IndicTrans2 works sentence by sentence; it is not a document translator.

    Handles the Devanagari danda and the Latin terminators.
    """
    import re

    parts = re.split(r"(?<=[.!?।॥])\s+", (text or "").replace("\n", " ").strip())
    return [p.strip() for p in parts if p.strip()]
