"""AI4Bharat IndicTrans2 — all three directions this branch needs.

| Engine                              | Model                                        | Used for |
|-------------------------------------|----------------------------------------------|----------|
| Indic → English                     | `indictrans2-indic-en-dist-200M`             | **Validation.** Backtranslate the composed regional explanation and check the concepts survived. |
| English → Indic                     | `indictrans2-en-indic-dist-200M`             | Fallback when composition fails, and reaching languages the lesson model is weaker in. |
| Indic → Indic                       | `indictrans2-indic-indic-dist-320M`          | Tamil → the other four **without** going back through English, so the teacher voice survives. |

Why Indic→Indic matters: the composed Tamil is warm, spoken, Tanglish-flavoured
teacher talk. The English `simple_explanation` is flat and factual. If you want a
Telugu student to get the *teacher*, translate from the Tamil, not the English —
which is exactly the direct pair this 320M stitched model exists for.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.tutor.indic import languages
from app.tutor.indic.runtime import LazyComponent, device, split_sentences

log = logging.getLogger(__name__)

INDIC_EN_MODEL = os.getenv("INDICTRANS2_INDIC_EN", "ai4bharat/indictrans2-indic-en-dist-200M")
EN_INDIC_MODEL = os.getenv("INDICTRANS2_EN_INDIC", "ai4bharat/indictrans2-en-indic-dist-200M")
INDIC_INDIC_MODEL = os.getenv(
    "INDICTRANS2_INDIC_INDIC", "ai4bharat/indictrans2-indic-indic-dist-320M"
)

MAX_LENGTH = 512
NUM_BEAMS = 5


class IndicTrans2:
    """One loaded IndicTrans2 checkpoint. Direction is fixed by the checkpoint."""

    def __init__(self, model_id: str) -> None:
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self._device = device()
        self._processor = IndicProcessor(inference=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device.startswith("cuda") else torch.float32,
        ).to(self._device)
        self._model.eval()
        self.model_id = model_id

    def translate(self, text: str, src_flores: str, tgt_flores: str) -> str:
        sentences = split_sentences(text) or [text]
        batch = self._processor.preprocess_batch(
            sentences, src_lang=src_flores, tgt_lang=tgt_flores
        )
        inputs = self._tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=MAX_LENGTH,
        ).to(self._device)

        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                max_length=MAX_LENGTH,
                min_length=0,
                num_return_sequences=1,
            )

        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        return " ".join(self._processor.postprocess_batch(decoded, lang=tgt_flores))


INDIC_EN = LazyComponent(
    "IndicTrans2 indic→en", lambda: IndicTrans2(INDIC_EN_MODEL), "TUTOR_INDIC_EN"
)
EN_INDIC = LazyComponent(
    "IndicTrans2 en→indic", lambda: IndicTrans2(EN_INDIC_MODEL), "TUTOR_EN_INDIC"
)
INDIC_INDIC = LazyComponent(
    "IndicTrans2 indic→indic", lambda: IndicTrans2(INDIC_INDIC_MODEL), "TUTOR_INDIC_INDIC"
)


def _engine_for(src_flores: str, tgt_flores: str) -> LazyComponent[Any]:
    if src_flores == languages.ENGLISH_FLORES:
        return EN_INDIC
    if tgt_flores == languages.ENGLISH_FLORES:
        return INDIC_EN
    return INDIC_INDIC


def translate(text: str, *, src_flores: str, tgt_flores: str) -> str | None:
    """Translate, picking the right checkpoint. `None` if the model is unavailable.

    Callers must handle `None` — that is the whole degradation contract.
    """
    if not (text or "").strip() or src_flores == tgt_flores:
        return text

    engine_holder = _engine_for(src_flores, tgt_flores)
    engine = engine_holder.get()
    if engine is None:
        return None

    try:
        return engine.translate(text, src_flores, tgt_flores)
    except Exception as exc:  # noqa: BLE001
        log.warning("translation %s→%s failed: %s", src_flores, tgt_flores, exc)
        return None


def to_english(text: str, language_code: str) -> str | None:
    """Backtranslation for validation."""
    lang = languages.get(language_code)
    return translate(text, src_flores=lang.flores, tgt_flores=languages.ENGLISH_FLORES)


def from_english(text: str, language_code: str) -> str | None:
    """English → regional. The fallback path, not the primary one."""
    lang = languages.get(language_code)
    return translate(text, src_flores=languages.ENGLISH_FLORES, tgt_flores=lang.flores)


def between_indic(text: str, *, source_code: str, target_code: str) -> str | None:
    """Regional → regional, preserving the composed teacher voice."""
    src = languages.get(source_code)
    tgt = languages.get(target_code)
    return translate(text, src_flores=src.flores, tgt_flores=tgt.flores)


def status() -> dict[str, str]:
    """What actually loaded — surfaced on /api/tutor/capabilities for the demo."""
    return {
        "indic_en": _describe(INDIC_EN),
        "en_indic": _describe(EN_INDIC),
        "indic_indic": _describe(INDIC_INDIC),
    }


def _describe(holder: LazyComponent[Any]) -> str:
    if holder.available:
        return "loaded"
    return holder.reason or "not loaded yet"
