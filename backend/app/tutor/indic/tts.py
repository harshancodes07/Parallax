"""AI4Bharat Indic Parler-TTS — the lesson, read aloud in the student's language.

`ai4bharat/indic-parler-tts`, 21 languages. Parler is description-driven: a
plain-English sentence about the *voice* ("warm, slightly slow, like a teacher")
steers delivery, while the `prompt` is the text actually spoken. Those go through
two different tokenizers — the description through the text encoder's, the spoken
text through the model's own. Getting that backwards produces audio that says the
description out loud, which is a genuinely funny way to lose a demo.

This matters more here than in a generic TTS integration: the Tamil we generate is
already spoken-register teacher talk, so it lands naturally when read aloud.
Stiff written Tamil read by a TTS is what sounds robotic — the composition
decision and the voice quality are the same decision.

Install: `pip install git+https://github.com/huggingface/parler-tts.git soundfile`
"""

from __future__ import annotations

import io
import logging
import os

from app.tutor.indic import languages
from app.tutor.indic.runtime import LazyComponent, device

log = logging.getLogger(__name__)

TTS_MODEL = os.getenv("INDIC_PARLER_TTS_MODEL", "ai4bharat/indic-parler-tts")

# Long inputs make Parler drift. A lesson is a handful of sentences, so we chunk
# and concatenate rather than trying to synthesise one long paragraph.
MAX_CHARS_PER_CHUNK = 400


class IndicParlerTTS:
    def __init__(self, model_id: str) -> None:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._torch = torch
        self._device = device()
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        # The description is read by the text encoder, which has its own tokenizer.
        self._description_tokenizer = AutoTokenizer.from_pretrained(
            self._model.config.text_encoder._name_or_path
        )
        self.sampling_rate = self._model.config.sampling_rate
        self.model_id = model_id

    def synthesize(self, text: str, description: str):
        import numpy as np

        chunks = _chunk(text)
        pieces = []
        for chunk in chunks:
            description_ids = self._description_tokenizer(
                description, return_tensors="pt"
            ).to(self._device)
            prompt_ids = self._tokenizer(chunk, return_tensors="pt").to(self._device)

            with self._torch.inference_mode():
                generation = self._model.generate(
                    input_ids=description_ids.input_ids,
                    attention_mask=description_ids.attention_mask,
                    prompt_input_ids=prompt_ids.input_ids,
                    prompt_attention_mask=prompt_ids.attention_mask,
                )
            pieces.append(generation.cpu().numpy().squeeze())

        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]


ENGINE = LazyComponent("Indic Parler-TTS", lambda: IndicParlerTTS(TTS_MODEL), "TUTOR_TTS")


def speak(text: str, language_code: str, description: str | None = None) -> bytes | None:
    """Render `text` to WAV bytes. `None` if TTS is unavailable."""
    if not (text or "").strip():
        return None

    engine = ENGINE.get()
    if engine is None:
        return None

    lang = languages.get(language_code)
    try:
        import soundfile as sf

        audio = engine.synthesize(text, description or lang.tts_voice)
        buffer = io.BytesIO()
        sf.write(buffer, audio, engine.sampling_rate, format="WAV")
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        log.warning("TTS failed for %s: %s", lang.code, exc)
        return None


def _chunk(text: str) -> list[str]:
    from app.tutor.indic.runtime import split_sentences

    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text) or [text]:
        if current and len(current) + len(sentence) + 1 > MAX_CHARS_PER_CHUNK:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def status() -> dict[str, str]:
    return {"tts": "loaded" if ENGINE.available else (ENGINE.reason or "not loaded yet")}
