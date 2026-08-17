"""AI4Bharat IndicConformer — speech → text, for the hands-free stretch goal.

`ai4bharat/indic-conformer-600m-multilingual`, 22 languages, CTC and RNNT
decoding. We default to CTC: it is faster and this is a short classroom question,
not dictation. RNNT is available per-call for a second opinion when CTC output
looks empty.

The model wants 16kHz mono. Anything the browser sends (webm/opus at 48k, usually)
gets resampled here rather than being the frontend's problem.

Install: `pip install torchaudio onnxruntime`
"""

from __future__ import annotations

import io
import logging
import os

from app.tutor.indic import languages
from app.tutor.indic.runtime import LazyComponent, device

log = logging.getLogger(__name__)

ASR_MODEL = os.getenv("INDIC_CONFORMER_MODEL", "ai4bharat/indic-conformer-600m-multilingual")
TARGET_SAMPLE_RATE = 16_000


class IndicConformer:
    def __init__(self, model_id: str) -> None:
        import torch
        import torchaudio
        from transformers import AutoModel

        self._torch = torch
        self._torchaudio = torchaudio
        self._device = device()
        self._model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model_id = model_id

    def transcribe(self, audio_bytes: bytes, language_code: str, decoding: str = "ctc") -> str:
        wav, sample_rate = self._torchaudio.load(io.BytesIO(audio_bytes))

        # Mono: the model takes one channel, and averaging beats picking channel 0
        # for phone recordings where one side can be near-silent.
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = self._torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
            )
            wav = resampler(wav)

        with self._torch.inference_mode():
            return str(self._model(wav, language_code, decoding)).strip()


ENGINE = LazyComponent("IndicConformer ASR", lambda: IndicConformer(ASR_MODEL), "TUTOR_ASR")


def transcribe(audio_bytes: bytes, language_code: str, decoding: str = "ctc") -> str | None:
    """Speech → text in the student's language. `None` if ASR is unavailable."""
    engine = ENGINE.get()
    if engine is None:
        return None

    lang = languages.get(language_code)
    try:
        text = engine.transcribe(audio_bytes, lang.asr, decoding)
    except Exception as exc:  # noqa: BLE001
        log.warning("ASR failed (%s, %s): %s", lang.asr, decoding, exc)
        return None

    # CTC occasionally returns nothing on a short or quiet clip; RNNT is the
    # better model on exactly those, so it earns one retry.
    if not text and decoding == "ctc":
        log.info("empty CTC transcript, retrying with RNNT")
        try:
            text = engine.transcribe(audio_bytes, lang.asr, "rnnt")
        except Exception as exc:  # noqa: BLE001
            log.warning("RNNT retry failed: %s", exc)
            return None

    return text or None


def status() -> dict[str, str]:
    return {"asr": "loaded" if ENGINE.available else (ENGINE.reason or "not loaded yet")}
