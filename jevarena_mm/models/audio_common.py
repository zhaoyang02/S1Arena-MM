"""Shared audio adapter validation and probability parsing; no provider dependencies on discovery."""

import io
import math
import wave

import numpy as np

from .. import media, render
from ..schema import Answer, ModelResponse, normalize


def audio_ref(request):
    refs = request.media()
    if request.modalities() != {"audio"} or len(refs) != 1:
        raise ValueError("this adapter requires exactly one audio reference (optional text is allowed)")
    return refs[0][1]


def waveform(request, sample_rate=16000):
    ref = audio_ref(request)
    reason = media.check(ref)
    if reason:
        raise ValueError(f"audio integrity check failed: {reason}")
    if media.path_of(ref).stat().st_size > 256_000_000:
        raise ValueError("audio exceeds the 256 MB local input limit; no truncation performed")
    samples, rate = media.audio(ref, sample_rate=sample_rate, mono=True)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("decoded audio must be nonempty and finite")
    if ref.window and len(samples) < round((ref.window[1] - ref.window[0]) * rate) - 2:
        raise ValueError("audio ends before the requested window; no partial input sent")
    return samples, {"audio": {"sample_rate": rate, "channels": 1, "samples": len(samples),
                                "seconds": len(samples) / rate, "window": ref.window,
                                "decoder": "pyav", "chunking": "none"}}


def wav_bytes(samples, rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def reparse(response: ModelResponse, request) -> ModelResponse:
    try:
        parsed = render.parse(response.raw_text or "", request)
        answers = {}
        for name, question in request.questions.items():
            probs = parsed.get(name, {})
            if not probs or set(probs) - set(question.option_ids()):
                raise ValueError(f"missing or invalid options: {name}")
            total = sum(probs.values())
            if any(not math.isfinite(v) or v < 0 for v in probs.values()) or not math.isfinite(total) or total <= 0:
                raise ValueError(f"invalid probability distribution: {name}")
            answers[name] = Answer(probs=normalize(probs, question.option_ids()))
        return response.model_copy(update={"answers": answers, "error": None, "status": "ok"})
    except (ValueError, TypeError) as error:
        # model_copy bypasses validation: set failed explicitly so the shared scorer records a failure.
        return response.model_copy(update={"answers": {}, "error": f"parse: {error}", "status": "failed"})
