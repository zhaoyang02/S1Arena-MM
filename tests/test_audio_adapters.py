import base64
import io
import json
import wave

import pytest

from conftest import make_wav
from jevarena_mm import media, render, runner
from jevarena_mm.models import audio_common, audio_gemini
from jevarena_mm.models.audio_gemini import AudioGemini
from jevarena_mm.models.audio_qwen import AudioQwen, REVISION
from jevarena_mm.registry import make_model
from jevarena_mm.schema import DecisionRequest, ModelResponse, Question


@pytest.fixture
def request_audio(workspace):
    make_wav(workspace / "media" / "private_source" / "secret-label.wav", seconds=2, rate=8000)
    ref = media.make_ref("audio", "private_source/secret-label.wav", window=(0.25, 1.25))
    return DecisionRequest(state={"clip": ref, "text": "Listen to the excerpt."},
                           questions={"q": Question(type="boolean", instructions="Is speech audible?")})


def test_shared_decoder_window_and_model_prompt(request_audio):
    model = AudioGemini("example@sample_rate=16000")
    body = model.payload(request_audio)
    parts = body["contents"][0]["parts"]
    data = base64.b64decode(parts[0]["inlineData"]["data"])
    with wave.open(io.BytesIO(data)) as stream:
        assert stream.getframerate() == 16000 and stream.getnchannels() == 1
        assert stream.getnframes() == 16000  # exactly the selected one-second window
    assert parts[1]["text"] == render.messages(request_audio)[1]["content"]
    for private in ("secret-label", "private_source", request_audio.state["clip"].sha256):
        assert private not in parts[1]["text"]
    conversation = AudioQwen.conversation(request_audio)
    assert conversation[1]["content"][1]["text"] == parts[1]["text"]


def test_audio_reference_and_integrity_errors(request_audio):
    bad = request_audio.model_copy(deep=True)
    bad.state["clip"] = bad.state["clip"].model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="integrity"):
        audio_common.waveform(bad)
    bad.state["clip"] = request_audio.state["clip"].model_copy(update={"window": (1.0, 4.0)})
    with pytest.raises(ValueError, match="ends before"):
        audio_common.waveform(bad)
    bad.state["second"] = request_audio.state["clip"]
    with pytest.raises(ValueError, match="exactly one"):
        audio_common.waveform(bad)


def test_capabilities_and_sampling_policy_cache_identity(request_audio):
    assert make_model("audio_gemini:example").supports(request_audio)
    assert make_model("audio_qwen").supports(request_audio)
    assert AudioGemini("example@sample_rate=8000").slug != AudioGemini("example").slug
    assert "decoder=pyav" in AudioQwen().model_id and REVISION in AudioQwen().model_id
    with pytest.raises(ValueError, match="unknown"):
        AudioGemini("example@truncate=true")


@pytest.mark.parametrize("reply", ['{}', '{"q": {"true": -1}}', '{"q": {"true": 1e999}}',
                                  '{"q": {"wrong": 1}}', '{"q": {"true": 0, "false": 0}}'])
def test_bad_probabilities_are_failed_not_successful(request_audio, reply):
    result = audio_common.reparse(ModelResponse(raw_text=reply), request_audio)
    assert result.status == "failed" and result.error and not result.answers
    assert runner.is_final({"response": result.model_dump()})  # bad replies are scored, not resampled
    repaired = audio_common.reparse(result.model_copy(update={"raw_text": '{"q":{"true":0.8,"false":0.2}}'}), request_audio)
    assert repaired.status == "ok" and repaired.error is None


def test_gemini_transport_and_usage_without_network(request_audio, monkeypatch):
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-key")
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"q":{"true":1}}'}]}}],
                                         "usageMetadata": {"promptTokenCount": 7}})

    monkeypatch.setattr(httpx, "post", post)
    result = AudioGemini("example").decide(request_audio)
    assert len(calls) == 1 and result.status == "ok"
    assert result.usage["input"]["audio"]["seconds"] == 1
    assert result.usage["promptTokenCount"] == 7
    assert result.cost_usd is None
    assert "secret-label" not in json.dumps(calls[0][1]["json"])


def test_file_upload_cleanup_after_provider_error(request_audio, monkeypatch):
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-key")
    model = AudioGemini("example")
    inline_size = len(json.dumps(model.payload(request_audio), ensure_ascii=False).encode())
    monkeypatch.setattr(audio_gemini, "MAX_INLINE_BYTES", inline_size - 1)
    deleted = []

    def post(url, **kwargs):
        if url.endswith("/upload/v1beta/files"):
            return httpx.Response(200, headers={"x-goog-upload-url": "https://generativelanguage.googleapis.com/upload-session"})
        if url.endswith("/upload-session"):
            return httpx.Response(200, json={"file": {"name": "files/fixture", "state": "ACTIVE", "uri": "test-file"}})
        return httpx.Response(429, text="private error body")

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "delete", lambda url, **kwargs: deleted.append(url) or httpx.Response(204))
    result = model.decide(request_audio)
    assert result.status == "failed" and result.error == "transport: HTTP 429"
    assert len(deleted) == 1 and deleted[0].endswith("files/fixture")
    assert not runner.is_final({"response": result.model_dump()})


def test_qwen_uses_shared_decoder_without_loading_gpu(request_audio, monkeypatch):
    model = AudioQwen()
    monkeypatch.setattr(model, "_load", lambda: None)
    captured = []

    def generate(request, audio, sample):
        captured.append((len(audio), sample))
        return '{"q":{"true":0.6,"false":0.4}}', {"audio_frames": 100}

    monkeypatch.setattr(model, "_generate", generate)
    result = model.decide(request_audio, sample=3)
    assert result.status == "ok" and captured == [(16000, 3)]
    assert result.usage["input"]["audio"]["decoder"] == "pyav"
