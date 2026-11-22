"""Scale selection, isolated controller policy, and deadline accounting."""

import asyncio
import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from conftest import make_wav
from jevarena_mm.streaming import jev_controller, qwen
from jevarena_mm.streaming.core import summarize
from jevarena_mm.streaming.prepare import prepare


def test_all_eligible_with_unequal_task_sizes(tmp_path):
    pytest.importorskip("soundfile")
    pytest.importorskip("scipy")  # prepare resamples with it; the .[audio-stream] environment has it
    wav = make_wav(tmp_path / "sample.wav", seconds=0.1, rate=16000).read_bytes()
    categories = [
        "candor_turn_taking",
        "candor_pause_handling",
        "user_interruption",
        "user_backchannel",
    ]
    source = tmp_path / "source"
    source.mkdir()
    for n, category in enumerate(categories, 1):
        with zipfile.ZipFile(source / f"{category}.zip", "w") as z:
            for i in range(n):
                folder = f"{category}/{i}"
                for name in [
                    "input.wav",
                    "context.wav",
                    "interrupt.wav",
                    "backchannel.wav",
                ]:
                    z.writestr(f"{folder}/{name}", wav)
                for name in ["turn_taking.json", "pause.json"]:
                    z.writestr(
                        f"{folder}/{name}", json.dumps([{"timestamp": [0.01, 0.05]}])
                    )
            if category == "candor_turn_taking":
                z.writestr(f"{category}/99/input.wav", wav)
                z.writestr(
                    f"{category}/99/turn_taking.json", '[{"timestamp":[0.05,0.05]}]'
                )
    manifest = prepare(source, tmp_path / "all", per_task=0)
    assert len(manifest["cases"]) == 10
    assert manifest["selection"]["skipped"] == [
        {"source_id": "candor_turn_taking/99", "reason": "invalid_held_out_event"}
    ]
    assert len(prepare(source, tmp_path / "one", per_task=1)["cases"]) == 4
    with pytest.raises(ValueError, match="Insufficient"):
        prepare(source, tmp_path / "insufficient", per_task=2)


def test_candor_sets_as_folders(tmp_path):
    """BetterUp keeps the CANDOR sets as folders; they read like the ZIPs Full-Duplex-Bench used to ship."""
    pytest.importorskip("soundfile")
    pytest.importorskip("scipy")  # prepare resamples with it; the .[audio-stream] environment has it
    wav = make_wav(tmp_path / "sample.wav", seconds=0.1, rate=16000).read_bytes()
    source = tmp_path / "source"
    for category, label in [("candor_turn_taking", "turn_taking.json"), ("candor_pause_handling", "pause.json")]:
        folder = source / category / "1"
        folder.mkdir(parents=True)
        (folder / "input.wav").write_bytes(wav)
        (folder / label).write_text(json.dumps([{"timestamp": [0.01, 0.05]}]))
    for category in ["user_interruption", "user_backchannel"]:
        with zipfile.ZipFile(source / f"{category}.zip", "w") as z:
            for name in ["context.wav", "interrupt.wav", "backchannel.wav"]:
                z.writestr(f"{category}/1/{name}", wav)
            z.writestr(f"{category}/1/input.wav", wav)
    manifest = prepare(source, tmp_path / "out", per_task=1)
    assert len(manifest["cases"]) == 4
    assert set(manifest["archives_sha256"]) == {"candor_turn_taking/", "candor_pause_handling/",
                                               "user_interruption.zip", "user_backchannel.zip"}
    assert {c["license"] for c in manifest["cases"] if c["mode"] == "stream"} == {"CANDOR Corpus License v2.0 (BetterUp)"}


@pytest.mark.parametrize("action", ["stop", "continue", "wait", "speak"])
def test_controller_commands_and_config_isolation(tmp_path, monkeypatch, action):
    monkeypatch.setattr(jev_controller, "OnlineVAD", lambda: None)
    assert (
        qwen.Episode.config["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    )
    assert jev_controller.Episode.config["audio"]["input"]["turn_detection"] is None

    async def check():
        e = jev_controller.Episode({"id": "case"}, Path(tmp_path), Path(tmp_path))
        e.observed.extend(bytes(3200))
        if action in ("stop", "continue"):
            e.active = {"old"}
            e.playback.add(bytes(960))
        sent = []

        class WS:
            async def send(self, raw):
                event = json.loads(raw)
                sent.append(event["type"])
                if event["type"] == "response.cancel":
                    e.active.clear()

        monkeypatch.setattr(
            jev_controller.asyncio,
            "to_thread",
            AsyncMock(return_value={"action": action}),
        )
        await e.endpoint_queue.put((3200, 0))
        task = asyncio.create_task(e.control(WS()))
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert ("response.cancel" in sent) == (action == "stop")
        assert ("input_audio_buffer.commit" in sent) == (action in ("stop", "speak"))
        if action == "stop":
            assert not e.playback.buffered
        if action == "continue":
            assert e.playback.buffered

    asyncio.run(check())


def test_pending_controller_is_censored_at_observation_end():
    case = {"mode": "stream", "gold_events": []}
    trace = [
        {
            "event": "observation_end",
            "t": 1,
            "buffered_s": 0,
            "controller_pending": True,
        }
    ]
    metrics = summarize(case, trace, [])
    assert metrics["controller_pending_at_deadline"]
    assert metrics["observation_censored"]


def test_new_speech_discards_stale_controller_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(jev_controller, "OnlineVAD", lambda: None)

    async def check():
        e = jev_controller.Episode({"id": "case"}, tmp_path, tmp_path)
        e.observed.extend(bytes(3200))
        sent = []

        class WS:
            async def send(self, raw):
                sent.append(raw)

        async def decision(_):
            e.vad_generation += 1
            return {"action": "speak"}

        monkeypatch.setattr(jev_controller.asyncio, "to_thread", decision)
        await e.endpoint_queue.put((3200, 0))
        task = asyncio.create_task(e.control(WS()))
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert not sent
        assert any(row["event"] == "stale_decision_discarded" for row in e.trace)

    asyncio.run(check())
