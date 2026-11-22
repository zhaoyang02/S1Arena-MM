"""Qwen runtime-duplex evaluation on a frozen audio-live-v1 manifest."""

import asyncio
import base64
import contextlib
import json
import time
import wave
from .gemini import Episode as BaseEpisode, SYSTEM
from .core import summarize

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CONFIG = {
    "model": MODEL,
    "instructions": SYSTEM,
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 16000},
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": True,
                "interrupt_response": True,
            },
        }
    },
}


class Episode(BaseEpisode):
    config = CONFIG
    model_name = MODEL
    implementation_level = "turn_based_duplex"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rid = None
        self.played_response = 0
        self.capabilities = None
        self.closed = asyncio.Event()
        self.ready = asyncio.Event()
        self.active = set()

    async def receive(self, ws):
        async for raw in ws:
            e = json.loads(raw)
            typ = e["type"]
            self.log(
                "server_event",
                data={
                    k: v
                    for k, v in e.items()
                    if k not in ("delta", "audio", "resume_token")
                },
            )
            if typ == "error":
                raise RuntimeError(json.dumps(e))
            if typ == "session.created":
                self.capabilities = e["session"]["capabilities"]
                self.ready.set()
            if typ == "session.closed":
                self.closed.set()
                return
            if typ == "response.created":
                self.rid = e["response"]["id"]
                self.active.add(self.rid)
                self.played_response = 0
                if self.playback.buffered:
                    self.log(
                        "buffer_clear_on_new_response",
                        discarded_buffer_s=self.playback.clear(),
                    )
            if typ == "response.output_audio.delta":
                if e.get("sample_rate_hz", 24000) != 24000:
                    raise ValueError("Unexpected PCM rate")
                audio = base64.b64decode(e.get("delta") or e.get("audio") or "")
                self.received.append(audio)
                self.playback.add(audio)
                self.log(
                    "output_chunk",
                    frames=len(audio) // 2,
                    response_id=e.get("response_id"),
                    buffered_s=self.playback.buffered / 48000,
                )
            if typ == "response.done":
                self.active.discard(e["response"]["id"])
                if e["response"]["status"] in ("cancelled", "canceled"):
                    self.log(
                        "interrupted",
                        discarded_buffer_s=self.playback.clear(),
                        response_id=e["response"]["id"],
                    )

    async def consume(self):
        while True:
            tick = time.monotonic()
            await asyncio.sleep(0.02)
            now = time.monotonic()
            pcm = self.playback.tick(now - self.start)
            if pcm:
                self.played.append((now - self.start, pcm))
                self.played_response += len(pcm) / 48000
                if self.rid:
                    await self.ws.send(
                        json.dumps(
                            {
                                "type": "playback.ack",
                                "response_id": self.rid,
                                "played_ms": int(self.played_response * 1000),
                            }
                        )
                    )
            if now - tick > 0.05:
                self.log("sink_stall", elapsed_s=now - tick)

    async def send_pcm(self, ws, pcm, source):
        sent = 0
        origin = time.monotonic()
        size = self.chunk_frames * 2
        for offset in range(0, len(pcm), size):
            data = pcm[offset : offset + size]
            sent += len(data) // 2
            due = origin + sent / 16000
            await asyncio.sleep(max(0, due - time.monotonic()))
            self.check_background()
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode(),
                    }
                )
            )
            self.log(
                "input_chunk",
                source=source,
                media_start_s=offset / 32000,
                media_end_s=sent / 16000,
                lateness_s=max(0, time.monotonic() - due),
            )

    async def run(self, url):
        import websockets

        self.out.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        status = "completed"
        failure = None
        with (self.out / "events.jsonl").open("w") as self.event_file:
            try:
                async with asyncio.timeout(110):
                    async with websockets.connect(url, max_size=64 * 1024**2) as ws:
                        self.ws = ws
                        self.start = time.monotonic()
                        self.log("connected", model=self.model_name)
                        await ws.send(
                            json.dumps({"type": "session.update", "session": self.config})
                        )
                        self.receiver = asyncio.create_task(self.receive(ws))
                        self.sink = asyncio.create_task(self.consume())
                        try:
                            await asyncio.wait_for(self.ready.wait(), 10)
                            if (
                                not self.capabilities
                                or self.capabilities.get("implementation_level")
                                != self.implementation_level
                                or not self.capabilities.get(
                                    "supports_independent_io_streams"
                                )
                                or not self.capabilities.get("supports_playback_ack")
                            ):
                                raise RuntimeError("Unexpected capabilities")
                            await self.drive(ws)
                            self.check_background()
                            self.log(
                                "observation_end",
                                buffered_s=self.playback.buffered / 48000,
                                active_responses=len(self.active),
                            )
                        finally:
                            with contextlib.suppress(Exception):
                                await ws.send(json.dumps({"type": "session.close"}))
                                await asyncio.wait_for(self.closed.wait(), 5)
                            for task in (self.receiver, self.sink):
                                task.cancel()
                            await asyncio.gather(
                                self.receiver, self.sink, return_exceptions=True
                            )
            except Exception as exc:
                status = "failed"
                failure = type(exc).__name__
                self.log("error", error_type=failure)
                (self.out / "failure-private.txt").write_text(str(exc))
        self.event_file = None
        with wave.open(str(self.out / "received.wav"), "wb") as w:
            w.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            w.writeframes(b"".join(self.received))
        metrics = summarize(self.case, self.trace, self.playback.intervals)
        end = next((e for e in self.trace if e["event"] == "observation_end"), {})
        metrics["active_generation_at_deadline"] = end.get("active_responses", 0) > 0
        metrics["observation_censored"] = (
            metrics["observation_censored"] or metrics["active_generation_at_deadline"]
        )
        result = {
            "id": self.case["id"],
            "category": self.case["category"],
            "mode": self.case["mode"],
            "status": status,
            "error_type": failure,
            "model": self.model_name,
            "metrics": metrics,
            "capabilities": self.capabilities,
            "playback": "clocked virtual sink; no physical device",
            "playback_intervals": self.playback.intervals,
        }
        (self.out / "result.json").write_text(json.dumps(result, indent=2))
        return result
