"""Native Gemini Live PCM streaming; private traces; clocked virtual playback.

Run as a script. Optional dependencies are isolated from ordinary jevarena_mm imports.
No transcripts, gold events, category names or future audio are sent to the model.
"""

import asyncio
import hashlib
import json
import time
import wave

from .core import Playback, summarize

SYSTEM = (
    "You are a helpful voice assistant having a conversation. Listen to the user and "
    "respond naturally in English. Respect pauses inside an unfinished turn. "
    "If the user changes their request, respond to the new request. Brief acknowledgments "
    "such as yes or uh-huh do not necessarily mean they want you to abandon your explanation. "
    "For explanation or advice requests, provide several useful points."
)
CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": SYSTEM,
    "output_audio_transcription": {},
    "realtime_input_config": {"automatic_activity_detection": {"disabled": False}},
}


def pcm_file(root, spec):
    path = (root / spec["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Audio path escapes manifest directory")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("Audio hash mismatch")
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected PCM16 mono 16kHz")
        pcm = w.readframes(w.getnframes())
        if len(pcm) != w.getnframes() * 2:
            raise ValueError("Truncated WAV")
    return pcm


class Episode:
    def __init__(self, case, root, out, chunk_ms=100):
        self.case = case
        self.root = root
        self.out = out
        self.chunk_frames = 16000 * chunk_ms // 1000
        self.playback = Playback()
        self.trace = []
        self.received = []
        self.played = []
        self.start = 0
        self.receiver = None
        self.sink = None
        self.event_file = None

    def log(self, event, **fields):
        item = {"event": event, "t": time.monotonic() - self.start, **fields}
        self.trace.append(item)
        if self.event_file:
            self.event_file.write(json.dumps(item) + "\n")
            self.event_file.flush()
        return item

    async def receive(self, session):
        while True:
            async for response in session.receive():
                content = response.server_content
                if response.usage_metadata:
                    self.log(
                        "usage",
                        usage=response.usage_metadata.model_dump(
                            mode="json", exclude_none=True
                        ),
                    )
                if not content:
                    continue
                if content.interrupted:
                    discarded = self.playback.clear()
                    self.log("interrupted", discarded_buffer_s=discarded)
                if content.model_turn:
                    for part in content.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            mime = part.inline_data.mime_type or ""
                            if (
                                not mime.startswith("audio/pcm")
                                or "rate=24000" not in mime
                            ):
                                raise ValueError(
                                    "Unexpected output PCM format: " + mime
                                )
                            audio = part.inline_data.data
                            self.received.append(audio)
                            self.playback.add(audio)
                            self.log(
                                "output_chunk",
                                frames=len(audio) // 2,
                                buffered_s=self.playback.buffered / 48000,
                            )
                if content.output_transcription:
                    self.log(
                        "output_transcription", text=content.output_transcription.text
                    )
                if content.turn_complete:
                    self.log("turn_complete")
                if content.generation_complete:
                    self.log("generation_complete")

    async def consume(self):
        # Never drain the queue faster than physical playback. Delays slow the sink
        # and are logged, rather than silently consuming a burst after a stall.
        while True:
            tick = time.monotonic()
            await asyncio.sleep(0.020)
            now = time.monotonic()
            pcm = self.playback.tick(now - self.start)
            if pcm:
                self.played.append((now - self.start, pcm))
            if now - tick > 0.050:
                self.log("sink_stall", elapsed_s=now - tick)

    def check_background(self):
        for task in (self.receiver, self.sink):
            if task and task.done():
                task.result()
                raise RuntimeError("Background stream unexpectedly ended")

    async def send_pcm(self, session, pcm, source):
        from google.genai import types

        sent = 0
        origin = time.monotonic()
        size = self.chunk_frames * 2
        for offset in range(0, len(pcm), size):
            data = pcm[offset : offset + size]
            sent += len(data) // 2
            # Expose a chunk only when its final sample's time has elapsed.
            due = origin + sent / 16000
            await asyncio.sleep(max(0, due - time.monotonic()))
            self.check_background()
            await session.send_realtime_input(
                audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
            )
            self.log(
                "input_chunk",
                source=source,
                media_start_s=offset / 32000,
                media_end_s=sent / 16000,
                lateness_s=max(0, time.monotonic() - due),
            )

    async def silence(self, session, seconds):
        await self.send_pcm(session, b"\0\0" * round(16000 * seconds), "silence")

    async def drive(self, session):
        await self.send_pcm(session, pcm_file(self.root, self.case["audio"]), "input")
        if self.case["mode"] == "stream":
            await self.silence(session, 5)
            return
        # Feedback edge: the second user utterance is triggered by measured model
        # playback, NOT the source dataset's fixed event timestamp or a timer alone.
        deadline = time.monotonic() + 20
        threshold = self.case["trigger_after_audible_s"]
        while time.monotonic() < deadline:
            now = time.monotonic() - self.start
            audible = (
                self.playback.last_audible is not None
                and now - self.playback.last_audible < 0.08
            )
            if (
                self.playback.audible_frames / 24000 >= threshold
                and audible
                and self.playback.buffered
            ):
                self.log(
                    "reactive_user_event",
                    assistant_audible=True,
                    played_audible_s=self.playback.audible_frames / 24000,
                )
                await self.send_pcm(
                    session, pcm_file(self.root, self.case["user_event"]), "user_event"
                )
                self.log("reactive_user_event_end")
                await self.silence(session, 8)
                return
            await self.silence(session, 0.1)
        self.log(
            "trigger_timeout", reason="No sufficiently long active assistant playback"
        )

    async def run(self, client, model):
        self.out.mkdir(parents=True, exist_ok=False)
        self.start = time.monotonic()
        status = "completed"
        failure = None
        with (self.out / "events.jsonl").open("w") as self.event_file:
            try:
                async with asyncio.timeout(110):
                    async with client.aio.live.connect(
                        model=model, config=CONFIG
                    ) as session:
                        self.start = time.monotonic()
                        self.log("connected", model=model)
                        self.receiver = asyncio.create_task(self.receive(session))
                        self.sink = asyncio.create_task(self.consume())
                        try:
                            await self.drive(session)
                            self.check_background()
                            self.log(
                                "observation_end",
                                buffered_s=self.playback.buffered / 48000,
                            )
                        finally:
                            for task in (self.receiver, self.sink):
                                task.cancel()
                            await asyncio.gather(
                                self.receiver, self.sink, return_exceptions=True
                            )
            except Exception as error:
                status = "failed"
                failure = type(error).__name__
                # Exception messages can contain authenticated URLs. Keep them out
                # of logs; aggregate only exception class and safe provider code.
                self.log("error", error_type=failure, code=getattr(error, "code", None))
        self.event_file = None
        for name, data in [("received.wav", b"".join(self.received))]:
            with wave.open(str(self.out / name), "wb") as w:
                w.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                w.writeframes(data)
        # Preserve pauses and cancelled playback in a wall-clock-aligned output.
        timeline = bytearray(round((time.monotonic() - self.start) * 24000) * 2)
        for end, pcm in self.played:
            at = max(0, round(end * 24000) - len(pcm) // 2) * 2
            timeline[at : at + len(pcm)] = pcm
        with wave.open(str(self.out / "playback_timeline.wav"), "wb") as w:
            w.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            w.writeframes(timeline)
        metrics = summarize(self.case, self.trace, self.playback.intervals)
        result = {
            "id": self.case["id"],
            "category": self.case["category"],
            "mode": self.case["mode"],
            "status": status,
            "error_type": failure,
            "model": model,
            "metrics": metrics,
            "playback": "clocked virtual sink; no physical speaker/microphone",
            "playback_intervals": self.playback.intervals,
        }
        (self.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
