"""Jev decisions drive an external Qwen voice service after causal local VAD."""

import asyncio, base64, json, time, urllib.request
from .qwen import Episode as VoiceEpisode, MODEL as VOICE_MODEL
from .gemini import SYSTEM
from .online_vad import OnlineVAD

MODEL = "Jev-Omni controller + Qwen3-Omni external voice"
CONFIG = {
    "model": VOICE_MODEL,
    "instructions": SYSTEM,
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 16000},
            "turn_detection": None,
        }
    },
}


class Episode(VoiceEpisode):
    config = CONFIG
    model_name = MODEL
    decision_event = "jev_decision"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.vad = OnlineVAD()
        self.observed = bytearray()
        self.committed = 0
        self.endpoint_queue = asyncio.Queue()
        self.controller = None
        self.vad_generation = 0
        self.decisions = 0
        self.pending_decision = False

    def log(self, event, **fields):
        if event == "observation_end":
            fields.update(
                controller_pending=self.pending_decision,
                controller_queued=self.endpoint_queue.qsize(),
            )
        return super().log(event, **fields)

    async def run(self, url):
        result = await super().run(url)
        result["model"] = self.model_name
        end = next((r for r in self.trace if r["event"] == "observation_end"), {})
        pending = bool(end.get("controller_pending") or end.get("controller_queued"))
        result["metrics"]["controller_pending_at_deadline"] = pending
        result["metrics"]["observation_censored"] |= pending
        result["controller_decisions"] = [
            r
            for r in self.trace
            if r["event"]
            in (
                self.decision_event,
                "stale_decision_discarded",
                "external_voice_request",
                "controller_stop",
            )
        ]
        (self.out / "result.json").write_text(json.dumps(result, indent=2))
        return result

    async def receive(self, ws):
        self.controller = asyncio.create_task(self.control(ws))
        try:
            await super().receive(ws)
        finally:
            self.controller.cancel()
            await asyncio.gather(self.controller, return_exceptions=True)

    def check_background(self):
        super().check_background()
        if self.controller and self.controller.done():
            self.controller.result()
            raise RuntimeError("Controller ended")

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
            self.observed.extend(data)
            self.log(
                "input_chunk",
                source=source,
                media_start_s=offset / 32000,
                media_end_s=sent / 16000,
                lateness_s=max(0, time.monotonic() - due),
            )
            for event in self.vad.feed(data):
                self.log("local_vad", details=event)
                if "start" in event:
                    self.vad_generation += 1
                if "end" in event:
                    await self.endpoint_queue.put(
                        (len(self.observed), self.vad_generation)
                    )

    async def control(self, ws):
        while True:
            count, generation = await self.endpoint_queue.get()
            self.decisions += 1
            if count <= self.committed:
                continue
            speaking = bool(self.active or self.playback.buffered)
            payload = {
                "pcm": base64.b64encode(bytes(self.observed[:count])).decode(),
                "speaking": speaking,
                "key": self.case["id"] + ":" + str(self.decisions),
            }

            def predict():
                req = urllib.request.Request(
                    "http://127.0.0.1:8767/decision",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.load(response)

            self.log("decision_start", observed_samples=count // 2, speaking=speaking)
            self.pending_decision = True
            try:
                answer = await asyncio.to_thread(predict)
            finally:
                self.pending_decision = False
            if generation != self.vad_generation:
                self.log("stale_decision_discarded", decision=answer)
                continue
            self.log(self.decision_event, decision=answer, observed_samples=count // 2)
            action = answer["action"]
            if action in ("wait", "continue"):
                if action == "continue":
                    self.committed = count
                continue
            if action == "stop":
                # Immediate local stop, with provider cancellation sent concurrently to stop generation.
                self.log("controller_stop", discarded_buffer_s=self.playback.clear())
                if self.active:
                    await ws.send(json.dumps({"type": "response.cancel"}))
                deadline = time.monotonic() + 5
                while self.active and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                if self.active:
                    raise TimeoutError("Voice cancellation acknowledgment missing")
            audio = bytes(self.observed[self.committed : count])
            self.committed = count
            await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio).decode(),
                        "is_speech": True,
                    }
                )
            )
            self.log(
                "external_voice_request", audio_samples=len(audio) // 2, action=action
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.commit",
                        "final": True,
                        "create_response": True,
                        "is_speech": True,
                    }
                )
            )
