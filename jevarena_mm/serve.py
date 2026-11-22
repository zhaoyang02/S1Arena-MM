"""A local decision service: one model, answering classified actions for the closed-loop suite bridges.

The bridges (jevarena_mm/bridges/) run inside each suite's own Python environment and post one packet per step:

    {"protocol": "jevarena-mm-decision-v1",
     "state": "<suite text; <image:k> marks where the k-th image sits>",
     "question": "Which action ...?",
     "candidates": [{"id": "a000", "text": "move left"}, ...],
     "images": [{"path": "/abs/step-3.png", "sha256": "..."}],
     "context": {...}}            # optional, echoed into the call log only

The service turns it into a DecisionRequest ({"text": ..., "images": [...]} and one choice question), calls
model.decide() and answers {"status": "ok" | "failed" | "unsupported", "selected": <candidate id>, "probabilities",
"latency_ms", "usage", "error", "model_id"}. Ties go to the earlier candidate. Calls are served one at a time, so
several bridges can share one model.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import media
from .schema import DecisionRequest, ModelResponse, Question

PROTOCOL = "jevarena-mm-decision-v1"


def request_of(packet: dict[str, Any]) -> DecisionRequest:
    """The DecisionRequest for one bridge packet (images are loaded and held in memory)."""
    if packet.get("protocol") != PROTOCOL:
        raise ValueError(f"unknown protocol {packet.get('protocol')!r}")
    images = []
    for image in packet.get("images") or []:
        path = Path(image["path"])
        data = path.read_bytes()  # hashed and decoded from the same bytes (screenshots are small; no hash cache)
        if hashlib.sha256(data).hexdigest() != image["sha256"]:
            raise ValueError(f"image changed after the bridge hashed it: {path}")
        from PIL import Image

        with Image.open(io.BytesIO(data)) as picture:
            images.append(media.inline_image(picture.convert("RGB")))
    markers = [f"<image:{i}>" for i in range(1, len(images) + 1)]
    position = 0
    for marker in markers:  # every image is placed exactly once, in order
        found = packet["state"].find(marker, position)
        if found < 0:
            raise ValueError(f"state does not place {marker}")
        position = found + len(marker)
    state: dict[str, Any] = {"text": packet["state"]}
    if images:
        state["images"] = images
    criteria = {c["id"]: c["text"] for c in packet["candidates"]}
    return DecisionRequest(state=state, questions={"action": Question(type="choice", instructions=packet["question"], criteria=criteria)})


def answer(model, packet: dict[str, Any]) -> dict[str, Any]:
    """Decide one packet; never raises."""
    candidates = [c["id"] for c in packet.get("candidates") or []]
    try:
        if len(candidates) == 1:  # nothing to decide; the bridge still records the step
            return {"status": "ok", "selected": candidates[0], "probabilities": {candidates[0]: 1.0}, "forced": True,
                    "model_id": model.model_id, "latency_ms": 0.0}
        try:
            request = request_of(packet)
        except ValueError as error:
            status = "unsupported" if "choice questions need" in str(error) else "failed"
            return {"status": status, "error": f"{type(error).__name__}: {error}"[:500], "model_id": model.model_id}
        if not request.modalities() <= set(model.modalities):
            return {"status": "unsupported", "model_id": model.model_id,
                    "error": f"model does not take {sorted(request.modalities() - set(model.modalities))}"}
        response: ModelResponse = model.decide(request)
        if response.status != "ok" or "action" not in response.answers:
            # an input the model cannot take (over its context or memory) ends the episode as unsupported, final as
            # rejected static items are, instead of a technical failure that every play retries
            rejected = (response.error or "").startswith("rejected: ")
            status = "unsupported" if rejected else response.status if response.status != "ok" else "failed"
            return {"status": status, "error": response.error or "no answer", "model_id": model.model_id, "usage": response.usage}
        probs = response.answers["action"].probs
        selected = max(candidates, key=lambda c: (probs.get(c, 0.0), -candidates.index(c)))
        return {"status": "ok", "selected": selected, "probabilities": probs, "latency_ms": response.latency_ms,
                "usage": response.usage, "model_id": model.model_id}
    except (Exception, SystemExit) as error:  # a model crash (or a dead engine's exit) is a failed decision, reported
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"[:500], "traceback": traceback.format_exc()[-2000:],
                "model_id": model.model_id}


class DecisionServer:
    """Serve one model on 127.0.0.1 in a background thread: `with DecisionServer(model) as url: ...`."""

    def __init__(self, model, port: int = 0, log: Path | None = None):
        self.model, self.port, self.log = model, port, log
        self.lock = threading.Lock()
        self.calls = 0

    def handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, value: dict) -> None:
                body = json.dumps(value, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.reply({"status": "ready", "protocol": PROTOCOL, "model_id": server.model.model_id,
                            "modalities": sorted(server.model.modalities)})

            def do_POST(self):
                packet = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with server.lock:  # one model, one call at a time
                    result = answer(server.model, packet)
                    server.calls += 1
                    if server.log is not None:
                        server.log.parent.mkdir(parents=True, exist_ok=True)
                        with server.log.open("a", encoding="utf-8") as out:
                            out.write(json.dumps({"context": packet.get("context"), "candidates": len(packet.get("candidates") or []),
                                                  **{k: v for k, v in result.items() if k != "traceback"}}, ensure_ascii=False) + "\n")
                self.reply(result)

        return Handler

    def __enter__(self) -> str:
        getattr(self.model, "load", lambda: None)()  # load weights before the first bridge waits on a call
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), self.handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}/"

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def serve_forever(model, port: int, log: Path | None = None) -> None:
    """Blocking variant for `jevarena-mm serve`."""
    server = DecisionServer(model, port, log)
    getattr(model, "load", lambda: None)()
    httpd = HTTPServer(("127.0.0.1", port), server.handler())
    print(f"serving {model.model_id} on http://127.0.0.1:{httpd.server_address[1]}/", flush=True)
    httpd.serve_forever()
