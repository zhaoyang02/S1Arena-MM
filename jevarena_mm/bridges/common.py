"""Shared by the bridges (standard library only)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = "jevarena-mm-decision-v1"
_RECORD: dict = {"dir": None, "count": 0}
_RECORD_LOCK = threading.Lock()  # VS-Bench asks its players at once (asyncio.to_thread): one number per call


class Unsupported(Exception):
    """The model cannot take this step (e.g. a modality or an option count it does not support)."""


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def record_to(directory) -> None:
    """Also write every request and reply of this process to DIRECTORY/decisions/NNNNN.json."""
    _RECORD.update(dir=Path(directory) / "decisions", count=0)


def decide(endpoint: str, state: str, question: str, candidates: list[dict], images: list[str], context: dict | None = None,
           timeout: float = 900) -> dict:
    """Ask the service for one action. candidates: [{"id", "text"}]; images: file paths in <image:k> order.
    Returns the service's reply (with "selected"); raises Unsupported or RuntimeError, never picks a default."""
    packet = {"protocol": PROTOCOL, "state": state, "question": question,
              "candidates": [{"id": c["id"], "text": c["text"]} for c in candidates],
              "images": [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in images], "context": context or {}}
    data = json.dumps(packet).encode()
    for attempt in range(3):
        try:
            request = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    if _RECORD["dir"] is not None:
        with _RECORD_LOCK:
            number = _RECORD["count"]
            _RECORD["count"] += 1
        write_json(_RECORD["dir"] / f"{number:05d}.json", {"request": packet, "reply": result})
    if result["status"] == "unsupported":
        raise Unsupported(result.get("error", "unsupported"))
    if result["status"] != "ok":
        raise RuntimeError(f"model call failed: {result.get('error')}")
    if result["selected"] not in {c["id"] for c in candidates}:
        raise RuntimeError("service selected an unknown candidate")
    return result


def outcome(out, status: str, **fields) -> dict:
    """Write DIR/outcome.json: status completed | unsupported | technical_failure, plus the suite's results."""
    value = {"status": status, **fields}
    write_json(Path(out) / "outcome.json", value)
    return value
