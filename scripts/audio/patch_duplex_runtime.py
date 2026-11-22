"""Create an isolated vLLM-Omni overlay; never modify the installed runtime."""

import argparse, hashlib, json, shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
target = a.out / "vllm_omni"
if target.exists():
    raise ValueError("Use a fresh overlay")
source_file = a.source / "engine/duplex/session/engine_session.py"
expected = "d36509c528090f743755cf0604e7ad37972541f9f81a58d2ed01e6e5cab43ea7"
if hashlib.sha256(source_file.read_bytes()).hexdigest() != expected:
    raise ValueError("Unexpected runtime source; review this version before patching")
shutil.copytree(a.source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
f = target / "engine/duplex/session/engine_session.py"
before = f.read_bytes()
s = before.decode()
start = s.index("    def end_response(")
end = s.index("    def register_history_item(", start)
part = s[start:end]
needle = (
    '        assistant_text = "".join(self._response.assistant_text_buffer).strip()\n'
)
assert part.count(needle) == 1
part = part.replace(
    needle,
    needle
    + """        # Audio-only responses must remain ACK-admissible while PCM drains.
        # Keep the original history position without inventing transcript text.
        if response_id is not None and not assistant_text and self.playback.sent_ms > 0:
            self.snapshot_active_response_for_drain()
""",
)
needle = "        elif response_id is not None and not assistant_text:\n"
assert part.count(needle) == 1
part = part.replace(
    needle,
    "        elif response_id is not None and not assistant_text and self.playback.sent_ms == 0:\n",
)
f.write_text(s[:start] + part + s[end:])
(a.out / "patch-provenance.json").write_text(
    json.dumps(
        {
            "patch": "retain-audio-only-response-until-playback-v1",
            "file": str(f.relative_to(a.out)),
            "original_sha256": hashlib.sha256(before).hexdigest(),
            "patched_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "installed_runtime_modified": False,
        },
        indent=2,
    )
    + "\n"
)
print((a.out / "patch-provenance.json").read_text())
