"""Prepare full-stream and output-triggered episodes from Full-Duplex-Bench's sets.

The two CANDOR sets come from BetterUp (betterup/candor-full-duplex-bench on Hugging Face, after accepting the CANDOR
Corpus License), which keeps them as folders: <set>/<id>/input.wav and its annotation JSON. The two synthetic v1.5 sets
come from Full-Duplex-Bench as ZIPs. Each set is read from <source>/<set>.zip if present, else from <source>/<set>/.
"""

import hashlib
import io
import json
import math
import zipfile

SUBSETS = [("candor_turn_taking", "stream"), ("candor_pause_handling", "stream"),
           ("user_interruption", "reactive"), ("user_backchannel", "reactive")]


class _Folder:
    """A set kept as a folder, read like the ZIP of the same name (member names start with <set>/)."""

    def __init__(self, root, subset):
        self.root = root
        self.names = sorted(p.relative_to(root).as_posix() for p in (root / subset).rglob("*")
                            if p.is_file() and not p.name.startswith("."))

    def namelist(self):
        return self.names

    def read(self, name):
        return (self.root / name).read_bytes()

    def fingerprint(self):
        """sha256 over "<name>\t<sha256>" lines of every file, sorted by name."""
        lines = "".join(f"{n}\t{hashlib.sha256(self.read(n)).hexdigest()}\n" for n in self.names)
        return hashlib.sha256(lines.encode()).hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def prepare(source, out, per_task=2, max_input_s=60):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if per_task < 0:
        raise ValueError("per-task must be nonnegative; 0 selects all eligible cases")
    if (out / "manifest.json").exists():
        raise ValueError("Use a new manifest directory")
    out.mkdir(parents=True, exist_ok=True)
    media = out / "media"
    media.mkdir(exist_ok=True)
    cases = []
    skipped = []
    archives = {}

    def save(z, name):
        original = z.read(name)
        x, sr = sf.read(io.BytesIO(original), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            gcd = math.gcd(sr, 16000)
            x = resample_poly(x, 16000 // gcd, sr // gcd)
        pcm = np.clip(np.rint(x * 32768), -32768, 32767).astype("<i2").tobytes()
        digest = hashlib.sha256(pcm).hexdigest()
        path = media / (digest + ".wav")
        sf.write(path, np.frombuffer(pcm, dtype="<i2"), 16000, subtype="PCM_16")
        return {
            "path": str(path.relative_to(out)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "duration_s": len(pcm) / 32000,
            "source_member": name,
            "source_sha256": hashlib.sha256(original).hexdigest(),
        }

    for subset, mode in SUBSETS:
        archive = source / (subset + ".zip")
        if archive.exists():
            archives[archive.name] = hashlib.sha256(archive.read_bytes()).hexdigest()
            opened = zipfile.ZipFile(archive)
        else:
            opened = _Folder(source, subset) if (source / subset).is_dir() else None
            if not opened or not opened.names:
                raise FileNotFoundError(f"{subset}: neither {archive} nor the folder {source / subset} is there")
            archives[subset + "/"] = opened.fingerprint()
        with opened as z:
            folders = sorted(
                {
                    n.rsplit("/", 1)[0]
                    for n in z.namelist()
                    if n.startswith(subset + "/") and n.endswith("/input.wav")
                },
                key=lambda n: int(n.split("/")[-1]),
            )
            count = 0
            for folder in folders:
                if per_task and count >= per_task:
                    break
                name = folder + ("/input.wav" if mode == "stream" else "/context.wav")
                duration = sf.info(io.BytesIO(z.read(name))).duration
                if duration > max_input_s:
                    skipped.append(
                        {
                            "source_id": folder,
                            "reason": "input_exceeds_duration_selection_limit",
                            "duration_s": duration,
                        }
                    )
                    continue
                case = {
                    "id": folder.replace("/", "-"),
                    "source_id": folder,
                    "mode": mode,
                    "category": subset,
                    "audio": save(z, name),
                    "license": "CANDOR Corpus License v2.0 (BetterUp)"
                    if mode == "stream"
                    else "MIT synthetic FDB v1.5 data",
                }
                if mode == "stream":
                    label = (
                        "turn_taking.json"
                        if subset == "candor_turn_taking"
                        else "pause.json"
                    )
                    annotations = json.loads(z.read(folder + "/" + label))
                    case["gold_events"] = [
                        {
                            "type": "turn"
                            if subset == "candor_turn_taking"
                            else "pause",
                            "interval_s": e["timestamp"],
                        }
                        for e in annotations
                    ]
                    if any(
                        not 0
                        <= e["interval_s"][0]
                        < e["interval_s"][1]
                        <= case["audio"]["duration_s"]
                        for e in case["gold_events"]
                    ):
                        skipped.append(
                            {"source_id": folder, "reason": "invalid_held_out_event"}
                        )
                        continue
                else:
                    event = (
                        "interrupt.wav"
                        if subset == "user_interruption"
                        else "backchannel.wav"
                    )
                    case["user_event"] = save(z, folder + "/" + event)
                    case["trigger_after_audible_s"] = 1.0
                    case["gold_events"] = []
                cases.append(case)
                count += 1
            if count == 0 or (per_task and count < per_task):
                raise ValueError(f"Insufficient eligible cases in {subset}")
    manifest = {
        "version": "audio-live-v1",
        "selection": {
            "per_task": per_task,
            "ordering": "numeric source ID",
            "max_input_s": max_input_s,
            "skipped": skipped,
        },
        "source": "CANDOR sets: https://huggingface.co/datasets/betterup/candor-full-duplex-bench; v1.5 sets: "
                  "https://github.com/DanielLin94144/Full-Duplex-Bench",
        "archives_sha256": archives,
        "cases": cases,
        "scope": "Full audio replay plus timing-reactive scripted user; not official FDB scoring or semantic user simulation",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
