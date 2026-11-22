"""audio_delivery: the audio files the Hugging Face dataset does not host (hub.NOT_HOSTED_TASKS), rebuilt from sources.

The audio delivery stores each clip as 16-bit PCM WAV with a minimal header (`fmt ` and `data` chunks), at the
source's own rate and channels, named by the sha256 of the source file it was taken from. For each not-hosted task this
script gets that source from its owner or distributor, converts it the same way, and keeps a file only if its sha256 is
the one the items record (`_common.save`):

- MMAU test-mini (aud.know/*/mmau_mini): gamma-lab-umd/MMAU-test-mini at a pinned revision, read over HTTP ranges; a
  clip is the release's audio bytes, or those bytes decoded to 16-bit PCM (MP3 and float WAV sources).
- Full-Duplex-Bench v1 (aud.dialog/turn_timing/full_duplex_bench_v1): the first `visible_until_s` seconds of
  <source_id>/input.wav in BetterUp's betterup/candor-full-duplex-bench at a pinned revision, read with your Hugging Face
  access once you have accepted BetterUp's terms on its page. The CANDOR Corpus License lets a benchmark refer to its
  audio by id and timestamp, with labels, but not include it (clause 4(a)), so only the references are committed.

Decoding uses soundfile 0.14.0 (libsndfile 1.2.2); `pip install -e '.[audio]'` pins it, since another MP3 decoder can
produce other samples. With a local copy of the prepared delivery, set JEVARENA_AUDIO_DATA to it instead: the script
then links the whole delivery (`python -m jevarena_mm.audio_delivery attach`).

    python scripts/fetch/audio_delivery.py "$JEVARENA_MM_MEDIA/audio_delivery"
"""

import hashlib
import io
import os
import platform
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path

from _common import by_sha, missing, report, save, wanted
from jevarena_mm import dataio, hub
from jevarena_mm.registry import TASKS, discover

MMAU = ("gamma-lab-umd/MMAU-test-mini", "ccd9696c0111ea7060827598f310558df0b71b0a", "test_mini.parquet")
CANDOR = ("betterup/candor-full-duplex-bench", "e6ad443be50c779edf1900af22bc39687733719a")


def pcm16_wav(data, rate: int) -> bytes:
    """The delivery's WAV: 16-bit PCM, minimal header, as Python's wave module writes it."""
    channels = 1 if data.ndim == 1 else data.shape[1]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(data.astype("<i2").tobytes())
    return buffer.getvalue()


def decoded(raw: bytes) -> bytes:
    import soundfile

    data, rate = soundfile.read(io.BytesIO(raw), dtype="int16")
    return pcm16_wav(data, rate)


def keep(dest: Path, rel: str, sha: str, payload: bytes) -> None:
    try:
        save(io.BytesIO(payload), dest / rel, sha)
        print(f"audio_delivery: {rel}", flush=True)
    except ValueError as error:  # reported again at the end, with every file still missing
        print(f"audio_delivery: {error}", flush=True)


def mmau(dest: Path, todo: dict[str, str], _items) -> None:
    """Every row of the release is a candidate: its bytes as they are, or decoded. Several delivery files can hold the
    same audio (different source files that decode alike), so a match is saved under each of their names."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    wanted_sha = by_sha(todo)
    repo, revision, name = MMAU
    with HfFileSystem().open(f"datasets/{repo}@{revision}/{name}", "rb", block_size=16 << 20) as remote:
        table = pq.ParquetFile(remote)
        for group in range(table.num_row_groups):
            for cell in table.read_row_group(group, columns=["context"]).column(0).to_pylist():
                raw = (cell or {}).get("bytes") or b""
                for payload in (raw, None):
                    payload = payload if payload is not None else decoded(raw)
                    sha = hashlib.sha256(payload).hexdigest()
                    if sha in wanted_sha:
                        for rel in wanted_sha.pop(sha):
                            keep(dest, rel, sha, payload)
                        break
            if not wanted_sha:
                return
    left = sum(len(rels) for rels in wanted_sha.values())
    print(f"audio_delivery: {left} MMAU files match no row of {repo}@{revision[:7]}, as it is or decoded here "
          f"({platform.machine()}): listed below as missing", flush=True)


def candor(dest: Path, todo: dict[str, str], records: dict[str, dict]) -> None:
    """Each clip is the start of its sample's input.wav, up to the moment the item asks about."""
    import soundfile
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    repo, revision = CANDOR
    for rel, sha in sorted(todo.items()):
        record = records[rel]
        try:
            path = hf_hub_download(repo, f"{record['source_id']}/input.wav", repo_type="dataset", revision=revision)
        except GatedRepoError:  # the other sources still come; the final report lists these files as missing
            print(f"audio_delivery: accept BetterUp's terms on https://huggingface.co/datasets/{repo}, then log in "
                  "with `hf auth login`", flush=True)
            return
        data, rate = soundfile.read(path, dtype="int16")
        keep(dest, rel, sha, pcm16_wav(data[:round(record["visible_until_s"] * rate)], rate))


SOURCES = {
    "aud.dialog/turn_timing/full_duplex_bench_v1": candor,
    "aud.know/reasoning/mmau_mini": mmau,
}


def records(task_name: str) -> dict[str, dict]:
    """rel path under the folder -> the provenance record of the item that uses it (source id, times)."""
    out = {}
    for item in dataio.load_task(TASKS[task_name]()):
        for _, ref in item.request.media():
            out[ref.uri.split("/", 1)[1]] = item.meta["audio_provenance"]
    return out


def main(dest: Path) -> None:
    if os.environ.get("JEVARENA_AUDIO_DATA"):
        if dest.is_dir() and not dest.is_symlink() and not any(dest.iterdir()):
            dest.rmdir()
        raise SystemExit(subprocess.call([sys.executable, "-m", "jevarena_mm.audio_delivery", "attach",
                                          os.environ["JEVARENA_AUDIO_DATA"]]))
    discover()
    files = wanted("audio_delivery")
    by_task: dict[str, dict[str, str]] = defaultdict(dict)
    annotations = []
    for rel, sha in missing(dest, files).items():
        task_name = hub.task_files().get(f"audio_delivery/{rel}")
        if task_name:
            by_task[task_name][rel] = sha
        else:  # the delivery's annotation files the remaining recipes are built from
            annotations.append(rel)
    if annotations:
        print(f"audio_delivery: {len(annotations)} annotation files of the delivery (for the recipes of the tasks "
              "whose annotations are not committed) are not public; set JEVARENA_AUDIO_DATA to a delivery")
    groups, names = defaultdict(dict), defaultdict(list)  # one pass per source, over all its tasks
    for task_name, todo in sorted(by_task.items()):
        fetch = SOURCES.get(task_name)
        if fetch is None:
            print(f"audio_delivery: {len(todo)} files of {task_name} ({hub.NOT_HOSTED_TASKS[task_name]}) have no "
                  f"fetch step yet; they come from {hub.FETCHED_FROM_TASKS[task_name]}")
            continue
        groups[fetch].update(todo)
        names[fetch].append(task_name)
    for fetch, todo in groups.items():
        found = {} if fetch is mmau else {rel: rec for name in names[fetch] for rel, rec in records(name).items()}
        fetch(dest, todo, found)
    report("audio_delivery", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
