"""Shared helpers for scripts/fetch/<folder>.py: the files a folder needs, zip members over HTTP ranges, checked writes.

A fetch script is run as `python scripts/fetch/<folder>.py <$JEVARENA_MM_MEDIA/<folder>>` and must leave every file the
items need from that folder (or, for a partly hosted folder, the files the Hugging Face dataset does not hold) at its
item path, byte for byte, without downloading whole archives when the members can be read on their own. Files are
fetched several at a time (`save_all`), and a zip member is read in one streamed request (`zip_jobs`).
"""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import shutil
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from jevarena_mm import dataio, env, hub, media, recipes
from jevarena_mm.registry import tasks_under

CHUNK = 8 << 20
WORKERS = 8  # files fetched at once by save_all


def wanted(folder: str, only_not_hosted: bool = True) -> dict[str, str]:
    """{path under the folder: sha256} of the files the items need from it (by default only those the dataset lacks),
    plus the annotation files under it that recipes are rebuilt from (jevarena_mm.recipes.SOURCES)."""
    env.load_env()
    out = {}
    for uri, (sha, _) in hub.references(tasks_under("")).items():
        if hub.folder(uri) == folder and (not only_not_hosted or hub.not_hosted_key(uri)):
            out[uri.split("/", 1)[1]] = sha
    for source in recipes.SOURCES.values():
        if hub.folder(source.path) == folder:
            out[source.path.split("/", 1)[1]] = source.sha256
    return out


def missing(dest: Path, files: dict[str, str]) -> dict[str, str]:
    """The files not yet present with the right sha256 (hashes cached by size and mtime, so reruns are cheap); a file
    whose sha256 is not pinned (None) counts as present when it exists."""
    return {rel: sha for rel, sha in files.items()
            if not (dest / rel).is_file() or sha not in (None, media.sha256_file(dest / rel))}


def save(stream, target: Path, sha: str | None) -> None:
    """Copy a readable stream to target through a .partial file; keep it only if its sha256 is the expected one (any,
    when None: a recipe source whose rebuilt items check it)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    try:
        with open(partial, "wb") as out:
            for block in iter(lambda: stream.read(CHUNK), b""):
                digest.update(block)
                out.write(block)
        if sha is not None and digest.hexdigest() != sha:
            raise ValueError(f"{target}: sha256 {digest.hexdigest()} is not the expected {sha}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def hf_url(repo: str, path: str, revision: str = "main", repo_type: str = "dataset") -> str:
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{path}"


def hf_token() -> str | None:
    """The token `hf auth login` stored, or HF_TOKEN."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except ImportError:
        return os.environ.get("HF_TOKEN")


class RangeFile(io.RawIOBase):
    """A remote file read with HTTP Range requests (seekable, read-only). A Hugging Face URL is resolved once with the
    token and then read from the signed location without it; the location is re-resolved when it expires. `stream`
    opens one range as a single response, read as it arrives (whole zip members, see zip_jobs)."""

    def __init__(self, url: str, token: str | None = None):
        self.url, self.token, self.pos = url, token, 0
        self.location, self.size = self._resolve()

    def _resolve(self) -> tuple[str, int]:
        """Follow the redirects to the location that serves the bytes (Hugging Face answers with one or more relative
        redirects on huggingface.co, then one to its CDN) and read the file's size there. The token is sent to
        huggingface.co only."""
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        location = self.url
        for _ in range(10):
            headers = {"Range": "bytes=0-0", "Accept-Encoding": "identity"}
            if self.token and urllib.parse.urlsplit(location).netloc == "huggingface.co":
                headers["Authorization"] = f"Bearer {self.token}"
            try:
                with opener.open(urllib.request.Request(location, headers=headers), timeout=120) as response:
                    return location, int(response.headers["Content-Range"].rsplit("/", 1)[1])
            except urllib.error.HTTPError as error:
                if error.code not in (301, 302, 303, 307, 308):
                    raise
                location = urllib.parse.urljoin(location, error.headers["Location"])
        raise OSError(f"{self.url}: more than 10 redirects")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def stream(self, start: int, length: int):
        """The open response for bytes [start, start + length) of the file (length > 0)."""
        headers = {"Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity"}
        for attempt in range(6):
            try:
                response = urllib.request.urlopen(urllib.request.Request(self.location, headers=headers), timeout=300)
            except urllib.error.HTTPError as error:
                if error.code in (403, 410) and attempt < 5:  # signed location expired
                    self.location, _ = self._resolve()
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            if response.status != 206:
                response.close()
                raise OSError(f"{self.url}: range request answered with HTTP {response.status}")
            return response

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        length = min(len(buffer), self.size - self.pos)
        for attempt in range(6):
            try:
                with self.stream(self.pos, length) as response:
                    data = response.read()
                break
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        n = len(data)
        buffer[:n] = data
        self.pos += n
        return n


def open_zip(url: str, token: str | None = None) -> zipfile.ZipFile:
    """A remote zip archive whose members can be read without downloading the rest. The 1 MB buffer keeps small members
    cheap; `save` reads larger blocks, which bypass it."""
    return zipfile.ZipFile(io.BufferedReader(RangeFile(url, token), buffer_size=1 << 20))


class Inflated:
    """The decompressed bytes of a raw deflate stream (a zip member's data), read in blocks."""

    def __init__(self, stream):
        self.stream, self.inflate = stream, zlib.decompressobj(-15)

    def read(self, size: int = CHUNK) -> bytes:
        while not self.inflate.eof:
            block = self.stream.read(size)
            if not block:
                break
            data = self.inflate.decompress(block)
            if data:
                return data
        return self.inflate.flush()


def save_member(remote: RangeFile, info: zipfile.ZipInfo, target: Path, sha: str | None) -> None:
    """Save one member of a remote zip archive, its data read in a single streamed request."""
    with remote.stream(info.header_offset, 30) as response:
        header = response.read()
    if header[:4] != b"PK\x03\x04" or info.flag_bits & 1:
        raise ValueError(f"{info.filename}: no readable local header at offset {info.header_offset}")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError(f"{info.filename}: compression method {info.compress_type} is not read here")
    if not info.compress_size:
        return save(io.BytesIO(b""), target, sha)
    start = info.header_offset + 30 + sum(struct.unpack("<HH", header[26:30]))
    with remote.stream(start, info.compress_size) as response:
        save(response if info.compress_type == zipfile.ZIP_STORED else Inflated(response), target, sha)


def _listed(remote: RangeFile, members: dict) -> list[zipfile.ZipInfo]:
    """The entries of the remote archive's central directory for the names in members."""
    archive = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20))
    return [archive.getinfo(name) for name in sorted(set(archive.namelist()) & members.keys())]


def _member_jobs(remote: RangeFile, infos: list, members: dict, dest: Path) -> list:
    jobs = []
    for info in infos:
        rel, sha = members.pop(info.filename)
        jobs.append((rel, partial(save_member, remote, info, dest / rel, sha)))
    return jobs


def zip_jobs(url: str, token: str | None, members: dict[str, tuple[str, str | None]], dest: Path) -> list:
    """[(rel, job)] for save_all, one per member of the remote zip archive at url that `members` ({member name: (rel,
    sha256)}) names; only the archive's central directory is read here. The members found are taken out of `members`,
    so what is left is in none of the archives read so far."""
    remote = RangeFile(url, token)
    return _member_jobs(remote, _listed(remote, members), members, dest)


_lock = threading.Lock()


def say(message: str) -> None:
    """Print one line from any of save_all's threads."""
    with _lock:
        print(message, flush=True)


def save_all(jobs: list, label: str, workers: int = WORKERS) -> None:
    """Run the jobs [(rel, job)], `workers` at a time. A job that fails on the network is tried three times; one that
    gets the wrong bytes, a web page or no file is reported and left for `report` to list."""
    done = [0]

    def run(rel: str, job) -> None:
        for attempt in range(3):
            try:
                job()
                with _lock:
                    done[0] += 1
                    print(f"{label}: [{done[0]}/{len(jobs)}] {rel}", flush=True)
                return
            except urllib.error.HTTPError as error:
                if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    return say(f"{label}: {rel}: HTTP {error.code}")
            except (FileNotFoundError, ValueError, KeyError, zipfile.BadZipFile) as error:
                return say(f"{label}: {rel}: {type(error).__name__}: {error}")
            except (OSError, http.client.HTTPException) as error:
                if attempt == 2:
                    return say(f"{label}: {rel}: {type(error).__name__}: {error}")
            time.sleep(10 * (attempt + 1))

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(run, rel, job) for rel, job in jobs]:
            future.result()


def save_hf_zip_members(repo: str, path: str, revision: str, members: dict[str, tuple[str, str | None]], dest: Path,
                        label: str) -> None:
    """Save the members {member name: (rel, sha256)} that the zip archive `path` of the Hugging Face dataset `repo`
    holds, taking them out of `members`. Hugging Face serves range reads through a bridge far slower than its own
    client, so when the members are half of the archive or more (and the disk has room) the archive is downloaded
    whole with huggingface_hub into dest/.download, read there and deleted; otherwise the members are read over
    ranges, several at a time."""
    remote = RangeFile(hf_url(repo, path, revision), hf_token())
    infos = _listed(remote, members)
    needed = sum(info.compress_size for info in infos)
    dest.mkdir(parents=True, exist_ok=True)
    if not infos or needed < remote.size / 2 or shutil.disk_usage(dest).free < remote.size + needed + (1 << 30):
        return save_all(_member_jobs(remote, infos, members, dest), label)
    from huggingface_hub import hf_hub_download

    say(f"{label}: downloading {path} whole ({remote.size / 1e9:.1f} GB, {needed / remote.size:.0%} of it needed)")
    scratch = dest / ".download"
    try:
        local = hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=scratch)
        with zipfile.ZipFile(local) as archive:
            for n, info in enumerate(infos, 1):
                rel, sha = members.pop(info.filename)
                try:
                    with archive.open(info) as stream:
                        save(stream, dest / rel, sha)
                    say(f"{label}: [{n}/{len(infos)}] {rel}")
                except ValueError as error:
                    say(f"{label}: {rel}: {error}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def report(folder: str, files: dict[str, str], dest: Path) -> None:
    """Exit non-zero, listing a few, if any wanted file is still missing or wrong."""
    left = missing(dest, files)
    print(f"{folder}: {len(files) - len(left)} of {len(files)} files present and checked")
    for rel in sorted(left)[:10]:
        print(f"  missing: {rel}")
    if left:
        sys.exit(1)


def hf_parquet_images(repo: str, revision: str, files: list[str], columns: list[str],
                      wanted_sha: dict[str, list[str]], dest: Path, label: str) -> None:
    """Save every image cell (a datasets Image struct: {bytes, path}) of the given parquet files and columns whose
    sha256 is a wanted one, as `dest/<rel>` for each rel of wanted_sha = {sha256: [rel, ...]} (see by_sha). The files
    are read over HTTP ranges, one row group at a time, with the token `hf auth login` stored (gated sources need your
    access to have been granted)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    fs = HfFileSystem(token=hf_token())
    try:
        for name in files:
            if not wanted_sha:
                return
            with fs.open(f"datasets/{repo}@{revision}/{name}", "rb", block_size=16 << 20) as remote:
                table = pq.ParquetFile(remote)
                for group in range(table.num_row_groups):
                    for column in table.read_row_group(group, columns=columns).columns:
                        for cell in column.to_pylist():
                            data = (cell or {}).get("bytes") or b""
                            sha = hashlib.sha256(data).hexdigest()
                            for rel in wanted_sha.pop(sha, []):
                                save(io.BytesIO(data), dest / rel, sha)
                                print(f"{label}: {rel}", flush=True)
    except (GatedRepoError, HfHubHTTPError, PermissionError, FileNotFoundError) as error:
        if "403" in str(error) or "401" in str(error) or isinstance(error, (GatedRepoError, PermissionError)):
            print(f"{label}: no access to {repo} ({type(error).__name__}). Request access on "
                  f"https://huggingface.co/datasets/{repo}, then log in with `hf auth login` or set HF_TOKEN.")
            sys.exit(2)
        raise


def by_sha(files: dict[str, str]) -> dict[str, list[str]]:
    """{rel: sha256} -> {sha256: [rel, ...]}: one source file can sit at several item paths."""
    out: dict[str, list[str]] = {}
    for rel, sha in sorted(files.items()):
        out.setdefault(sha, []).append(rel)
    return out


def source_paths(folder: str) -> dict[str, str]:
    """{path under the folder: the source_path its item gives for it} (meta.media, in the order of the request's
    media), for the files the items reference in that folder."""
    out: dict[str, str] = {}
    for task in tasks_under(""):
        if not task.media_root and not task.name.startswith("img."):
            continue
        rows = recipes.load(task)  # a recipe's media records are never tokenized, so they are read before a build
        pairs = ([(recipes.row_media(row), row["item"]["meta"].get("media")) for row in rows] if rows else
                 [([ref for _, ref in item.request.media()], item.meta.get("media")) for item in dataio.load_task(task)])
        for refs, records in pairs:
            for ref, record in zip(refs, records or []):
                if hub.folder(ref.uri) == folder and isinstance(record, dict) and record.get("source_path"):
                    out[ref.uri.split("/", 1)[1]] = record["source_path"]
    return out
