"""scripts/fetch/_common.py: zip members read over HTTP ranges, several at a time, behind redirects."""

import hashlib
import http.server
import io
import random
import shutil
import sys
import threading
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "fetch"))
import _common  # noqa: E402


class Ranges(http.server.BaseHTTPRequestHandler):
    """Serves `data` at /file with byte ranges; /start redirects there through a relative and an absolute redirect, as
    Hugging Face does (huggingface.co, then its CDN)."""

    data = b""

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/start":
            self.send_response(307)
            self.send_header("Location", "/moved")
        elif self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/file")
        elif self.path == "/file":
            start, end = (int(x) for x in self.headers["Range"].removeprefix("bytes=").split("-"))
            end = min(end, len(self.data) - 1)
            body = self.data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        else:
            self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def served():
    """(url of /start, the archive's members {name: bytes}) for a zip with stored, deflated and empty members."""
    rng = random.Random(0)
    members = {"videos/a.mp4": rng.randbytes(3 << 20), "videos/b.txt": b"jevarena " * 300_000,
               "videos/empty.txt": b"", "videos/c.bin": rng.randbytes(1 << 16)}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data, zipfile.ZIP_DEFLATED if name.endswith(".txt") else zipfile.ZIP_STORED)
    Ranges.data = buffer.getvalue()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ranges)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/start", members
    server.shutdown()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_range_file_follows_redirects_to_the_bytes(served):
    url, _ = served
    remote = _common.RangeFile(url)
    assert remote.location.endswith("/file") and remote.size == len(Ranges.data)
    with remote.stream(10, 20) as response:
        assert response.read() == Ranges.data[10:30]


def test_members_saved_in_parallel_and_wrong_bytes_reported(served, tmp_path, capsys):
    url, members = served
    want = {name: (name.removeprefix("videos/"), sha(data)) for name, data in members.items()}
    want["videos/c.bin"] = ("c.bin", "0" * 64)  # not the member's bytes
    want["videos/absent.mp4"] = ("absent.mp4", "1" * 64)
    jobs = _common.zip_jobs(url, None, want, tmp_path)
    assert set(want) == {"videos/absent.mp4"}  # what the archive does not hold is left
    _common.save_all(jobs, "test", workers=3)
    for name, data in members.items():
        target = tmp_path / name.removeprefix("videos/")
        assert (target.read_bytes() == data) if name != "videos/c.bin" else not target.exists()
    assert "c.bin: ValueError" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("wanted, whole", [(("videos/a.mp4", "videos/b.txt"), True), (("videos/c.bin",), False)])
def test_hub_archive_downloaded_whole_only_when_mostly_needed(served, tmp_path, monkeypatch, wanted, whole):
    url, members = served
    archive = tmp_path / "archive.zip"
    archive.write_bytes(Ranges.data)
    calls = []

    def hf_hub_download(repo, path, repo_type, revision, local_dir):
        calls.append(path)
        local = Path(local_dir) / path
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(archive, local)
        return str(local)

    monkeypatch.setattr(_common, "hf_url", lambda *args: url)
    monkeypatch.setattr(_common, "hf_token", lambda: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)
    dest = tmp_path / "media"
    todo = {name: (name.removeprefix("videos/"), sha(members[name])) for name in wanted}
    _common.save_hf_zip_members("owner/repo", "data.zip", "0" * 40, todo, dest, "test")
    assert not todo and calls == (["data.zip"] if whole else [])
    for name in wanted:
        assert (dest / name.removeprefix("videos/")).read_bytes() == members[name]
    assert not (dest / ".download").exists()
