"""Something-Something v2 validation videos (Goyal et al., 2017), from Qualcomm's official package.

Terms: the Something-Something v2 license (research use; PAGE) forbids redistribution, so the videos are not in the
Hugging Face dataset; by downloading them you accept that license. Qualcomm serves the package without a login as one
.tgz split into two parts (20bn-something-something-v2-00 and -01, 19.4 GB). It is streamed once, writing only the
needed videos and stopping after the last one; videos/<id>.webm is the archive member 20bn-something-something-v2/<id>.webm,
unchanged. SSV2_DIR (or a second argument) may point at a downloaded copy instead: the parts (also the older 20-part
split, zipped or not) or extracted videos.

vid.basic/temporal/ssv2 is a recipe filled from the validation labels (recipes source ssv2): annotations/validation.json is
labels/validation.json of Qualcomm's label package (20bn-something-something-download-package-labels.zip, read over HTTP
ranges), or of that zip or file in SSV2_DIR, checked by sha256; `jevarena-mm build` also checks every item it fills
against the recipe's id.
"""

import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from _common import missing, open_zip, report, save, wanted

PAGE = "https://www.qualcomm.com/developer/software/something-something-v-2-dataset"
URL = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/"
PARTS = ("20bn-something-something-v2-00", "20bn-something-something-v2-01")
INNER = "20bn-something-something-v2/"  # folder of the videos in the archive
LABELS = "20bn-something-something-download-package-labels.zip"  # holds labels/validation.json
LABELS_REL = "annotations/validation.json"


def labels(dest: Path, folder: Path | None) -> None:
    """labels/validation.json from SSV2_DIR (the label zip, or the file itself), else from Qualcomm's label package."""
    local = folder and next((p for p in (folder / LABELS, folder / "labels" / "validation.json", folder / "validation.json")
                             if p.is_file()), None)
    if local and local.suffix == ".json":
        with open(local, "rb") as f:
            keep(f, dest, LABELS_REL, None)
        return
    with (zipfile.ZipFile(local) if local else open_zip(URL + LABELS)) as archive:
        with archive.open("labels/validation.json") as member:
            keep(member, dest, LABELS_REL, None)


def local_parts(folder: Path) -> list:
    """Openers of the package parts in folder, in order (plain or zipped)."""
    openers = []
    for n in range(20):
        part = folder / f"20bn-something-something-v2-{n:02d}"
        if part.is_file():
            openers.append(lambda part=part: open(part, "rb"))
        elif part.with_suffix(".zip").is_file():
            archive = zipfile.ZipFile(part.with_suffix(".zip"))
            openers.append(lambda archive=archive: archive.open(max(archive.infolist(), key=lambda i: i.file_size)))
    return openers


class Joined:
    """The parts read one after another as a single stream."""

    def __init__(self, openers: list):
        self.openers, self.current = iter(openers), None

    def read(self, size: int = -1) -> bytes:
        while True:
            if self.current is None:
                opener = next(self.openers, None)
                if opener is None:
                    return b""
                self.current = opener()
            data = self.current.read(size)
            if data:
                return data
            self.current.close()
            self.current = None


def keep(stream, dest: Path, rel: str, sha: str | None) -> None:
    try:
        save(stream, dest / rel, sha)
        print(rel, flush=True)
    except ValueError as error:  # an official file that differs from the one the items were built on
        print(error, file=sys.stderr)


def main(dest: Path, source: str | None) -> None:
    files = wanted("ssv2")  # also loads .env, which may set SSV2_DIR
    source = source or os.environ.get("SSV2_DIR")
    folder = Path(source) if source else None
    todo = {INNER + rel.removeprefix("videos/"): (rel, sha) for rel, sha in missing(dest, files).items()
            if rel != LABELS_REL}
    if LABELS_REL in missing(dest, files):
        labels(dest, folder)
    if folder:  # videos already extracted
        for name, (rel, sha) in list(todo.items()):
            path = next((p for p in (folder / name, folder / Path(name).name) if p.is_file()), None)
            if path:
                with open(path, "rb") as f:
                    keep(f, dest, rel, sha)
                del todo[name]
    if todo:
        openers = local_parts(folder) if folder else []
        if not openers:
            print(f"ssv2: streaming Qualcomm's package (19.4 GB) for {len(todo)} videos; the Something-Something v2 "
                  f"license applies ({PAGE})", flush=True)
            openers = [lambda part=part: urllib.request.urlopen(URL + part, timeout=600) for part in PARTS]
        with tarfile.open(fileobj=Joined(openers), mode="r|gz", bufsize=1 << 20) as archive:
            for member in archive:
                if member.name in todo:
                    keep(archive.extractfile(member), dest, *todo.pop(member.name))
                    if not todo:
                        break
    report("ssv2", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
