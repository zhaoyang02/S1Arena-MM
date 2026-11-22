"""jevarena_img: the images of the hosted image folder that the dataset does not host (hub.NOT_HOSTED_FILES), rebuilt
from their sources byte for byte. jevarena_img_files.csv gives each file's source, the source's sha256 and how it was
encoded.

- img.safety_synthetic (250 files): ImageNet validation photographs, the real half of GenImage's Midjourney subset
  (ImageNet terms of access: shared only with those who first agree to them, https://image-net.org/). Each original is
  member imagenet_midjourney/val/nature/<file> of GenImage's Midjourney split zip; all of them, and the zip's central
  directory, sit in its last part, Midjourney/imagenet_midjourney.zip of the ENSTA-U2IS/GenImage copy on Hugging Face
  (pinned), read here over HTTP ranges (about 40 MB of its 2 GB). Each photograph was encoded twice: decoded, converted
  to RGB, resized to long edge 512 when larger (Lanczos, sizes rounded with round()) and saved as JPEG quality 95, once
  by GenImage's loader and once more when the items' media were written.
- img.social_art_emotion (1 file): a WikiArt painting of ArtEmis (ArtEmis terms of use), from WikiArt's image server,
  encoded once the same way at long edge 1536.

The bytes depend on Pillow's JPEG codec. They were built with Pillow 12.3.0 (libjpeg-turbo 3.1.4.1) and rebuild byte for
byte with Pillow 10.4.0, 11.1.0, 11.2.1, 11.3.0 and 13.0.0; Pillow 11.0.0 (libjpeg-turbo 3.0.4) encodes 4 of the
photographs differently, so the image extra excludes it. A file that differs is reported with whether its source or its
encoding differs. Nothing to supply.
"""

import csv
import hashlib
import io
import struct
import sys
import urllib.request
from functools import partial
from pathlib import Path

from _common import Inflated, RangeFile, hf_token, hf_url, missing, report, save, save_all, wanted

GENIMAGE, GENIMAGE_REVISION = "ENSTA-U2IS/GenImage", "e0e56803752d519fde43af038314db5e84bcfa8f"
LAST_PART = "Midjourney/imagenet_midjourney.zip"


def encode(data: bytes, long_edge: int) -> bytes:
    """Decode, convert to RGB, resize to long_edge when larger (Lanczos), save as JPEG quality 95 (Pillow's defaults
    otherwise: 4:2:0, no ICC profile or EXIF, not optimized or progressive)."""
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    image = image.convert("RGB")
    if max(image.size) > long_edge:
        scale = long_edge / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def central_directory(part: RangeFile) -> tuple[dict[str, tuple[int, int, int, int]], int]:
    """({name: (disk, local header offset, compressed size, method)}, the part's disk number) of a split zip, read
    from its last part (zip64)."""
    start = max(0, part.size - (1 << 16) - 98)
    with part.stream(start, part.size - start) as response:
        tail = response.read()
    k = tail.rfind(b"PK\x06\x06")
    if k < 0:
        raise ValueError(f"{LAST_PART}: no zip64 end of central directory")
    _, _, _, _, disk, cd_disk, _, count, cd_size, cd_offset = struct.unpack("<IQHHIIQQQQ", tail[k:k + 56])
    if cd_disk != disk:
        raise ValueError(f"{LAST_PART}: the central directory starts on part {cd_disk}, not on this one ({disk})")
    with part.stream(cd_offset, cd_size) as response:
        cd = response.read()
    entries, pos = {}, 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        h = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
        method, size, name_length, extra_length, comment_length, member_disk, offset = (
            h[4], h[8], h[10], h[11], h[12], h[13], h[16])
        name = cd[pos + 46:pos + 46 + name_length].decode("utf-8")
        extra, e = cd[pos + 46 + name_length:pos + 46 + name_length + extra_length], 0
        while e + 4 <= len(extra):  # zip64 extra field: the 0xFFFF... values, in this order
            field, length = struct.unpack("<HH", extra[e:e + 4])
            if field == 1:
                values, q = extra[e + 4:e + 4 + length], 0
                if h[9] == 0xFFFFFFFF:
                    q += 8  # uncompressed size, not needed
                if size == 0xFFFFFFFF:
                    size, q = struct.unpack("<Q", values[q:q + 8])[0], q + 8
                if offset == 0xFFFFFFFF:
                    offset, q = struct.unpack("<Q", values[q:q + 8])[0], q + 8
                if member_disk == 0xFFFF:
                    member_disk = struct.unpack("<I", values[q:q + 4])[0]
            e += 4 + length
        entries[name] = (member_disk, offset, size, method)
        pos += 46 + name_length + extra_length + comment_length
    if len(entries) != count:
        raise ValueError(f"{LAST_PART}: read {len(entries)} central directory entries, expected {count}")
    return entries, disk


def genimage_member(part: RangeFile, entry: tuple[int, int, int, int]) -> bytes:
    _, offset, size, method = entry
    with part.stream(offset, 30) as response:
        header = response.read()
    if header[:4] != b"PK\x03\x04" or method not in (0, 8):
        raise ValueError(f"{LAST_PART}: no readable member at offset {offset}")
    with part.stream(offset + 30 + sum(struct.unpack("<HH", header[26:30])), size) as response:
        stream = Inflated(response) if method == 8 else response
        return b"".join(iter(lambda: stream.read(1 << 20), b""))


def wikiart(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "jevarena-mm-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def build(read, row: dict, target: Path, sha: str) -> None:
    """Read the source, check it, encode it the way the item's file was encoded and keep the result if its sha256 is
    the item's."""
    data = read()
    if hashlib.sha256(data).hexdigest() != row["source_sha256"]:
        raise ValueError(f"the source {row['source']} differs from the one the items were built from "
                         f"(sha256 {hashlib.sha256(data).hexdigest()}, expected {row['source_sha256']})")
    for _ in range(int(row["encodes"])):
        data = encode(data, int(row["long_edge"]))
    if hashlib.sha256(data).hexdigest() != sha:
        import PIL
        from PIL import features

        raise ValueError(f"the source is right but Pillow {PIL.__version__} (libjpeg-turbo "
                         f"{features.version('libjpeg_turbo')}) encodes it differently; see the versions in "
                         "scripts/fetch/jevarena_img.py")
    save(io.BytesIO(data), target, sha)


def main(dest: Path) -> None:
    files = wanted("jevarena_img")
    with open(Path(__file__).with_name("jevarena_img_files.csv"), newline="") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}
    todo = missing(dest, files)
    jobs, entries = [], None
    for rel, sha in sorted(todo.items()):
        row = rows.get(rel)
        if row is None:
            print(f"jevarena_img: {rel} is not in jevarena_img_files.csv", flush=True)
        elif row["source"].startswith("https://"):
            jobs.append((rel, partial(build, partial(wikiart, row["source"]), row, dest / rel, sha)))
        else:
            if entries is None:
                part = RangeFile(hf_url(GENIMAGE, LAST_PART, GENIMAGE_REVISION), hf_token())
                entries, disk = central_directory(part)
            entry = entries.get(row["source"])
            if entry is None or entry[0] != disk:
                print(f"jevarena_img: {row['source']} is not in {LAST_PART}", flush=True)
                continue
            jobs.append((rel, partial(build, partial(genimage_member, part, entry), row, dest / rel, sha)))
    save_all(jobs, "jevarena_img")
    report("jevarena_img", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
