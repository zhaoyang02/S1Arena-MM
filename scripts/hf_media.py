"""Maintainer tool for the media dataset on the Hugging Face Hub (jevarena_mm.hub.REPO).

    python scripts/hf_media.py stage --out DIR [--folders a b ...]   # symlink tree + card for the folders (default: all hosted)
    python scripts/hf_media.py upload --dir DIR                       # upload the staged tree, one commit per folder
    python scripts/hf_media.py copy --from REPO --folders a b ...     # server-side copy from another dataset
    python scripts/hf_media.py card [--out README.md]                 # the card (from what the dataset holds) and LICENSES/
    python scripts/hf_media.py check                                  # remote files against the sha256 the items record
    python scripts/hf_media.py attribution                            # licenses/attribution/*.csv from the sources' records

`stage` links every referenced file of the chosen folders (checked against its sha256 first) and writes README.md for
every folder staged so far, so folders staged later are added without dropping earlier ones from the card. `copy`
copies folders that another Hugging Face dataset already holds byte for byte without downloading them; `check` then
compares every file with the sha256 the items record. `card` also uploads licenses/ as LICENSES/: the license texts
some sources ask to keep with their data and the per-file credits `attribution` writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from jevarena_mm import env, hub, media
from jevarena_mm.registry import discover, tasks_under


def all_tasks():
    discover()
    return tasks_under("")


def stage(out: Path, folders: list[str] | None) -> None:
    tasks = all_tasks()
    refs = hub.references(tasks)
    known = {hub.folder(uri) for uri in refs}
    chosen = set(folders or known - hub.NOT_HOSTED_FOLDERS)
    if unknown := chosen - known:
        raise SystemExit(f"no item references {sorted(unknown)}")
    if blocked := chosen & hub.NOT_HOSTED_FOLDERS:
        raise SystemExit(f"not hosted: {sorted(blocked)}")
    root, problems, linked = env.media_dir(), Counter(), 0
    for uri, (sha, _) in sorted(refs.items()):
        if hub.folder(uri) not in chosen or hub.not_hosted_key(uri):
            continue
        source = (root / uri).resolve()
        if not source.is_file():
            problems["missing"] += 1
            continue
        if media.sha256_file(source) != sha:
            problems["sha256 mismatch"] += 1
            continue
        link = out / uri
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and link.resolve() != source:
            link.unlink()
        if not link.is_symlink():
            link.symlink_to(source)
            linked += 1
    media.flush_sha_cache()
    for link in out.rglob("*"):  # drop links to files no longer hosted or referenced
        uri = link.relative_to(out).as_posix()
        if link.is_symlink() and (uri not in refs or hub.not_hosted_key(uri)):
            link.unlink()
    if problems:
        raise SystemExit(f"not staged: {dict(problems)}")
    hosted = {}
    for uri in refs:
        name = hub.folder(uri)
        if (out / uri).is_symlink():
            files, size = hosted.get(name, (0, 0))
            hosted[name] = (files + 1, size + (out / uri).stat().st_size)
    (out / "README.md").write_text(hub.card(tasks, hosted), encoding="utf-8")
    print(f"{linked} new links; staged folders: {', '.join(sorted(hosted))}; "
          f"{sum(v[0] for v in hosted.values()):,} files, {sum(v[1] for v in hosted.values()) / 1e9:.1f} GB")


def upload(folder: Path) -> None:
    """Upload the staged tree, one commit per top-level folder, then the card."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hub.REPO, repo_type="dataset", private=True, exist_ok=True)
    for name in sorted(os.listdir(folder)):
        if (folder / name).is_dir() and not name.startswith("."):
            api.upload_folder(repo_id=hub.REPO, repo_type="dataset", folder_path=folder, allow_patterns=[f"{name}/**"],
                              commit_message=f"Media: {name}")
            print(f"{name}: uploaded")
    print("run `card` to write the dataset card from what the dataset now holds")


def card(out: Path | None) -> None:
    """Write the card from the files REPO holds, and upload it with LICENSE.md and licenses/ (as LICENSES/) unless
    --out is given."""
    from huggingface_hub import CommitOperationAdd, HfApi

    tasks = all_tasks()
    refs = hub.references(tasks)
    hosted: dict[str, tuple[int, int]] = {}
    latency = set(hub.latency_files())
    for entry in HfApi().list_repo_tree(hub.REPO, repo_type="dataset", recursive=True):
        if hasattr(entry, "size") and (entry.path in latency or entry.path in refs and not hub.not_hosted_key(entry.path)):
            files, size = hosted.get(hub.folder(entry.path), (0, 0))
            hosted[hub.folder(entry.path)] = (files + 1, size + entry.size)
    text = hub.card(tasks, hosted)
    if out:
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
        return
    licenses = env.ROOT / "licenses"
    operations = [CommitOperationAdd("README.md", text.encode()), CommitOperationAdd("LICENSE.md", hub.LICENSE_MD.encode())]
    operations += [CommitOperationAdd(f"LICENSES/{path.relative_to(licenses).as_posix()}", str(path))
                   for path in sorted(licenses.rglob("*")) if path.is_file()]
    HfApi().create_commit(hub.REPO, operations, commit_message="Card and licenses", repo_type="dataset")
    print(f"{hub.REPO}: card, LICENSE.md and {len(operations) - 2} files of LICENSES/ uploaded")


def copy(source: str, folders: list[str]) -> None:
    """Server-side copy of hosted folders from another dataset that holds them at the same paths (no download)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hub.REPO, repo_type="dataset", private=True, exist_ok=True)
    for name in folders:  # a folder, or a subfolder such as jevarena_img/img.code_render_match
        name = name.strip("/")
        if hub.not_hosted_key(f"{name}/x"):
            raise SystemExit(f"{name}: not hosted in {hub.REPO}")
        # trailing slashes: the folder's contents into the same path, whether or not that path exists yet
        api.copy_files(f"hf://datasets/{source}/{name}/", f"hf://datasets/{hub.REPO}/{name}/")
        print(f"{name}: copied from {source}")


def check() -> None:
    from huggingface_hub import HfApi

    refs = hub.references(all_tasks())
    remote = {}
    for entry in HfApi().list_repo_tree(hub.REPO, repo_type="dataset", recursive=True, expand=True):
        if getattr(entry, "lfs", None) is not None:
            remote[entry.path] = entry.lfs.sha256
        elif hasattr(entry, "size"):  # a small file kept in git: compare git blob ids, the local file matching the item
            remote[entry.path] = ("git", entry.blob_id)
    counts, bad = Counter(), []
    for uri, (sha, _) in refs.items():
        if isinstance(remote.get(uri), tuple):
            local = env.media_dir() / uri
            if local.is_file():
                data = local.read_bytes()
                same = (media.sha256_file(local) == sha
                        and hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == remote[uri][1])
            else:  # copied server-side, so not here: small files kept in git are cheap to download
                from huggingface_hub import hf_hub_download

                same = hashlib.sha256(Path(hf_hub_download(hub.REPO, uri, repo_type="dataset")).read_bytes()
                                      ).hexdigest() == sha
            remote[uri] = sha if same else None
        if hub.not_hosted_key(uri):
            counts["not hosted"] += 1
        elif uri not in remote:
            counts["not uploaded"] += 1
        elif remote[uri] != sha:
            counts["sha256 mismatch"] += 1
            bad.append(uri)
        else:
            counts["ok"] += 1
    # license files a source asks to keep with every copy (e.g. audio_delivery/LICENSE-TAU-..., LICENSES/) are not media
    latency = set(hub.latency_files())  # the latency sets' closed-loop packets, in no item
    absent = sorted(latency - set(remote))
    extra = sorted(p for p in remote if p not in refs and p not in latency and p not in ("README.md", ".gitattributes")
                   and not Path(p).name.startswith("LICENSE") and not p.startswith("LICENSES/"))
    print(f"{hub.REPO}: {dict(counts)}; latency packets: {len(latency) - len(absent)} of {len(latency)} files; "
          f"{len(extra)} files no item references")
    for uri in bad[:20] + absent[:20] + extra[:20]:
        print(" ", uri)
    if bad or absent:
        sys.exit(1)


COCO_2014 = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip"
AOKVQA = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
VISPR_TEST = "https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017_anno.tar.gz"
FLICKR_FILE = re.compile(r"/(\d+)_[0-9a-f]+(?:_[a-z])?\.jpg$")  # a Flickr image URL: .../<photo id>_<secret>[_<size>].jpg


def flickr_page(photo_id: str) -> str:
    if not photo_id.isdigit():
        raise ValueError(f"not a Flickr id: {photo_id}")
    return f"https://www.flickr.com/photo.gne?id={photo_id}"


def read_tar(url: str) -> dict[str, bytes]:
    with urllib.request.urlopen(url, timeout=300) as response:
        with tarfile.open(fileobj=io.BytesIO(response.read())) as tar:
            return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def attribution(out: Path) -> None:
    """<out>/<folder>.csv: the Flickr page (and license, where the source records it) of every hosted file of the
    folders whose photos and videos keep their own Creative Commons licenses, from each source's own records."""
    sys.path.insert(0, str(Path(__file__).with_name("fetch")))
    from _common import open_zip

    files: dict[str, list[str]] = {}
    for uri in hub.references(all_tasks()):
        if not hub.not_hosted_key(uri):
            files.setdefault(hub.folder(uri), []).append(uri)
    coco, licenses = {}, {}
    archive = open_zip(COCO_2014)
    for member in ("annotations/captions_train2014.json", "annotations/captions_val2014.json"):
        data = json.loads(archive.read(member))
        licenses.update({entry["id"]: entry for entry in data["licenses"]})
        coco.update({image["id"]: image for image in data["images"]})

    def from_coco(image_id: int) -> tuple[str, str, str]:
        image = coco[image_id]
        return (flickr_page(FLICKR_FILE.search(image["flickr_url"]).group(1)), licenses[image["license"]]["name"],
                licenses[image["license"]]["url"])

    aokvqa = {q["question_id"]: q["image_id"] for q in json.loads(read_tar(AOKVQA)["aokvqa_v1p0_val.json"])}
    vispr = {Path(name).stem: json.loads(data)["source_url"] for name, data in read_tar(VISPR_TEST).items()}
    open_images = ("CC BY 2.0 (as Open Images lists it)", "https://creativecommons.org/licenses/by/2.0/")
    rows = {name: {uri: (flickr_page(Path(uri).stem), "", "") for uri in files[name]}  # named by Flickr id
            for name in ("koniq10k", "konvid-1k", "nextqa")}
    rows["okvqa"] = {uri: from_coco(int(Path(uri).stem.rsplit("_", 1)[1])) for uri in files["okvqa"]}
    rows["aokvqa"] = {uri: from_coco(aokvqa[Path(uri).stem]) for uri in files["aokvqa"]}
    rows["vispr"] = {uri: ((flickr_page(FLICKR_FILE.search(vispr[Path(uri).stem]).group(1)), *open_images)
                           if vispr[Path(uri).stem] else ("", "source not recorded by VISPR", ""))
                     for uri in files["vispr"]}
    out.mkdir(parents=True, exist_ok=True)
    for name, entries in rows.items():
        with open(out / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["file", "source_page", "license", "license_url"])
            writer.writerows([uri, *entries[uri]] for uri in sorted(entries))
        print(f"{out / name}.csv: {len(entries)} files")


def main() -> None:
    env.load_env()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folders", nargs="*")
    p = sub.add_parser("upload")
    p.add_argument("--dir", type=Path, required=True)
    p = sub.add_parser("copy")
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--folders", nargs="+", required=True)
    p = sub.add_parser("card")
    p.add_argument("--out", type=Path, help="write it here instead of uploading it")
    sub.add_parser("check")
    p = sub.add_parser("attribution")
    p.add_argument("--out", type=Path, default=env.ROOT / "licenses" / "attribution")
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.out, args.folders)
    elif args.command == "upload":
        upload(args.dir)
    elif args.command == "copy":
        copy(args.source, args.folders)
    elif args.command == "card":
        card(args.out)
    elif args.command == "attribution":
        attribution(args.out)
    else:
        check()


if __name__ == "__main__":
    main()
