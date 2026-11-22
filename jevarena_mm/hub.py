"""The gated Hugging Face dataset that hosts the media: which files it holds, its card, and downloading from it.

The dataset mirrors $JEVARENA_MM_MEDIA: a file an item references as `<folder>/<path>` is `<folder>/<path>` in the
dataset, byte for byte the file the item hashes (a source's original file, or one a team built from it). Everything is
hosted, so that a run needs one download, except media whose terms say they may not be redistributed (or be shared
only with people who accepted them), or whose terms can be read only after registering: as uri patterns (NOT_HOSTED),
as every file of a task whose files share a folder with hosted ones (NOT_HOSTED_TASKS), or as single files
(NOT_HOSTED_FILES). scripts/fetch/<folder>.py gets those from their owner or distributor, with the user's own access.
"""

from __future__ import annotations

import fnmatch
import json
import os
from collections import defaultdict
from functools import cache
from pathlib import Path

from . import env, recipes

REPO = os.environ.get("JEVARENA_MM_HF_REPO", "Swindl/S1Arena-MM")
# The dataset's tag for this release: a release reads the media it was checked with, whatever the dataset holds later.
REVISION = os.environ.get("JEVARENA_MM_HF_REVISION", "v1.0.0")

# Media whose terms forbid redistribution (or sharing beyond people who accepted them), or can be read only after
# registering, as uri patterns (fnmatch, `*` also matches "/") with the terms; scripts/fetch/<folder>.py gets these
# files from their owner or distributor.
NOT_HOSTED = {
    "charades/*": "Charades license, local evaluation only (https://prior.allenai.org/projects/charades)",
    "ego4d/*": "Ego4D license agreement (https://ego4d-data.org/)",
    "egoplan-bench2/*": "cut from Ego4D; Ego4D license agreement (https://ego4d-data.org/)",
    "egoschema/*": "cut from Ego4D; Ego4D license agreement (https://ego4d-data.org/)",
    "hateful_memes/*": "Hateful Memes Dataset License (section 6): no distributing or hosting; obtained from "
                       "DrivenData",
    "infographicvqa/*": "InfographicVQA terms (RRC portal, after registration): images may not be redistributed; "
                        "annotations CC BY 4.0 (https://rrc.cvc.uab.es/?ch=17)",
    "mvbench/Moments_in_Time_Raw/*": "Moments in Time clips; its terms: no distributing them, nor copying them onto a "
                                     "public site (http://moments.csail.mit.edu/)",
    "mvbench/original-publisher/charades-official-pinned/*": "a Charades video; Charades license",
    "mvbench/ssv2_video/*": "Something-Something v2 clips; Something-Something v2 license",
    "mvbench/sta/*": "Charades-STA clips cut from Charades; Charades license",
    "mvbench/star/*": "STAR clips cut from Charades; Charades license",
    "nlvr2/*": "NLVR2 Terms of Service (term 4): access only for a researcher's associates and colleagues who accept "
               "them; NLVR2 does not license the images (https://github.com/lil-lab/nlvr)",
    "soccernet-mvfoul/*": "SoccerNet shares MVFoul's clips and annotations under its NDA "
                          "(https://github.com/SoccerNet/sn-mvfoul)",
    "ssv2/*": "Something-Something v2 license (https://www.qualcomm.com/developer/software/something-something-v-2-dataset)",
    "videomme/*": "Video-MME terms: no distribution without the authors' approval (https://github.com/MME-Benchmarks/Video-MME)",
    "vsi-bench/*scannet_*": "ScanNet terms of use: shared only with people who signed them (http://www.scan-net.org/)",
    "vsi-bench/*scannetpp_*": "ScanNet++ terms of use: shared only after signing them (https://kaldir.vc.in.tum.de/scannetpp/)",
    "wikiart/*": "WikiArt images of ArtEmis; ArtEmis terms of use: non-commercial research, shared only with those who "
                 "accept them (https://www.artemisdataset.org/materials/artemis_terms_of_use.txt)",
    "winoground/*": "Winoground license agreement (Meta, section 3): no distributing or hosting its images or their "
                    "metadata (https://huggingface.co/datasets/facebook/winoground)",
}
# Where scripts/fetch/<folder>.py gets them. The user's own access is needed for Ego4D (license keys), Winoground
# (approved Hugging Face access) and Hateful Memes and InfographicVQA (the user's own download).
FETCHED_FROM = {
    "charades/*": "AI2's public Charades archive",
    "ego4d/*": "Ego4D's S3 buckets, with your Ego4D license keys",
    "egoplan-bench2/*": "EgoPlan-Bench2's reference clips (TencentARC/SEED-Bench-R1 on Hugging Face)",
    "egoschema/*": "EgoSchema's own release (Google Drive, or the lmms-lab/egoschema mirror)",
    "hateful_memes/*": "neuralcatcher/hateful_memes on Hugging Face (pinned), a user's copy, not an official release: "
                       "none is left (DrivenData closed sign-ups in 2021, hatefulmemeschallenge.com offline since "
                       "2024); or your own copy (HATEFUL_MEMES_DIR)",
    "infographicvqa/*": "your copy from the RRC portal (INFOGRAPHICVQA_DIR)",
    "mvbench/Moments_in_Time_Raw/*": "MVBench's release (OpenGVLab/MVBench)",
    "mvbench/original-publisher/charades-official-pinned/*": "AI2's public Charades archive",
    "mvbench/ssv2_video/*": "MVBench's release (OpenGVLab/MVBench)",
    "mvbench/sta/*": "MVBench's release (OpenGVLab/MVBench)",
    "mvbench/star/*": "MVBench's release (OpenGVLab/MVBench)",
    "nlvr2/*": "the lmms-lab/NLVR2 release on Hugging Face (public)",
    "soccernet-mvfoul/*": "SoccerNet's release (SoccerNet/SN-MVFouls-2024)",
    "ssv2/*": "Qualcomm's official package (streamed; no login)",
    "videomme/*": "Video-MME's release (lmms-lab/Video-MME)",
    "vsi-bench/*scannet_*": "VSI-Bench's release (nyu-visionx/VSI-Bench)",
    "vsi-bench/*scannetpp_*": "VSI-Bench's release (nyu-visionx/VSI-Bench)",
    "wikiart/*": "WikiArt's image server (the image URLs ArtEmis lists)",
    "winoground/*": "facebook/winoground, with your Hugging Face token after you accept its license",
}
# Tasks none of whose files are hosted, for folders that also hold hosted files under content-hash names (the
# audio_delivery/media/<sha256>.wav), with their terms and where the fetch script gets them.
NOT_HOSTED_TASKS = {
    "aud.dialog/turn_timing/full_duplex_bench_v1": "CANDOR Corpus License v2.0 (BetterUp), clauses 3(a) and 4(a): a "
                                                   "benchmark may refer to its audio by id and timestamp, "
                                                   "with labels, but not include it",
    "aud.know/reasoning/mmau_mini": "its IEMOCAP and VoxCeleb clips: IEMOCAP's license does not allow redistribution, "
                                    "and VoxCeleb's authors no longer distribute its audio",
}
FETCHED_FROM_TASKS = {
    "aud.dialog/turn_timing/full_duplex_bench_v1": "BetterUp's gated betterup/candor-full-duplex-bench at e6ad443, "
                                                   "after you accept BetterUp's terms on its page",
    "aud.know/reasoning/mmau_mini": "MMAU's release (gamma-lab-umd/MMAU-test-mini), decoded to WAV",
}
# Single files of hosted folders that are not hosted (not_hosted_files.json: uri -> key), with the terms and where the
# fetch script gets them.
NOT_HOSTED_FILE_TERMS = {
    "imagenet": "ImageNet photographs (the real half of GenImage's Midjourney subset): ImageNet terms of access, shared "
                "only with research associates and colleagues who first agree to them (https://image-net.org/)",
    "wikiart": "A WikiArt painting of ArtEmis, resized; ArtEmis terms of use: non-commercial research, shared only with "
               "those who accept them (https://www.artemisdataset.org/materials/artemis_terms_of_use.txt)",
}
FETCHED_FROM_FILES = {
    "imagenet": "the last part of GenImage's Midjourney split zip in the ENSTA-U2IS/GenImage copy on Hugging Face "
                "(pinned), re-encoded as the items were built",
    "wikiart": "WikiArt's image server, resized as the item was built",
}
NOT_HOSTED_FILES: dict[str, str] = json.loads(
    (Path(__file__).with_name("not_hosted_files.json")).read_text(encoding="utf-8"))["files"]
# Folders not hosted at all.
NOT_HOSTED_FOLDERS = {key[:-2] for key in NOT_HOSTED if key.endswith("/*") and key.count("/") == 1}


# Of these tasks, only the files of the items from the given original datasets are not hosted; the dataset hosts the
# rest. MMAU's reasoning clips from IEMOCAP (its license does not allow redistribution) and VoxCeleb (its authors no
# longer distribute its audio); a file that any such item uses counts as theirs. MMAU's other clips are hosted: MMAU
# test-mini is CC BY-NC 4.0, and none of its other sources forbids redistribution. Its Song Describer and FMA clips
# (MP3s) also decode differently on x86 and ARM CPUs, so they could not be rebuilt byte for byte everywhere.
NOT_HOSTED_SOURCES = {"aud.know/reasoning/mmau_mini": ("iemocap", "voxceleb")}


@cache
def task_files() -> dict[str, str]:
    """uri -> name of the NOT_HOSTED_TASKS task that references it (from its recipe where items are not built); for a
    task in NOT_HOSTED_SOURCES, only the files of its items from those sources."""
    from . import dataio
    from .registry import TASKS, discover

    discover()
    out = {}
    for name in (n for n in NOT_HOSTED_TASKS if n in TASKS):
        task = TASKS[name]()
        refs = recipes.media_of(task)
        if name in NOT_HOSTED_SOURCES:
            theirs = {ref.uri for item in dataio.load_task(task)
                      if (item.meta.get("audio_provenance") or {}).get("original_dataset") in NOT_HOSTED_SOURCES[name]
                      for _, ref in item.request.media()}
            refs = [ref for ref in refs if ref.uri in theirs]
        out.update({ref.uri: name for ref in refs})
    return out


def not_hosted_key(uri: str) -> str | None:
    """The NOT_HOSTED pattern, NOT_HOSTED_TASKS task or NOT_HOSTED_FILE_TERMS key that covers uri, or None if the
    dataset hosts it."""
    key = next((key for key in NOT_HOSTED if fnmatch.fnmatchcase(uri, key)), None)
    return key or (task_files().get(uri) if NOT_HOSTED_TASKS else None) or NOT_HOSTED_FILES.get(uri)


def partly_hosted(name: str) -> bool:
    """A hosted folder some of whose files are not (its fetch script gets those)."""
    if name in NOT_HOSTED_FOLDERS:
        return False
    return (any(folder(key) == name for key in NOT_HOSTED)
            or (bool(NOT_HOSTED_TASKS) and any(folder(uri) == name for uri in task_files()))
            or any(folder(uri) == name for uri in NOT_HOSTED_FILES))


GATE = """\
extra_gated_prompt: >-
  These are media files of public research datasets, hosted so that S1Arena-MM can be run. Each file
  stays under its source's license and terms (tables below), which this page does not change. Several sources allow
  non-commercial research use only; by requesting access you agree to follow each file's license and terms, including
  those limits.
extra_gated_fields:
  Name: text
  Affiliation: text
  Country: country
  I will follow each file's source license and terms, including non-commercial limits where a source has them: checkbox
"""


def references(tasks) -> dict[str, tuple[str, set[str]]]:
    """uri -> (sha256, names of the tasks that reference it), over every media and marks file of the given tasks
    (from their recipes where the items are not built yet)."""
    out: dict[str, tuple[str, set[str]]] = {}
    for task in tasks:
        for ref in recipes.media_of(task):
            pairs = [(ref.uri, ref.sha256)] + ([(ref.marks.uri, ref.marks.sha256)] if ref.marks else [])
            for uri, sha in pairs:
                if uri in out and out[uri][0] != sha:
                    raise ValueError(f"{uri}: referenced with two sha256 values")
                out.setdefault(uri, (sha, set()))[1].add(task.name)
    return out


def folder(uri: str) -> str:
    return uri.split("/", 1)[0]


# The closed-loop decision packets of the latency sets (data/latency/<version>.json): hosted, but in no item.
LATENCY = "latency"
LATENCY_SOURCE = ("Closed-loop decision packets of the latency set: screenshots and legal actions from recorded "
                  "GameWorld, BALROG and VS-Bench episodes (built for this benchmark)")
LATENCY_LICENSE = "each environment's own license"


def latency_files() -> list[str]:
    """Every file of the latency sets' closed-loop packets as the dataset holds them: each packet.json and the
    screenshots it lists (image-1.png, ...)."""
    files = []
    for path in sorted((env.ROOT / "data" / "latency").glob("*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")).get("loop", []):
            where = entry["packet"].rsplit("/", 1)[0]
            files += [entry["packet"], *(f"{where}/image-{i}.png" for i in range(1, entry["images"] + 1))]
    return files


XET_WORKERS = 64  # files one Xet download group fetches at once
FILE_WORKERS = 16  # files downloaded one by one at once (those kept in git rather than on Xet)


def fetch(folders: list[str], dest: Path) -> None:
    """Download the given folders from the dataset into dest (= $JEVARENA_MM_MEDIA); files already there (same size)
    are kept. The files stored on Xet come in one hf_xet download group per folder: one group fetches many files at
    once, where huggingface_hub's snapshot_download starts a group per file (measured 2026-11-21: 1,000 small images in
    6 s against 76 s). The few files kept in git are downloaded one by one, several at a time. Where hf_xet lacks the
    session API, snapshot_download fetches everything."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        try:
            _fetch_xet(folders, dest)
        except (ImportError, AttributeError) as error:
            print(f"{REPO}: no hf_xet download groups ({error}); downloading with snapshot_download", flush=True)
            snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=dest,
                              allow_patterns=[f"{name}/*" for name in folders])
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise SystemExit(f"{REPO}: no access ({type(error).__name__}). Accept the terms on "
                         f"https://huggingface.co/datasets/{REPO}, then log in with `hf auth login` "
                         "or set HF_TOKEN.") from error


def _fetch_xet(folders: list[str], dest: Path) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", str(XET_WORKERS))
    from hf_xet import XetFileInfo, XetSession
    from huggingface_hub import HfApi, constants, hf_hub_download
    from huggingface_hub.utils import build_hf_headers

    api = HfApi()
    commit = api.dataset_info(REPO, revision=REVISION).sha
    wanted, groups, one_by_one = set(folders), defaultdict(list), []
    for entry in api.list_repo_tree(REPO, repo_type="dataset", revision=commit, recursive=True):
        if not hasattr(entry, "size") or folder(entry.path) not in wanted:
            continue
        target = dest / entry.path
        if target.is_file() and target.stat().st_size == entry.size:
            continue
        (groups[folder(entry.path)] if getattr(entry, "xet_hash", None) else one_by_one).append(entry)
    refresh = f"{constants.ENDPOINT}/api/datasets/{REPO}/xet-read-token/{commit}"
    session = XetSession()
    for name, entries in sorted(groups.items()):
        start = time.time()
        try:
            with session.new_file_download_group(token_refresh_url=refresh,
                                                 token_refresh_headers=build_hf_headers()) as group:
                for entry in entries:
                    part = dest / f"{entry.path}.incomplete"
                    part.parent.mkdir(parents=True, exist_ok=True)
                    group.start_download_file(XetFileInfo(entry.xet_hash, entry.size), str(part))
        except Exception as error:  # the files it did deliver are kept; the others go one by one
            print(f"{name}: download group stopped ({type(error).__name__}: {error})", flush=True)
        for entry in entries:
            part = dest / f"{entry.path}.incomplete"
            if part.is_file() and part.stat().st_size == entry.size:
                part.replace(dest / entry.path)
            else:
                part.unlink(missing_ok=True)
                one_by_one.append(entry)
        size = sum(entry.size for entry in entries)
        print(f"{name}: {len(entries):,} files, {size / 1e9:.2f} GB in {time.time() - start:.0f} s", flush=True)
    if one_by_one:
        print(f"{REPO}: {len(one_by_one):,} more files one by one", flush=True)

        def get(entry) -> None:
            hf_hub_download(REPO, entry.path, repo_type="dataset", revision=commit, local_dir=dest)

        with ThreadPoolExecutor(FILE_WORKERS) as pool:
            list(pool.map(get, one_by_one))


MODALITIES = (("img", "Image"), ("vid", "Video"), ("lvid", "Long video"), ("av", "Audio-visual"), ("aud", "Audio"),
              ("loop", "Closed loop"))


def _items(task) -> int:
    from . import dataio

    return len(recipes.load(task)) or len(dataio.load_task(task))


LICENSE_MD = """# Licenses

Each source of S1Arena-MM keeps its own license, listed in README.md; LICENSES/ holds the license texts and credits some
sources ask to keep with their files. Media that S1Arena-MM generates are licensed CC BY 4.0. The S1Arena-MM code is
licensed Apache-2.0.
"""


def card(tasks, hosted: dict[str, tuple[int, int]]) -> str:
    """Dataset card. hosted: folder -> (files, bytes) in REPO."""
    sources: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    refs = references(tasks)
    task_by_name = {task.name: task for task in tasks}
    for uri, (_, names) in refs.items():
        for name in names:
            task = task_by_name[name]
            if task.leaf != "lvid.route/escalation":  # quotes other tasks' items; their cards carry the licenses
                # hosted folders list the sources of their hosted files, patterns every source they cover; a task
                # in NOT_HOSTED_TASKS has its own row under "Not hosted"
                key = not_hosted_key(uri)
                if key not in NOT_HOSTED_TASKS:
                    sources[key or folder(uri)].add((task.source, task.license, task.url))

    sources[LATENCY].add((LATENCY_SOURCE, LATENCY_LICENSE, ""))

    def described(name):
        for source, license, url in sorted(sources.get(name, {("", "", "")})):
            yield f"[{source}]({url})" if url else source, license

    def modality_of(name: str) -> str:
        if name == LATENCY:
            return "Closed loop"
        prefixes = {t.split(".")[0] for uri, (_, names) in refs.items() if folder(uri) == folder(name) for t in names}
        return ", ".join(label for prefix, label in MODALITIES if prefix in prefixes)

    hosted_rows = "\n".join(f"| `{n}` | {modality_of(n)} | {s} | {lic} | {hosted[n][0]:,} | "
                            f"{hosted[n][1] / 1e9:.2f} GB |"
                            for n in sorted(hosted) for s, lic in described(n))
    missing_rows = "\n".join(f"| `{n}` | {s} | {NOT_HOSTED[n]} | {FETCHED_FROM[n]} |"
                             for n in sorted(NOT_HOSTED) for s, _ in described(n))
    for name in sorted(n for n in NOT_HOSTED_TASKS if n in task_by_name):
        task = task_by_name[name]
        where = folder(recipes.media_of(task)[0].uri)
        source = f"[{task.source}]({task.url})" if task.url else task.source
        missing_rows += (f"\n| `{name}` (its files in `{where}`) | {source} | {NOT_HOSTED_TASKS[name]} | "
                         f"{FETCHED_FROM_TASKS[name]} |")
    for key in sorted(NOT_HOSTED_FILE_TERMS):
        uris = sorted(uri for uri, k in NOT_HOSTED_FILES.items() if k == key and uri in refs)
        if uris:
            where = "/".join(uris[0].split("/")[:-1])
            names = sorted({name for uri in uris for name in refs[uri][1]})
            source = "; ".join(f"[{task_by_name[n].source}]({task_by_name[n].url})" for n in names)
            count = f"{len(uris):,} file{'s' if len(uris) != 1 else ''}"
            missing_rows += (f"\n| {count} in `{where}` | {source} | {NOT_HOSTED_FILE_TERMS[key]} | "
                             f"{FETCHED_FROM_FILES[key]} |")
    summary = []
    for prefix, label in MODALITIES:
        chosen = [task for task in tasks if task.name.split(".")[0] == prefix]
        if chosen and prefix != "loop":
            summary.append(f"| {label} | {len(chosen)} | {sum(_items(t) for t in chosen):,} |")
    files, size = sum(v[0] for v in hosted.values()), sum(v[1] for v in hosted.values())
    return f"""---
pretty_name: S1Arena-MM
license: other
license_name: per-source
license_link: LICENSE.md
task_categories:
- visual-question-answering
- image-classification
- video-classification
- audio-classification
language:
- en
tags:
- benchmark
- calibration
- multimodal
- image
- video
- audio
size_categories:
- 10K<n<100K
datasets:
- betterup/candor-full-duplex-bench
viewer: false
{GATE}---

# S1Arena-MM

### Which multimodal decisions can be left to System 1?

S1Arena-MM measures how well models make discrete decisions about images, videos and sounds, and how good their
probabilities are. Every item holds an input and one or more typed questions, and every model answers each question
with a probability distribution over its options. This dataset holds the media the items reference: {files:,} files,
{size / 1e9:.1f} GB. The items, the code and the results are in the S1Arena-MM repository on GitHub.

| Modality | Tasks | Items |
|---|---|---|
{chr(10).join(summary)}

## Use

Request access on this page and log in with `hf auth login` (or set `HF_TOKEN`). Then, in a clone of the repository:

```bash
pip install -e '.[image,video,audio]'
export JEVARENA_MM_MEDIA=/path/to/media
jevarena-mm fetch       # this dataset's folders, and the sources it does not host from their publishers
jevarena-mm build       # rebuild the items shipped as recipes
jevarena-mm verify      # every file present, sha256 as the items record
```

A file an item references as `<folder>/<path>` is at `<folder>/<path>` here. One folder without the code:

```bash
hf download {REPO} --repo-type dataset --include 'mmmu/*' --local-dir /path/to/media
```

## Licenses

Each source keeps its own license; check it before you use a source (the tables below, the task cards, and
`meta.license` on each item). The code of S1Arena-MM is Apache-2.0. Media that S1Arena-MM generates are licensed
CC BY 4.0. `LICENSES/` holds the license texts some sources ask to keep with their files, and per-file credits for the
photos and videos that keep their own Creative Commons licenses. Full-Duplex-Bench v1 uses the CANDOR Corpus from
BetterUp Labs, BetterUp, Inc. ([betterup/candor-tabular](https://huggingface.co/datasets/betterup/candor-tabular)).

Several sources allow non-commercial research use only. Most items come from the test or validation splits of public
datasets; do not train on them. Some tasks contain offensive or disturbing content by design (safety sources), and many
files show or record people.

### Hosted folders

| Folder | Modality | Source | License | Files | Size |
|---|---|---|---|---|---|
{hosted_rows}

### Not hosted

These sources' terms do not allow redistribution, or allow it only to people who accepted them. `jevarena-mm fetch`
gets them from their owners or distributors, with your own access where they ask for it.

| Files | Source | Terms | `fetch` gets them from |
|---|---|---|---|
{missing_rows}

## Citation

```bibtex
@misc{{s1arenamm2026,
  title  = {{S1Arena-MM: Which Multimodal Decisions Can Be Left to System 1?}},
  author = {{Lin, Lang and Chen, Tianyu and Gu, Yi and Zhao, Yang and Song, Xinran and Liu, Xinyang and Cheng, Jianpeng
            and Zhou, Mingyuan}},
  year   = {{2026}}
}}
```
"""
