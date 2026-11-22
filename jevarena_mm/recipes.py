"""Items whose text or labels come from annotations that may not be redistributed are committed as recipes.

A recipe line is {"id": item id, "item": the item with each such string replaced by a token ⟦source:field:key⟧}. The
user fetches the source's annotation file with the media, under the source's own terms (`jevarena-mm fetch`), and
`jevarena-mm build` fills the tokens and keeps an item only if its content id equals the recipe's, so a rebuilt item is
the original byte for byte. Media references are never tokenized, so `fetch` can read them from the recipe.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from . import dataio, env
from .schema import Item, MediaRef

TOKEN = re.compile(r"⟦([a-z0-9_]+):([a-z_]+):([^⟧]*)⟧")
QUARTERS = ("first_quarter", "second_quarter", "third_quarter", "fourth_quarter")


@dataclass(frozen=True)
class Source:
    path: str  # under $JEVARENA_MM_MEDIA; written by the fetch script of its folder
    sha256: str | None  # None: not pinned (obtaining the file means accepting its license); each rebuilt id checks it
    terms: str
    format: str = "upstream"


SOURCES = {
    "videomme": Source("videomme/annotations/test-00000-of-00001.parquet",
                       "7fffab8ed38ecc2f9f0eca4c44d8a11636f0ee96116ede83580c8b9ae0faf986",
                       "Video-MME's release lmms-lab/Video-MME (public; its terms forbid redistribution)"),
    "ego4d_nlq": Source("ego4d/annotations/nlq_val.json",
                        "20413901b8dbfcb097e7219da6b3a577ebb1c309f8e6667a0f53fdd2ae0d6da7",
                        "Ego4D v2 annotations (Ego4D license)"),
    "ego4d_oscc": Source("ego4d/annotations/fho_oscc-pnr_val.json",
                         "bba6413f5b483af2cc8f411077c45d8863881c458e4966b26c314531314d8173",
                         "Ego4D v2 annotations (Ego4D license)"),
    # Charades-STA's "[video] [start] [end]##[sentence]" files; keys are line numbers from 0. The authors' download link
    # (github.com/jiyanggao/TALL) is gone, so scripts/fetch/charades.py reads the copy in VSLNet's repository.
    "charades_sta": Source("charades/annotations/charades_sta_test.txt",
                           "fab341679c20b0917173dcffffde367a71113fe63f3a645558cef2ebfbfefb55",
                           "Charades-STA test annotations (Gao et al., 2017), on Charades videos (Charades license: no "
                           "distribution to third parties)"),
    "charades_sta_train": Source("charades/annotations/charades_sta_train.txt",
                                 "e966fc1970cf6bc4cc73db2dd4fc10e148313810a372eef488defb0dd2ee359f",
                                 "Charades-STA train annotations (Gao et al., 2017), on Charades videos (Charades "
                                 "license: no distribution to third parties)"),
    # annotations.json of test_720p.zip in SoccerNet/SN-MVFouls-2024 at fdd521d; keys are action ids ("283").
    "soccernet_mvfoul": Source("soccernet-mvfoul/annotations.json",
                               "5d09741f9e76867945511002ffa360e2ae41af06d471ef5fc5fec770d1bdcf3c",
                               "SoccerNet-MVFoul annotations (SoccerNet shares its data and annotations under an NDA: "
                               "https://github.com/SoccerNet/sn-mvfoul)"),
    # data/examples.jsonl of facebook/winoground at b400e17 (id, caption_0, caption_1, ...), read with the user's own
    # access after accepting its license; keys "<id>/<image 0 or 1>/<caption order 01 or 10>/<option keys t or i>".
    "winoground": Source("winoground/examples.jsonl",
                         "055f68668e60bb9bac18ec37dcbb40b935690302011e3f8f53767f0b68a65911",
                         "Winoground license agreement (Meta): no distributing or hosting its images or their metadata"),
    # test_seen.jsonl of Hateful Memes ({"id", "img", "label", "text"} per line); keys are the image's file name without
    # .png. scripts/fetch/hateful_memes.py reads it from neuralcatcher/hateful_memes (pinned there) or the user's own
    # copy; not pinned here, as another copy of the file may differ in bytes. Every filled item is checked by its id.
    "hateful_memes": Source("hateful_memes/test_seen.jsonl", None,
                            "Hateful Memes Dataset License (DrivenData): no distributing or hosting (section 6)"),
    # ArtEmis 2.0's contrastive CSV, which ArtEmis's request form leads to (columns emotion, utterance, art_style,
    # painting, anchor_art_style, anchor_painting, repetition); keys are "<art_style>/<painting>".
    "artemis": Source("wikiart/artemis.csv", "240891587ac8d6042378a4f3d5931e198e3484fb1d44c284a912636dbd189b6e",
                      "ArtEmis terms of use: non-commercial research and education, shared only with people who accept "
                      "them"),
    # labels/validation.json of Qualcomm's label package: [{"id", "label", "template", "placeholders"}]; keys are video
    # ids. Downloading it accepts the Something-Something v2 license.
    "ssv2": Source("ssv2/annotations/validation.json",
                   "fbdd69ac72b40b9e3ed3aaf484e1575777152565caed693b0ced2d4959fd3958",
                   "Something-Something v2 labels (Qualcomm data license, research use; no distribution to third "
                   "parties)"),
}

# vid.pro/foul/soccernet_mvfoul: the action classes in the order of the options action_0 ... action_7
SOCCERNET_CLASSES = ("Tackling", "Standing tackling", "High leg", "Holding", "Pushing", "Elbowing", "Challenge", "Dive")

# img.social/art_emotion/artemis: the eight emotions it asks about ("something else" is not an option)
ARTEMIS_EMOTIONS = ("amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness")

# vid.basic/temporal/ssv2: the template families it asks about, each with its two directions in a fixed order (a, b),
# and the task's wording of the scene, the questions and the options.
SSV2_TEMPLATES = {
    "pushing something from left to right": ("push", "left_to_right"),
    "pushing something from right to left": ("push", "right_to_left"),
    "pulling something from left to right": ("pull", "left_to_right"),
    "pulling something from right to left": ("pull", "right_to_left"),
    "moving something up": ("updown", "up"),
    "moving something down": ("updown", "down"),
    "putting something into something": ("container", "into"),
    "taking something out of something": ("container", "out_of"),
    "moving something towards the camera": ("camera", "towards_camera"),
    "moving something away from the camera": ("camera", "away_camera"),
}
SSV2_FAMILIES = {
    "push": {"pair": ("left_to_right", "right_to_left"),
             "scene": "Observe the main pushing action in the clip. Use image coordinates.",
             "choice": "During the main pushing action, which direction does the object move?",
             "options": {"left_to_right": "From left to right.", "right_to_left": "From right to left."},
             "yes_no": {"left_to_right": "During the main pushing action, does the object move from left to right?",
                        "right_to_left": "During the main pushing action, does the object move from right to left?"}},
    "pull": {"pair": ("left_to_right", "right_to_left"),
             "scene": "Observe the main pulling action in the clip. Use image coordinates.",
             "choice": "During the main pulling action, which direction does the object move?",
             "options": {"left_to_right": "From left to right.", "right_to_left": "From right to left."},
             "yes_no": {"left_to_right": "During the main pulling action, does the object move from left to right?",
                        "right_to_left": "During the main pulling action, does the object move from right to left?"}},
    "updown": {"pair": ("up", "down"),
               "scene": "Observe the main movement of the object in the clip.",
               "choice": "During the main movement, is the object raised or lowered?",
               "options": {"down": "The object is lowered.", "up": "The object is raised."},
               "yes_no": {"up": "During the main movement, is the object raised?",
                          "down": "During the main movement, is the object lowered?"}},
    "container": {"pair": ("into", "out_of"),
                  "scene": "Observe the main transfer of an object relative to a container in the clip.",
                  "choice": "During the main transfer, how does the object's location change?",
                  "options": {"out_of": "From inside the container to outside it.",
                              "into": "From outside the container to inside it."},
                  "yes_no": {"into": "During the main transfer, does the object go from outside the container to inside "
                                     "it?",
                             "out_of": "During the main transfer, does the object go from inside the container to "
                                       "outside it?"}},
    "camera": {"pair": ("towards_camera", "away_camera"),
               "scene": "Observe the main movement of the object relative to the camera in the clip.",
               "choice": "During the main movement, how does the object's distance from the camera change?",
               "options": {"towards_camera": "It becomes closer to the camera.",
                           "away_camera": "It becomes farther from the camera."},
               "yes_no": {"towards_camera": "During the main movement, does the object become closer to the camera?",
                          "away_camera": "During the main movement, does the object become farther from the camera?"}},
}


def ssv2_template(template: str) -> tuple[str, str]:
    """(family, direction) of an SSv2 template such as "Pushing [something] from left to right"."""
    key = " ".join(template.replace("[", "").replace("]", "").lower().split())
    if key not in SSV2_TEMPLATES:
        raise KeyError(f"SSv2 template {template!r} is not one vid.basic/temporal/ssv2 asks about")
    return SSV2_TEMPLATES[key]

# Local audio delivery files, pinned in tasks/aud/export.json; annotation contents are never committed.
_audio_export = Path(__file__).parent / "tasks" / "aud" / "export.json"
if _audio_export.exists():
    SOURCES.update({name: Source(**spec) for name, spec in
                    json.loads(_audio_export.read_text(encoding="utf-8"))["sources"].items()})


def path(task) -> Path:
    """data/<...>/<source>.recipe.jsonl, committed next to where the items of a redistributable task would be."""
    return dataio.task_path(task.name).with_suffix(".recipe.jsonl")


def load(task) -> list[dict]:
    file = path(task)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_media(row: dict) -> list[MediaRef]:
    """The media references of one recipe line, in state order (tokens never appear in them)."""
    refs = []
    for value in row["item"]["request"]["state"].values():
        for entry in value if isinstance(value, list) else [value]:
            if isinstance(entry, dict) and "uri" in entry:
                refs.append(MediaRef.model_validate(entry))
    return refs


def media_refs(task) -> list[MediaRef]:
    """Every media reference of the recipe's items, in state order."""
    return [ref for row in load(task) for ref in row_media(row)]


def media_of(task) -> list[MediaRef]:
    """The media references of a task: from its recipe when it has one (complete even when a build could only rebuild
    some items, e.g. before the user had the source's keys), else from its built items."""
    refs = media_refs(task)
    if refs:
        return refs
    return [ref for item in dataio.load_task(task) for _, ref in item.request.media()]


def sources_of(row: dict) -> set[str]:
    return {match[0] for match in TOKEN.findall(json.dumps(row["item"], ensure_ascii=False))}


def source_file(name: str) -> Path:
    return env.media_dir() / SOURCES[name].path


def available(name: str) -> bool:
    from .media import sha256_file

    file = source_file(name)
    return file.is_file() and SOURCES[name].sha256 in (None, sha256_file(file))


@cache
def _table(name: str, file: Path, revision: str = "") -> dict:
    if SOURCES[name].format == "jsonl":
        from .audio_delivery import read_index

        return read_index(file)
    if name == "videomme":
        import pyarrow.parquet as pq

        return {row["question_id"]: row for row in pq.read_table(file).to_pylist()}
    if name == "ssv2":
        return {row["id"]: row for row in json.loads(file.read_text(encoding="utf-8"))}
    if name == "artemis":  # the annotators' emotions per painting, "something else" left out
        import csv
        from collections import Counter

        counts: dict[str, Counter] = {}
        with open(file, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["emotion"] in ARTEMIS_EMOTIONS:
                    counts.setdefault(f"{row['art_style']}/{row['painting']}", Counter())[row["emotion"]] += 1
        return counts
    if name == "soccernet_mvfoul":
        return json.loads(file.read_text(encoding="utf-8"))["Actions"]
    if name == "winoground":
        return {str(row["id"]): row for row in map(json.loads, file.read_text(encoding="utf-8").splitlines()) if row}
    if name == "hateful_memes":
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {Path(row["img"]).stem: row for row in rows}
    if name.startswith("charades_sta"):
        rows = {}
        for n, line in enumerate(file.read_text(encoding="utf-8").splitlines()):
            head, _, sentence = line.partition("##")
            video, start, end = head.split()
            rows[str(n)] = {"video": video, "window": (float(start), float(end)), "sentence": sentence}
        return rows
    data = json.loads(file.read_text(encoding="utf-8"))
    if name == "ego4d_nlq":
        return {f"{annotation['annotation_uid']}#{n}": query
                for video in data["videos"] for clip in video["clips"] for annotation in clip["annotations"]
                for n, query in enumerate(annotation["language_queries"])}
    return {record["unique_id"]: record for record in data["clips"]}


def quarter(windows: list[tuple[float, float]], duration: float) -> str:
    """The quarter of [0, duration) that holds every window (the gold of lvid.ret/moment)."""
    bins = [(duration * k / 4, duration * (k + 1) / 4) for k in range(4)]
    hits = [k for k, (left, right) in enumerate(bins) if all(left - 1e-8 <= a < b <= right + 1e-8 for a, b in windows)]
    if len(hits) != 1:
        raise ValueError(f"windows {windows} are not inside one quarter of {duration}")
    return QUARTERS[hits[0]]


def value(source: str, field: str, key: str) -> Any:
    """A token's value. JSONL delivery sources can also supply whole questions, gold and provenance objects."""
    table = _table(source, source_file(source), SOURCES[source].sha256)
    if SOURCES[source].format == "jsonl":
        row = table[key]
        if field == "record":
            return deepcopy(row)
        if field == "questions":
            return deepcopy(row["request"]["questions"])
        return deepcopy(row[field])
    if source == "artemis":  # dist: share of each emotion among the painting's annotations; label: the most frequent
        counts, emotions = table[key], sorted(ARTEMIS_EMOTIONS)
        total = sum(counts.values())
        return ({e: counts[e] / total for e in emotions} if field == "dist"
                else max(emotions, key=lambda e: counts[e]))
    if source == "soccernet_mvfoul":  # the answers the annotation of an action gives
        action = table[key]
        if field == "action":
            return f"action_{SOCCERNET_CLASSES.index(action['Action class'])}"
        if field == "severity":  # severity 1-5 is option 0-4
            return str(int(float(action["Severity"])) - 1)
        offence = {"Offence": True, "No offence": False}[action["Offence"]]
        return "true" if offence == (field == "offence") else "false"  # field "offence" or "no_offence"
    if source == "winoground":  # the two captions as options in the given order, keyed by their text (t) or by the ids
        example, image, order, style = key.split("/")  # caption_1 / caption_2 (i); the answer is image i's caption i
        texts = [table[example][f"caption_{i}"] for i in order]
        keys = texts if style == "t" else ["caption_1", "caption_2"]
        answer = keys[texts.index(table[example][f"caption_{image}"])]
        return dict(zip(keys, texts)) if field == "options" else answer
    if source == "hateful_memes":  # the meme's text (surrounding spaces dropped, as the items were built), and
        row = table[key]  # whether it is hateful (label 1)
        return row["text"].strip() if field == "text" else ("true" if int(row["label"]) == 1 else "false")
    if source == "ssv2":  # key: video id, or "<id>/a" or "<id>/b" for the yes/no question about direction a or b
        video, _, which = key.partition("/")
        family, direction = ssv2_template(table[video]["template"])
        words = SSV2_FAMILIES[family]
        asked = words["pair"][which == "b"] if which else None
        return {"scene": words["scene"], "choice_question": words["choice"], "options": dict(words["options"]),
                "direction": direction, "yes_no_question": asked and words["yes_no"][asked],
                "yes_no_answer": asked and ("true" if direction == asked else "false")}[field]
    if source.startswith("charades_sta"):  # query: one line; windows / quarter: lines a+b (+ "@<duration>" for quarter)
        if field == "query":
            return table[key]["sentence"]
        keys, _, duration = key.partition("@")
        windows = [table[k]["window"] for k in keys.split("+")]
        return [list(w) for w in windows] if field == "windows" else quarter(windows, float(duration))
    if source == "videomme":
        question_id, _, index = key.partition("/")
        row = table[question_id]
        return row["options"][int(index)] if field == "option" else row[field]
    if source == "ego4d_nlq":
        if field == "query":
            return table[key]["query"]
        keys, _, duration = key.partition("@")
        return quarter([(table[k]["clip_start_sec"], table[k]["clip_end_sec"]) for k in keys.split("+")], float(duration))
    return "true" if table[key]["state_change"] else "false"


def fill(data: Any) -> Any:
    if isinstance(data, str):
        exact = TOKEN.fullmatch(data)
        if exact:
            return value(*exact.groups())
        return TOKEN.sub(lambda m: value(*m.groups()), data)
    if isinstance(data, list):
        return [fill(entry) for entry in data]
    if isinstance(data, dict):
        return {fill(k): fill(v) for k, v in data.items()}
    return data


def rebuild(task) -> tuple[list[Item], dict[str, int]]:
    """(items rebuilt from the recipe, {source: items left out because its annotation file is missing})."""
    items, missing = [], {}
    have = {name: available(name) for name in SOURCES}
    for row in load(task):
        lacking = sorted(name for name in sources_of(row) if not have[name])
        if lacking:
            for name in lacking:
                missing[name] = missing.get(name, 0) + 1
            continue
        item = Item.model_validate(fill(row["item"]))
        if item.id != row["id"]:
            raise ValueError(f"{task.name}: rebuilt item {item.id} differs from the recipe's {row['id']}")
        items.append(item)
    return items, missing
