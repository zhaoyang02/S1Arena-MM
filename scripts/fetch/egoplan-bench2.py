"""EgoPlan-Bench2 reference clips, from the Hugging Face dataset TencentARC/SEED-Bench-R1 at a pinned revision.

Terms: the clips are cut from Ego4D videos, so the Ego4D license agreement applies (https://ego4d-data.org/).
Mapping: a file's path is <SEED-Bench-R1 revision>/<member>; egoplan-bench2_files.csv names the gzipped tar archive
(videos/Ego4D/batch_<n>.tar.gz) that holds the member, the clip the EgoPlan-Bench2 authors confirmed as their reference
(https://github.com/qiulu66/EgoPlan-Bench2/issues/4). A tar.gz cannot be read selectively, so every archive holding a
missing file is streamed until its last needed member, writing only those members: all 500 files (7.7 GB) stream
about 22 GB of the 34 archives' 22.7 GB.
You need an Ego4D license (https://ego4d-data.org/docs/start-here/) to use them.
"""

import csv
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from _common import hf_url, missing, report, save, wanted

REPO = "TencentARC/SEED-Bench-R1"


def main() -> None:
    dest = Path(sys.argv[1])
    files = wanted("egoplan-bench2")
    with open(Path(__file__).with_name("egoplan-bench2_files.csv"), newline="") as f:
        archive_of = {row["path"]: row["archive"] for row in csv.DictReader(f)}
    todo = defaultdict(dict)  # (revision, archive) -> {member: path}
    for rel in missing(dest, files):
        if rel in archive_of:
            revision, member = rel.split("/", 1)
            todo[revision, archive_of[rel]][member] = rel
    if todo:
        print("egoplan-bench2: clips cut from Ego4D; the Ego4D license agreement applies (https://ego4d-data.org/)")
    for (revision, archive), members in sorted(todo.items()):
        print(f"egoplan-bench2: {len(members)} files from {archive}")
        try:
            with urllib.request.urlopen(hf_url(REPO, archive, revision), timeout=300) as response, \
                    tarfile.open(fileobj=response, mode="r|gz") as tar:
                for member in tar:
                    rel = members.pop(member.name, None)
                    if rel:
                        save(tar.extractfile(member), dest / rel, files[rel])
                    if not members:
                        break
        except (OSError, tarfile.TarError, ValueError) as error:
            print(f"{archive}: {type(error).__name__}: {error}")
    report("egoplan-bench2", files, dest)


if __name__ == "__main__":
    main()
