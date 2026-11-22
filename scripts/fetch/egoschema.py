"""EgoSchema clips, from EgoSchema's own Google Drive files, else the lmms-lab/egoschema mirror on Hugging Face.

Terms: the clips are cut from Ego4D videos, so the Ego4D license agreement applies (https://ego4d-data.org/).
Mapping: a file is <q_uid>.mp4; egoschema_files.csv gives its Google Drive id (from questions.json of
github.com/egoschema/EgoSchema at 505c787) and the zip of the mirror (lmms-lab/egoschema at 5835052) that holds the
same bytes as videos/<q_uid>.mp4 (size and CRC32 checked). Clips are fetched several at a time; mirror members are read
with HTTP ranges, without downloading the 21 GB zips; every file is checked against the sha256 the items record.
You need an Ego4D license (https://ego4d-data.org/docs/start-here/) to use them.
"""

import csv
import sys
import urllib.request
from collections import defaultdict
from functools import partial
from pathlib import Path

from _common import missing, report, save, save_all, save_hf_zip_members, wanted

DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
MIRROR, MIRROR_REVISION = "lmms-lab/egoschema", "58350524ea7eb29c47000121f4f4b65eb6b4acb9"


def main() -> None:
    dest = Path(sys.argv[1])
    files = wanted("egoschema")
    with open(Path(__file__).with_name("egoschema_files.csv"), newline="") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}
    need = {rel: sha for rel, sha in missing(dest, files).items() if rel in rows}
    if need:
        print("egoschema: clips cut from Ego4D; the Ego4D license agreement applies (https://ego4d-data.org/)")

    def from_drive(rel: str, sha: str) -> None:
        with urllib.request.urlopen(DRIVE.format(rows[rel]["google_drive_id"]), timeout=300) as response:
            if response.headers.get_content_type() == "text/html":  # quota or error page, not the file
                raise ValueError("Google Drive returned a web page")
            save(response, dest / rel, sha)

    save_all([(rel, partial(from_drive, rel, sha)) for rel, sha in sorted(need.items())], "egoschema (Google Drive)",
             workers=4)
    left = missing(dest, need)
    if left:
        print(f"egoschema: {len(left)} clips from {MIRROR}", flush=True)
        members = defaultdict(dict)
        for rel, sha in sorted(left.items()):
            members[rows[rel]["mirror_zip"]]["videos/" + rel] = (rel, sha)
        for name, in_zip in sorted(members.items()):
            save_hf_zip_members(MIRROR, name, MIRROR_REVISION, in_zip, dest, f"egoschema ({MIRROR})")
    report("egoschema", files, dest)


if __name__ == "__main__":
    main()
