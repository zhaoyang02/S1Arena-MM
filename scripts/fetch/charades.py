"""Charades videos (Sigurdsson et al., 2016) from AI2's public 480p archive, Charades_v1_480.zip (16 GB).

Terms: the Charades license (non-commercial research; https://prior.allenai.org/projects/charades) does not let
S1Arena-MM redistribute the videos. videos/<id>.mp4 is the archive member Charades_v1_480/<id>.mp4, unchanged; each is
read on its own with HTTP range requests, several at a time (about 0.9 GB for all the items need). Nothing to supply.

The Charades-STA annotations (annotations/charades_sta_{test,train}.txt) that the recipes of lvid.ret/moment/charades_sta
and lvid.route/escalation/jev_pair are filled from: the authors' download link (github.com/jiyanggao/TALL) is gone, so
they come from the copy in VSLNet's repository, at a pinned commit and checked by sha256.
"""

import sys
import urllib.request
from pathlib import Path

from _common import missing, report, save, save_all, wanted, zip_jobs

URL = ("https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
       "?versionId=2I0cX8bvsoqLbtsk6mi6Wrb0d0uV1H8E")  # the S3 object version the items were built from
STA = "https://raw.githubusercontent.com/26hzhang/VSLNet/acf2d4c2274f7840e48441a01d1833d2482a99f8/data/dataset/charades/{}"


def main(dest: Path) -> None:
    files = wanted("charades")
    todo = missing(dest, files)
    for rel in [rel for rel in todo if rel.startswith("annotations/")]:
        with urllib.request.urlopen(STA.format(rel.removeprefix("annotations/")), timeout=60) as response:
            save(response, dest / rel, todo.pop(rel))
        print(f"charades: {rel}", flush=True)
    if todo:
        members = {"Charades_v1_480/" + rel.removeprefix("videos/"): (rel, sha) for rel, sha in sorted(todo.items())}
        save_all(zip_jobs(URL, None, members, dest), "charades")
    report("charades", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
