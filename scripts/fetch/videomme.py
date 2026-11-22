"""videomme: Video-MME videos from lmms-lab/Video-MME on Hugging Face, the release the Video-MME authors link to.

Terms: Video-MME is for academic research only and may not be redistributed without the authors' approval
(https://github.com/MME-Benchmarks/Video-MME), so each user fetches it from that release. videos/<id>.mp4 is member
data/<id>.mp4 of one of videos_chunked_01..20.zip at the pinned revision, byte for byte. The wanted videos are 81 of
the archives' 101 GB, so each archive (about 5 GB) is downloaded whole with huggingface_hub, which is much faster than
range reads, and deleted once its members are saved. annotations/<file> is the release's videomme/<file>, the
questions that `jevarena-mm build` fills into the recipe of lvid.know/long_qa/video_mme and lvid.route/escalation.
"""

import sys
import urllib.request
from pathlib import Path

from _common import hf_url, missing, report, save, save_hf_zip_members, wanted

REPO, REVISION = "lmms-lab/Video-MME", "ead1408f75b618502df9a1d8e0950166bf0a2a0b"


def main(dest: Path) -> None:
    files = wanted("videomme")
    todo = {"data/" + rel.removeprefix("videos/"): (rel, sha) for rel, sha in missing(dest, files).items()}
    for member in [m for m in todo if m.startswith("data/annotations/")]:
        rel, sha = todo.pop(member)
        with urllib.request.urlopen(hf_url(REPO, "videomme/" + rel.removeprefix("annotations/"), REVISION), timeout=300) as r:
            save(r, dest / rel, sha)
        print(f"videomme: {rel}", flush=True)
    for chunk in range(1, 21):
        if not todo:
            break
        save_hf_zip_members(REPO, f"videos_chunked_{chunk:02d}.zip", REVISION, todo, dest, "videomme")
    report("videomme", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
