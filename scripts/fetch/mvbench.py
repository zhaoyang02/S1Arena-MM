"""mvbench: the MVBench clips the Hugging Face dataset does not host (hub.NOT_HOSTED), from their distributors.

- star/..., sta/..., ssv2_video/..., Moments_in_Time_Raw/...: member <path> of video/<first part>.zip in
  OpenGVLab/MVBench at the pinned revision. STAR and Charades-STA clips are cut from Charades (Charades license,
  non-commercial research, https://prior.allenai.org/projects/charades); ssv2_video holds Something-Something v2 clips
  (its license, https://www.qualcomm.com/developer/software/something-something-v-2-dataset); Moments_in_Time_Raw holds
  Moments in Time clips, whose terms forbid distributing them or copying them onto a public site
  (http://moments.csail.mit.edu/).
- original-publisher/charades-official-pinned/<member>: a Charades video that MVBench's archives lack, member <member>
  of the official Charades_v1_480.zip from AI2 (pinned S3 object version); Charades license.
Each file is its archive member byte for byte; only the wanted members are read (HTTP ranges, several at a time).
"""

import sys
from collections import defaultdict
from pathlib import Path

from _common import missing, report, save_all, save_hf_zip_members, wanted, zip_jobs

REPO, REVISION = "OpenGVLab/MVBench", "230a2d4fac8900333c61754641c7a13e069ac9c6"
CHARADES_DIR = "original-publisher/charades-official-pinned/"
CHARADES = ("https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
            "?versionId=2I0cX8bvsoqLbtsk6mi6Wrb0d0uV1H8E")


def source(rel: str) -> tuple[str, str]:
    """(archive: its path in REPO, or the Charades URL; member) of the file at rel."""
    if rel.startswith(CHARADES_DIR):
        return CHARADES, rel.removeprefix(CHARADES_DIR)
    return f"video/{rel.split('/', 1)[0]}.zip", rel


def main(dest: Path) -> None:
    files = wanted("mvbench")
    archives = defaultdict(dict)
    for rel, sha in missing(dest, files).items():
        archive, member = source(rel)
        archives[archive][member] = (rel, sha)
    for archive, members in archives.items():
        if archive == CHARADES:
            save_all(zip_jobs(CHARADES, None, members, dest), "mvbench")
        else:
            save_hf_zip_members(REPO, archive, REVISION, members, dest, "mvbench")
        for member in members:
            print(f"mvbench: {member} is not in {archive.split('?')[0]}", flush=True)
    report("mvbench", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
