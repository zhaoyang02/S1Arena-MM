"""vsi-bench: the ScanNet and ScanNet++ scene videos (hub.NOT_HOSTED), from VSI-Bench's release nyu-visionx/VSI-Bench.

Terms: ScanNet (http://www.scan-net.org/) and ScanNet++ (https://kaldir.vc.in.tum.de/scannetpp/) terms of use, which
their owners ask every user to sign. <dir>/<dataset>_<scene>.mp4 (dir is acquired-v2, batch-native-cache-v1 or videos,
the directories the files were first saved in) is member <dataset>/<scene>.mp4 of <dataset>.zip at the pinned
revision, byte for byte. scannetpp.zip is mostly wanted videos, so it is downloaded whole with huggingface_hub (much
faster than range reads) and deleted afterwards; the wanted members of scannet.zip are read over HTTP ranges.
"""

import sys
from collections import defaultdict
from pathlib import Path

from _common import missing, report, save_hf_zip_members, wanted

REPO, REVISION = "nyu-visionx/VSI-Bench", "bdcadb3fea447621a828a24911801faba3587c12"


def main(dest: Path) -> None:
    files = wanted("vsi-bench")
    archives = defaultdict(dict)
    for rel, sha in missing(dest, files).items():
        dataset, scene = Path(rel).stem.split("_", 1)
        archives[dataset][f"{dataset}/{scene}.mp4"] = (rel, sha)
    for dataset, members in archives.items():
        save_hf_zip_members(REPO, f"{dataset}.zip", REVISION, members, dest, "vsi-bench")
        for member in members:
            print(f"vsi-bench: {member} is not in {dataset}.zip", flush=True)
    report("vsi-bench", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
