"""SoccerNet-MVFoul test clips and annotations (Held et al., 2023) from SoccerNet's 720p release on Hugging Face, pinned
at REVISION.

Terms: SoccerNet's README asks for its NDA before the data and annotations are downloaded
(https://github.com/SoccerNet/sn-mvfoul), so S1Arena-MM hosts neither the clips nor the answers (the task is a recipe).
This upload (SoccerNet/SN-MVFouls-2024, 2026-09-07) is public, GPL-3.0 on its card and not encrypted, so no password is
needed. videos/action_<N>/clip_<K>.mp4 is the member action_<N>/clip_<K>.mp4 of test_720p.zip and annotations.json its
member annotations.json, unchanged. They are 1.3 GB of the 1.4 GB archive, so it is downloaded whole with
huggingface_hub (much faster than range reads) and deleted afterwards. Nothing to supply.
"""

import sys
import urllib.error
from pathlib import Path

from _common import missing, report, save_hf_zip_members, wanted

REPO, REVISION = "SoccerNet/SN-MVFouls-2024", "fdd521d146977bc58382eac754489469b147df19"


def main(dest: Path) -> None:
    files = wanted("soccernet-mvfoul")
    todo = missing(dest, files)
    if todo:
        members = {rel.removeprefix("videos/"): (rel, sha) for rel, sha in sorted(todo.items())}
        try:
            save_hf_zip_members(REPO, "test_720p.zip", REVISION, members, dest, "soccernet-mvfoul")
        except urllib.error.HTTPError as error:
            print(f"{REPO}@{REVISION[:7]}: HTTP {error.code}. If SoccerNet now gates it, accept its terms on "
                  f"https://huggingface.co/datasets/{REPO} and log in with `hf auth login`; the NDA and other "
                  "downloads are on https://www.soccer-net.org/data.", file=sys.stderr)
            sys.exit(2)
    report("soccernet-mvfoul", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
