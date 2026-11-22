"""hateful_memes: the Hateful Memes test_seen images and test_seen.jsonl, from a public copy or from your own.

Terms: the Hateful Memes Dataset License forbids distributing or hosting the dataset (section 6). It was obtained by
accepting that license on DrivenData's competition (sign-ups closed in 2021) and then on Meta's
hatefulmemeschallenge.com (offline since 2024); no official source is left. The items were built from the copy a user
put on Hugging Face, neuralcatcher/hateful_memes, and the script reads the same files from it at a pinned revision. It
is not an official release, and you use the files under the license's terms. With HATEFUL_MEMES_DIR set to your own
copy (the folder that holds img/ and test_seen.jsonl) it reads that instead. A file is img/<id>.png, kept if its sha256
is the one the items record; the task is a recipe filled from test_seen.jsonl, kept as hateful_memes/test_seen.jsonl.
"""

import os
import sys
import urllib.request
from functools import partial
from pathlib import Path

from _common import hf_url, missing, report, save, save_all, wanted

REPO, REVISION = "neuralcatcher/hateful_memes", "d201c488dc7024623d1ecbcc987b3f132c4c2e12"


def source_name(rel: str) -> str:
    """The file's path in a copy of the dataset: test_seen.jsonl, or img/<id>.png."""
    return rel if rel == "test_seen.jsonl" else "img/" + rel.rsplit("/", 1)[-1]


def main(dest: Path) -> None:
    files = wanted("hateful_memes")
    root = os.environ.get("HATEFUL_MEMES_DIR")

    def get(rel: str, sha: str) -> None:  # save_all retries a dropped connection, not a wrong sha256 or a missing file
        if root:
            stream = open(Path(root) / source_name(rel), "rb")
        else:
            request = urllib.request.Request(hf_url(REPO, source_name(rel), REVISION),
                                             headers={"User-Agent": "jevarena-mm-fetch"})
            stream = urllib.request.urlopen(request, timeout=120)
        with stream:
            save(stream, dest / rel, sha)

    save_all([(rel, partial(get, rel, sha)) for rel, sha in sorted(missing(dest, files).items())], "hateful_memes")
    report("hateful_memes", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
