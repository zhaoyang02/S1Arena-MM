"""winoground: the Winoground test images, from facebook/winoground on Hugging Face with your token.

Terms: Winoground's license agreement (accepted on its gated page, https://huggingface.co/datasets/facebook/winoground)
forbids distributing or hosting its images and their metadata, captions included. Accept it there, then log in with
`hf auth login` (or set HF_TOKEN). A file is the image_0 or image_1 bytes of a test row, byte for byte, found by
sha256. The task is a recipe filled from data/examples.jsonl, which the script also downloads (winoground/examples.jsonl).
Without access the script says so and exits with status 2.
"""

import sys
from pathlib import Path

from _common import by_sha, hf_parquet_images, missing, report, save, wanted

REPO, REVISION = "facebook/winoground", "b400e173549071916ad1b3d449293bc8d8b4b763"
FILES = ["data/test-00000-of-00001.parquet"]


def main(dest: Path) -> None:
    files = wanted("winoground")
    if "examples.jsonl" in missing(dest, files):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError

        try:
            path = hf_hub_download(REPO, "data/examples.jsonl", repo_type="dataset", revision=REVISION)
        except GatedRepoError:
            print(f"winoground: accept the license on https://huggingface.co/datasets/{REPO}, then log in")
            sys.exit(2)
        with open(path, "rb") as stream:
            save(stream, dest / "examples.jsonl", files["examples.jsonl"])
    hf_parquet_images(REPO, REVISION, FILES, ["image_0", "image_1"], by_sha(missing(dest, files)), dest, "winoground")
    report("winoground", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
