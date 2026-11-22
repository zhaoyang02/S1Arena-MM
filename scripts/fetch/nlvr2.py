"""nlvr2: the NLVR2 balanced public-test images, from the lmms-lab/NLVR2 release on Hugging Face.

Terms: NLVR2's authors do not license its images (web images whose copyright they do not hold) and share them through
a form whose Terms of Service let a researcher give access only to associates and colleagues who accept them (term 4;
https://github.com/lil-lab/nlvr), so each user fetches them. A file is the left_image or right_image bytes of a row of the release, byte for byte, found
by sha256.
"""

import sys
from pathlib import Path

from _common import by_sha, hf_parquet_images, missing, report, wanted

REPO, REVISION = "lmms-lab/NLVR2", "a101e1114e9e7811074a6a2f618c3b8d85149e1d"
FILES = ["data/balanced_test_public-00000-of-00002-c989ad9eb1dd7008.parquet",
         "data/balanced_test_public-00001-of-00002-f9a6d537d4449440.parquet"]


def main(dest: Path) -> None:
    files = wanted("nlvr2")
    hf_parquet_images(REPO, REVISION, FILES, ["left_image", "right_image"], by_sha(missing(dest, files)), dest, "nlvr2")
    report("nlvr2", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
