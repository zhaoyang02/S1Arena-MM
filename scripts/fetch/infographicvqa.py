"""infographicvqa: the InfographicVQA validation images, from your own copy of the dataset.

Terms: InfographicVQA is downloaded from the RRC portal after registration (https://rrc.cvc.uab.es/?ch=17). Download
its images (one archive holds every split), unpack them and set INFOGRAPHICVQA_DIR to that folder; the validation
images the items use are found under it by name (<id>.jpeg). A file is kept if its sha256 is the one the items record
(the items were built from the lmms-lab/DocVQA copy on Hugging Face, whose files are the RRC ones byte for byte).
Without INFOGRAPHICVQA_DIR, or when it is not a folder, the script says what to do and exits with status 2.
"""

import os
import sys
from pathlib import Path

from _common import missing, report, save, wanted


def main(dest: Path) -> None:
    files = wanted("infographicvqa")
    todo = missing(dest, files)
    root = os.environ.get("INFOGRAPHICVQA_DIR")
    if todo and not (root and Path(root).is_dir()):
        print("infographicvqa: register at https://rrc.cvc.uab.es/?ch=17, download the InfographicVQA images (one "
              "archive holds every split), unpack them and set INFOGRAPHICVQA_DIR to that folder"
              + (f" ({root} is not a folder)" if root else ""))
        sys.exit(2)
    names = {path.name: path for path in Path(root).rglob("*") if path.is_file()} if todo else {}
    for rel, sha in sorted(todo.items()):
        source = names.get(rel.rsplit("/", 1)[-1])
        if source:
            try:
                with open(source, "rb") as stream:
                    save(stream, dest / rel, sha)
                print(f"infographicvqa: {rel}", flush=True)
            except ValueError as error:
                print(f"infographicvqa: {error}", flush=True)
    report("infographicvqa", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
