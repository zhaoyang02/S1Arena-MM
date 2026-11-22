"""wikiart: the WikiArt images of the ArtEmis task, from WikiArt's image server.

Terms: ArtEmis restricts use to non-commercial research and lets a researcher share its data only with colleagues who
accept the same terms (https://www.artemisdataset.org/materials/artemis_terms_of_use.txt), so each user fetches the
images. Each file is downloaded from the URL its item gives for it (meta.media source_path; the item path is that URL
made file-safe) and kept if its sha256 is the one the items record.

The task is a recipe filled from ArtEmis 2.0's contrastive CSV, which ArtEmis shares through the request form on
https://www.artemisdataset-v2.org/ ("Dataset"): set ARTEMIS_CSV to the CSV you receive and the script copies it to
wikiart/artemis.csv, checked by sha256.
"""

import os
import sys
import urllib.request
from functools import partial
from pathlib import Path

from _common import missing, report, save, save_all, source_paths, wanted

HEADER = "emotion,utterance,art_style,painting,anchor_art_style,anchor_painting,repetition"


def main(dest: Path) -> None:
    files = wanted("wikiart")
    urls = source_paths("wikiart")
    todo = missing(dest, files)
    if "artemis.csv" in todo:  # without it the images are still fetched, and the report lists it as missing
        sha, csv = todo.pop("artemis.csv"), os.environ.get("ARTEMIS_CSV")
        if not csv:
            print("wikiart: request ArtEmis 2.0's annotations through its form (https://www.artemisdataset-v2.org/, "
                  "\"Dataset\"), accept its terms, and set ARTEMIS_CSV to the CSV you receive", flush=True)
        else:
            try:
                with open(csv, "rb") as stream:
                    save(stream, dest / "artemis.csv", sha)
                print("wikiart: artemis.csv", flush=True)
            except OSError as error:
                print(f"wikiart: ARTEMIS_CSV: {error}", flush=True)
            except ValueError:
                print(f"wikiart: ARTEMIS_CSV ({csv}) is not ArtEmis 2.0's contrastive CSV as its form's link gives it "
                      f"(its sha256 differs); that file's first line is {HEADER}", flush=True)

    def get(rel: str, sha: str) -> None:
        request = urllib.request.Request(urls.get(rel, "https://" + rel), headers={"User-Agent": "jevarena-mm-fetch"})
        with urllib.request.urlopen(request, timeout=120) as response:
            save(response, dest / rel, sha)

    save_all([(rel, partial(get, rel, sha)) for rel, sha in sorted(todo.items())], "wikiart")
    report("wikiart", files, dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
