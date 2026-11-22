"""Ego4D clips and full-scale videos, from Ego4D's S3 buckets with your Ego4D license keys.

Terms: Ego4D license agreement (https://ego4d-data.org/); not redistributable, so not in the Hugging Face dataset.
Mapping: ego4d_files.csv (next to this script) gives each file's Ego4D dataset (clips, full_scale or the v2 annotations
that recipes are rebuilt from), uid and S3 object;
the file name is the sha256 of the JSON-quoted S3 location. The Ego4D manifests carry no checksums: the files match
the objects' S3 ETags, and each download is checked here against the sha256 the items record.
You supply: an approved Ego4D license (https://ego4d-data.org/docs/start-here/), the AWS keys from its email set up as
for the Ego4D CLI (`aws configure`; AWS_PROFILE picks another profile), and `pip install boto3`. Files you already have
from the official CLI (`ego4d --output_directory DIR --datasets clips --video_uid_file UIDS`, one dataset per run) are
copied from DIR/v2/<dataset>/<uid>.mp4 when EGO4D_DIR=DIR is set.
"""

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _common import missing, report, save, wanted

HELP = """ego4d: these files need an Ego4D license and the AWS keys it comes with.
  1. Sign the license at https://ego4d-data.org/docs/start-here/ and wait for the approval email.
  2. pip install boto3; aws configure  (enter the email's keys; or set AWS_PROFILE to the profile that has them)
  3. Rerun. With an existing Ego4D CLI download, set EGO4D_DIR to its --output_directory to copy from it instead."""
DENIED = {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken", "403"}


def refuse(reason: str = "") -> None:
    print(reason + HELP)
    sys.exit(2)


def main() -> None:
    dest = Path(sys.argv[1])
    files = wanted("ego4d")
    with open(Path(__file__).with_name("ego4d_files.csv"), newline="") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}
    need = {rel: sha for rel, sha in missing(dest, files).items() if rel in rows}
    if os.environ.get("EGO4D_DIR"):
        for rel, sha in list(need.items()):
            name = Path(rel).name if rows[rel]["dataset"] == "annotations" else rows[rel]["uid"] + ".mp4"
            src = Path(os.environ["EGO4D_DIR"], "v2", rows[rel]["dataset"], name)
            if src.is_file():
                try:
                    with open(src, "rb") as f:
                        save(f, dest / rel, sha)
                    del need[rel]
                except ValueError as error:
                    print(error)
    if need:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:
            refuse("ego4d: boto3 is not installed.\n")
        try:
            session = boto3.Session()
        except BotoCoreError as error:  # e.g. AWS_PROFILE names a profile that the AWS config does not have
            refuse(f"ego4d: {error}.\n")
        if session.get_credentials() is None:
            refuse("ego4d: no AWS credentials found.\n")
        s3 = session.client("s3")

        def get(rel: str) -> None:
            bucket, key = rows[rel]["s3_path"].removeprefix("s3://").split("/", 1)
            try:
                save(s3.get_object(Bucket=bucket, Key=key)["Body"], dest / rel, need[rel])
            except ClientError as error:  # print only the code: messages can carry request details
                code = error.response.get("Error", {}).get("Code")
                if code in DENIED:
                    raise PermissionError(code) from None
                print(f"{rel} ({rows[rel]['uid']}): S3 error {code}")
            except (BotoCoreError, OSError, ValueError) as error:
                print(f"{rel} ({rows[rel]['uid']}): {type(error).__name__}: {error}")

        print(f"ego4d: downloading {len(need)} files from Ego4D's S3 buckets")
        try:
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(get, sorted(need)))
        except PermissionError as error:
            refuse(f"ego4d: S3 refused the keys ({error}).\n")
    report("ego4d", files, dest)


if __name__ == "__main__":
    main()
