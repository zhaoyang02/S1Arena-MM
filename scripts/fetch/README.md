# Fetch scripts

`jevarena-mm fetch` downloads the hosted media from the Hugging Face dataset (`jevarena_mm/hub.py`: `REPO`). Media it
does not hold (`NOT_HOSTED`: whole folders, subfolders or file patterns; `NOT_HOSTED_TASKS`: every file of a task, for
folders whose files are named by content hash; `NOT_HOSTED_FILES`: single files of hosted folders) come from one script
per folder, `scripts/fetch/<folder>.py`, which gets them from their owner or distributor. `fetch` runs these scripts
beside the download, several at a time, each writing its output to `$JEVARENA_MM_MEDIA/.fetch-logs/<folder>.log`. To
run one by hand:

```bash
python scripts/fetch/<folder>.py "$JEVARENA_MM_MEDIA/<folder>"
```

Writing a script:

- Fetch only what is missing (`_common.wanted` and `_common.missing`) and write each file at the path the items
  reference (`uri` = `<folder>/<path>`) with `_common.save`, which keeps it only if its sha256 matches.
- Read archive members over HTTP ranges instead of downloading whole archives: `_common.zip_jobs` lists the members
  from the central directory and `_common.save_all` saves them several at a time (`WORKERS`), each in one streamed
  request, retrying network failures. Hugging Face serves ranges through a bridge much slower than its own client, so
  for an archive on Hugging Face use `_common.save_hf_zip_members`: it downloads the archive whole with huggingface_hub
  (into `<folder>/.download`, deleted afterwards) when the wanted members are half of it or more, and reads ranges
  otherwise. Files downloaded one by one also go through `save_all`. Pin the source revision when the source has one.
- Respect access terms: when a source needs a signed agreement, a login or a password, read it from the environment;
  when it is missing, print where to obtain it and exit with status 2. Never write credentials to disk.
- End with `_common.report`; afterwards `jevarena-mm verify --leaf <prefix>` checks every file against the items.
- Images inside a source's parquet files on Hugging Face: `_common.hf_parquet_images` with `_common.by_sha` reads the
  files in ranges and keeps every image cell whose sha256 is wanted; gated sources exit 2 with where to request access.
- Rebuilt files (decoded audio, crops) must match byte for byte: pin the decoder or encoder in the extra the script
  needs, and say in the docstring how the files are built.
