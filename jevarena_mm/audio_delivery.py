"""Export the screened audio delivery as MM recipes, or attach it locally without copying media.

Maintainers: python -m jevarena_mm.audio_delivery export /path/to/delivery
Consumers:   python -m jevarena_mm.audio_delivery attach /path/to/delivery
No downloads, model calls, resampling, or access to conditional/ records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import dataio, env
from .schema import DecisionRequest, Item, MediaRef
from .tasks.aud.prepared import DIRECTORY, catalog, task_class


def read_index(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate id in {path.name}: {row['id']}")
            rows[row["id"]] = row
    return rows


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert(spec: dict, row: dict, target: dict, provenance: dict, group: dict) -> Item:
    """Keep questions/gold intact; map media, regenerate the content id, retain lineage outside request."""
    ident = row["id"]
    if any(value["id"] != ident for value in (target, provenance, group)):
        raise ValueError("input/target/provenance/group ids do not match")
    if provenance.get("delivery_track") != "default" or provenance.get("restrictions"):
        raise ValueError("conditional or restricted record in default delivery")
    if not group.get("connected_group"):
        raise ValueError("missing connected source group")
    if set(row["request"]["state"]) != {"audio"}:
        raise ValueError("expected one prepared audio reference")
    audio = row["request"]["state"]["audio"]
    relative = Path(audio["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("media",):
        raise ValueError("delivery audio must be under media/")
    if group.get("exact_audio_sha256") != audio["sha256"]:
        raise ValueError("split group and audio hashes differ")
    ref = MediaRef(kind="audio", uri="audio_delivery/" + relative.as_posix(), sha256=audio["sha256"],
                   info={k: audio[k] for k in ("sample_rate", "channels", "duration_s")})
    return Item(task=spec["name"], leaf=spec["leaf"],
                request=DecisionRequest(state={"audio": ref}, questions=row["request"]["questions"]),
                gold=target["gold"], meta={
                    "source": spec["source"], "license": spec["license"], "tier": "C",
                    "factory_id": ident, "source_task": spec["source_task"], "audio_task_id": spec["task_id"],
                    "group": group["connected_group"], "source_group": group["source_group"],
                    "source_split": provenance.get("original_split", group.get("original_split")),
                    "audio_diagnostic_only": spec["diagnostic_only"],
                    "audio_provenance": provenance,
                    **({"pair": provenance["pair"]} if provenance.get("pair") else {}),
                })


def export_delivery(root: Path, include_diagnostics: bool = False) -> dict:
    """Deterministic export. A task the catalog marks redistributable gets its items committed under data/; every
    other task gets a recipe and cards metadata only, its licensed annotations remaining in root."""
    root = root.resolve()
    if not (root / "manifest.json").is_file():
        raise ValueError("expected a prepared delivery with manifest.json")
    groups = read_index(root / "split_groups.jsonl")
    specs = catalog()
    tables, sources, counts, manifest = {}, {}, {}, []
    staged, stale = [], []
    for spec in specs:
        if spec["diagnostic_only"] and not include_diagnostics:
            continue
        task_id = spec["task_id"]
        if task_id not in tables:
            tables[task_id] = [read_index(root / "default" / task_id / f"{kind}.jsonl")
                               for kind in ("inputs", "targets", "provenance")]
        inputs, targets, provenance = tables[task_id]
        if inputs.keys() != targets.keys() or inputs.keys() != provenance.keys():
            raise ValueError(f"{task_id}: input/target/provenance ids do not match")
        prefixes = tuple(s["source_task"].replace("/", ".") + ":" for s in specs if s["task_id"] == task_id)
        if any(not ident.startswith(prefixes) for ident in inputs):
            raise ValueError(f"{task_id}: input from an unmapped source; update the catalog before export")
        prefix = spec["source_task"].replace("/", ".") + ":"
        items, recipes = [], []
        for ident in sorted(inputs):
            if not ident.startswith(prefix):
                continue
            item = convert(spec, inputs[ident], targets[ident], provenance[ident], groups[ident])
            items.append(item)
            template = item.model_dump(mode="json")
            template["id"] = ""  # rebuilt ids must be computed from the filled request and gold
            source = "audio_" + task_id.lower()
            template["request"]["questions"] = f"⟦{source}_inputs:questions:{ident}⟧"
            template["gold"] = f"⟦{source}_targets:gold:{ident}⟧"
            template["meta"]["audio_provenance"] = f"⟦{source}_provenance:record:{ident}⟧"
            recipes.append({"id": item.id, "item": template})
        if not items:
            continue
        if len({i.id for i in items}) != len(items):
            raise ValueError(f"{spec['name']}: duplicate content ids")
        if spec["diagnostic_only"]:
            pairs = {}
            for item in items:
                pairs.setdefault(item.meta["pair"], []).append(item.gold["injection"].hard())
            if any(sorted(labels) != ["false", "true"] for labels in pairs.values()):
                raise ValueError("incomplete diagnostic clean/attack pair")
        task = task_class(spec)()
        local = dataio.task_path(task.name, task.redistributable)
        payload = "".join(item.model_dump_json() + "\n" for item in items)
        recipe = dataio.task_path(task.name).with_suffix(".recipe.jsonl")
        if task.redistributable:
            staged.append((local, payload))
            stale.append(recipe)
        else:
            staged.append((recipe, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recipes)))
        labels = {}
        for item in items:
            for gold in item.gold.values():
                labels[gold.hard()] = labels.get(gold.hard(), 0) + 1
        manifest.append({**task.card(), "items": len(items), "modalities": ["audio"],
                         "excluded_over_budget": 0, "duplicates_replaced": 0,
                         "path": local.relative_to(env.ROOT).as_posix(),
                         "sha256": hashlib.sha256(payload.encode()).hexdigest(), "label_counts": labels})
        counts[task.name] = len(items)
        if task.redistributable:
            continue
        for kind in ("inputs", "targets", "provenance"):
            relative = f"default/{task_id}/{kind}.jsonl"
            sources[f"audio_{task_id.lower()}_{kind}"] = {
                "path": "audio_delivery/" + relative, "sha256": digest(root / relative),
                "terms": f"Prepared {task_id} audio delivery; source-specific terms in source_notices/", "format": "jsonl",
            }
    # Validate the entire delivery before changing any export files.
    for path, payload in staged:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    for path in stale:
        path.unlink(missing_ok=True)
    for domain in sorted({s["name"].split("/")[0] for s in manifest}):
        rows = [row for row in manifest if row["name"].startswith(domain + "/")]
        dataio.write_manifest(dataio.domain_dir(domain), rows, known={r["name"] for r in rows})
    exported = {"version": 1, "delivery_sha256": digest(root / "manifest.json"),
                "split_groups_sha256": digest(root / "split_groups.jsonl"), "tasks": counts, "sources": sources}
    (DIRECTORY / "export.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return exported


def attach_delivery(root: Path) -> Path:
    """Check the pinned annotations and add one local directory link; no copying or rewriting audio."""
    root = root.resolve()
    exported = json.loads((DIRECTORY / "export.json").read_text(encoding="utf-8"))
    expected = {"manifest.json": exported["delivery_sha256"], "split_groups.jsonl": exported["split_groups_sha256"]}
    expected.update({s["path"].removeprefix("audio_delivery/"): s["sha256"] for s in exported["sources"].values()})
    for relative, sha in expected.items():
        if digest(root / relative) != sha:
            raise ValueError(f"delivery checksum mismatch: {relative}")
    destination = env.media_dir() / "audio_delivery"
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != root:
            raise ValueError(f"{destination} already points to another delivery")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(root, target_is_directory=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("export", "attach"))
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--include-diagnostics", action="store_true", help="export A20 separately as a diagnostic preview")
    args = parser.parse_args(argv)
    if args.operation == "export":
        result = export_delivery(args.delivery, args.include_diagnostics)
        print(f"Exported {sum(result['tasks'].values())} items in {len(result['tasks'])} tasks")
    else:
        if args.include_diagnostics:
            parser.error("--include-diagnostics applies to export only")
        print(f"Attached {attach_delivery(args.delivery)}; run jevarena-mm build --leaf aud, then verify --leaf aud")


if __name__ == "__main__":
    main()
