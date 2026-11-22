"""Command line: jevarena-mm {tasks,build,fetch,verify,run,score,cards,...}."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from . import dataio, env, metrics, recipes, runner
from .env import SUMMARY, load_env, media_dir
from .registry import envs_under, make_model, tasks_under


def _rel(path) -> str:
    try:
        return str(path.relative_to(env.ROOT))
    except ValueError:
        return str(path)


def cmd_tasks(args: argparse.Namespace) -> None:
    for task in tasks_under(args.leaf):
        c = task.card()
        print(f"{c['name']:<48} {c['status']:<8} v{c['version']:<3} tier {c['tier']}  {c['license']:<18} {c['source']}")


def cmd_build(args: argparse.Namespace) -> None:
    load_env()
    manifest = []
    tasks = tasks_under(args.leaf)
    sizes = {t.name: args.n for t in tasks}
    if args.per_leaf:  # split one budget per leaf evenly across all its sources, also when rebuilding one
        for t in tasks:
            names = sorted(other.name for other in tasks_under(t.leaf))
            i = names.index(t.name)
            sizes[t.name] = args.per_leaf // len(names) + (1 if i < args.per_leaf % len(names) else 0)
    failed = []
    for task in tasks:
        if task.frozen:  # exported once, not built from the source: check the committed file, or rebuild it from its recipe
            items = dataio.load_task(task)
            recipe = recipes.load(task)
            if recipe and len(items) != len(recipe):
                items, missing = recipes.rebuild(task)
                if items:
                    dataio.write_items(dataio.task_path(task.name, task.redistributable), items)
                for name, n in sorted(missing.items()):
                    print(f"{task.name}: {n} of {len(recipe)} items need {recipes.SOURCES[name].path} under "
                          f"$JEVARENA_MM_MEDIA ({recipes.SOURCES[name].terms}); run `jevarena-mm fetch --leaf {task.name}`")
                if len(items) != len(recipe):  # partly built: usable, but the committed manifest entry stays
                    print(f"{task.name}: {len(items)} of {len(recipe)} items rebuilt from the recipe")
                    continue
            if not items:
                failed.append(task.name)
                print(f"{task.name}: FROZEN task has no items at {dataio.task_path(task.name, task.redistributable)}")
                continue
        else:
            try:
                items = task.build(n=sizes[task.name], seed=args.seed)
            except Exception as error:  # e.g. a gated dataset whose terms were not accepted; the others still build
                failed.append(task.name)
                print(f"{task.name}: FAILED, skipped ({type(error).__name__}: {str(error)[:200]})")
                continue
        path = dataio.task_path(task.name, task.redistributable)
        if not task.frozen:
            dataio.write_items(path, items)
        labels: dict[str, int] = {}
        for item in items:
            for gold in item.gold.values():
                labels[gold.hard()] = labels.get(gold.hard(), 0) + 1
        excluded = getattr(task, "excluded", 0)
        kinds = sorted({k for item in items for k in item.request.modalities()})
        manifest.append({**task.card(), "items": len(items), "modalities": kinds, "excluded_over_budget": excluded,
                         "duplicates_replaced": getattr(task, "duplicates", 0), "path": _rel(path),
                         "sha256": dataio.file_sha256(path), "label_counts": labels})
        print(f"{task.name}: {len(items)} items -> {_rel(path)}" + (" (frozen, checked)" if task.frozen else "")
              + (f" ({excluded} over the token budget excluded)" if excluded else ""))
    for domain in sorted({row["name"].split("/")[0] for row in manifest}):  # one manifest per <modality>.<domain>
        rows = [row for row in manifest if row["name"].startswith(domain + "/")]
        dataio.write_manifest(dataio.domain_dir(domain), rows, known={t.name for t in tasks_under(domain)})
    if failed:
        print(f"{len(failed)} task(s) not built: {', '.join(failed)}")


def consistency_problems(leaf: str = "") -> tuple[list[str], str]:
    """(problems, summary) of the committed data and docs, without media, models or writes: every item parses and has
    its content id, every item file matches its manifest entry, statuses are valid, a released task's items changed
    since the last release tag only with a new version, and the generated docs are current."""
    from .release import STATUSES, version_problems
    from .schema import content_id

    problems: list[str] = []
    tasks = tasks_under(leaf)
    checked = local = 0
    all_rows: dict[str, dict] = {}
    for task in tasks:
        if task.status not in STATUSES:
            problems.append(f"{task.name}: status {task.status!r} is not one of {', '.join(STATUSES)}")
    for domain in sorted({t.name.split("/")[0] for t in tasks}):
        rows = manifest_rows(domain)
        all_rows.update(rows)
        registered = {t.name for t in tasks_under(domain)}
        problems += [f"{name}: in {domain}'s manifest but not registered" for name in sorted(set(rows) - registered)]
        for task in (t for t in tasks if t.name.split("/")[0] == domain):
            row = rows.get(task.name)
            if row is None:
                problems.append(f"{task.name}: no manifest entry (run `jevarena-mm build --leaf {task.name}`)")
                continue
            stale = sorted(k for k, v in task.card().items() if row.get(k) != v)
            if stale:
                problems.append(f"{task.name}: manifest card differs in {', '.join(stale)} (run `jevarena-mm build --leaf {task.name}`)")
            path = dataio.task_path(task.name, task.redistributable)
            recipe = recipes.load(task)
            if recipe:
                ids = [row["id"] for row in recipe]
                unknown = set().union(*(recipes.sources_of(row) for row in recipe)) - set(recipes.SOURCES)
                for message, bad in ((f"recipe has {len(recipe)} items, manifest says {row.get('items')}", len(recipe) != row.get("items")),
                                     (f"{len(ids) - len(set(ids))} duplicate recipe ids", len(ids) != len(set(ids))),
                                     (f"recipe names unknown sources {sorted(unknown)}", unknown)):
                    if bad:
                        problems.append(f"{task.name}: {message}")
            elif not task.redistributable:
                problems.append(f"{task.name}: not redistributed and no recipe at {_rel(recipes.path(task))}")
            if not path.exists():
                if task.redistributable:
                    problems.append(f"{task.name}: item file {_rel(path)} missing")
                else:
                    local += 1  # not redistributed: its recipe and manifest entry are committed
                continue
            try:
                items = dataio.read_items(path)
            except Exception as error:
                problems.append(f"{task.name}: {type(error).__name__}: {str(error)[:300]}")
                continue
            if recipe and len(items) < len(recipe):  # built before every annotation source was fetched
                problems.append(f"{task.name}: {len(items)} of {len(recipe)} items rebuilt from the recipe; run "
                                f"`jevarena-mm fetch --leaf {task.name}`, then `jevarena-mm build --leaf {task.name}`")
                checked += 1
                continue
            ids = [item.id for item in items]
            wrong = [item.id for item in items if item.id != content_id(item)]
            other = {item.task for item in items} - {task.name} or {item.leaf for item in items} - {task.leaf}
            for message, bad in ((f"{len(wrong)} item ids are not content ids (first {wrong[:1]})", wrong),
                                 (f"{len(ids) - len(set(ids))} duplicate item ids", len(ids) != len(set(ids))),
                                 (f"items of another task or leaf: {sorted(other)}", other),
                                 (f"{len(items)} items, manifest says {row.get('items')}", len(items) != row.get("items")),
                                 ("file sha256 differs from the manifest", dataio.file_sha256(path) != row.get("sha256"))):
                if bad:
                    problems.append(f"{task.name}: {message}")
            checked += 1
    tag, changed = version_problems(tasks, all_rows, leaf)
    problems += changed
    docs = env.ROOT / "docs"
    generated = [(path, cards_text(path.stem.removeprefix("tasks_").replace("_", ".", 1)),
                  f"jevarena-mm cards --leaf {path.stem.removeprefix('tasks_').replace('_', '.', 1)} --out {_rel(path)}")
                 for path in sorted(docs.glob("tasks_*.md"))]
    generated += [(path, env_cards_text(path.stem.removeprefix("envs_").replace("_", ".", 1)),
                   f"jevarena-mm env-cards --env {path.stem.removeprefix('envs_').replace('_', '.', 1)} --out {_rel(path)}")
                  for path in sorted(docs.glob("envs_*.md"))]
    if not leaf:
        for path, text, command in generated:
            if path.read_text(encoding="utf-8") != text:
                problems.append(f"{_rel(path)} is out of date (run `{command}`)")
    released = sum(t.status == "released" for t in tasks)
    summary = (f"{checked} item files checked, {local} rebuilt from recipes (not built here), {len(generated) if not leaf else 0} "
               f"generated docs compared, {released} released and {len(tasks) - released} preview tasks, "
               + (f"versions compared with {tag}" if tag else "no release tag yet"))
    return problems, summary


def cmd_check(args: argparse.Namespace) -> None:
    """Consistency of the committed data and docs (run by CI); see consistency_problems."""
    problems, summary = consistency_problems(args.leaf)
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"{summary}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    if problems:
        raise SystemExit(1)


def cmd_release(args: argparse.Namespace) -> None:
    """Check the release, then write the snapshot, commit and tag (see release.py)."""
    from . import release

    load_env()
    problems, _ = consistency_problems("")
    code = release.release(args.version, tasks_under(""), envs_under(""), problems, dry_run=args.dry_run)
    if code:
        raise SystemExit(code)


def media_roots(task) -> set[str]:
    """Media folders a task uses: its media_root plus the first path component of every uri its items (or, before they
    are built, its recipe) reference."""
    from . import recipes

    roots = {task.media_root} if task.media_root else set()
    for ref in recipes.media_of(task):
        roots.add(ref.uri.split("/", 1)[0])
        if ref.marks is not None:
            roots.add(ref.marks.uri.split("/", 1)[0])
    return roots


SCRIPT_WORKERS = 4  # fetch scripts run at once, beside the download from the Hugging Face dataset


def cmd_fetch(args: argparse.Namespace) -> None:
    """Get every media folder the selected tasks use: hosted folders from the Hugging Face dataset (jevarena_mm.hub),
    the others with scripts/fetch/<root>.py, which gets them from their owner. The scripts read other sites, so they
    run beside the download, several at a time; each writes its output to <media>/.fetch-logs/<root>.log."""
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from . import hub

    load_env()
    roots = sorted(set().union(*(media_roots(task) for task in tasks_under(args.leaf))))
    hosted = [root for root in roots if root not in hub.NOT_HOSTED_FOLDERS]
    if args.leaf in ("", hub.LATENCY):  # everything, or only the latency sets' closed-loop packets
        hosted.append(hub.LATENCY)
    scripted = [root for root in roots if root in hub.NOT_HOSTED_FOLDERS or hub.partly_hosted(root)]
    failed, lock, logs = [], threading.Lock(), media_dir() / ".fetch-logs"

    def run_script(root: str) -> None:
        script = env.ROOT / "scripts" / "fetch" / f"{root}.py"
        if not script.exists():
            with lock:
                failed.append(root)
                print(f"{root}: not hosted ({hub.NOT_HOSTED.get(root + '/*', 'in part')}); no fetch script "
                      f"({_rel(script)})", flush=True)
            return
        dest = media_dir() / root
        dest.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{root}.log"
        with open(log, "w", encoding="utf-8") as out:
            code = subprocess.run([sys.executable, str(script), str(dest)], stdout=out, stderr=subprocess.STDOUT,
                                  check=False).returncode
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        with lock:
            if not code:
                print(f"{root}: {lines[-1] if lines else 'fetched'}", flush=True)
                return
            failed.append(root)
            print(f"{root}: fetch script exited with {code} (log: {log})", flush=True)
            # where to get access, if the script said so, then how it ended
            for line in dict.fromkeys([line for line in lines if "https://" in line][:3] + lines[-5:]):
                print(f"  {line}", flush=True)

    with ThreadPoolExecutor(SCRIPT_WORKERS) as pool:
        running = [pool.submit(run_script, root) for root in scripted]
        if hosted:
            print(f"{', '.join(hosted)}: from {hub.REPO} into {media_dir()}", flush=True)
            hub.fetch(hosted, media_dir())
        for future in running:
            future.result()
    if failed:
        raise SystemExit(f"{len(failed)} media folder(s) not fetched: {', '.join(sorted(failed))}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Check that every media file the selected items reference is present and matches its sha256."""
    from . import media

    load_env()
    problems = 0
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)
        if not items:
            continue
        refs = {(ref.uri, ref.sha256): ref for item in items for _, ref in item.request.media()}
        bad: dict[str, int] = {}
        for ref in refs.values():
            reason = media.check(ref)
            if reason:
                bad[reason] = bad.get(reason, 0) + 1
                if args.show:
                    print(f"  {reason}: {ref.uri}")
        affected = sum(1 for item in items if any(media.check(ref) for _, ref in item.request.media())) if bad else 0
        problems += affected
        status = "ok" if not bad else ", ".join(f"{n} {reason}" for reason, n in sorted(bad.items())) + f"; {affected} items affected"
        print(f"{task.name}: {len(items)} items, {len(refs)} media files: {status}")
    if problems:
        raise SystemExit(f"{problems} item(s) reference missing or changed media under {media_dir()}")


def cmd_run(args: argparse.Namespace) -> None:
    load_env()  # local adapters read their weight and engine paths from .env
    model = make_model(args.model)
    index, count = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    if not 0 <= index < count:
        raise SystemExit("--shard is I/N with 0 <= I < N")
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)[: args.limit or None]
        if not items:
            print(f"{task.name}: not built, skipping")
            continue
        items = [item for n, item in enumerate(sorted(items, key=lambda i: i.id)) if n % count == index]
        calls = runner.run_task(model, task.name, items, k=args.k, workers=args.workers,
                                retry_rejected=args.retry_rejected, shard=(index, count))
        print(f"{task.name}: {calls} new calls")


def cmd_reparse(args: argparse.Namespace) -> None:
    model = make_model(args.model)
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)[: args.limit or None]
        if items:
            print(f"{task.name}: {runner.reparse_task(model, task.name, items)} replies re-read successfully")


def cmd_score(args: argparse.Namespace) -> None:
    table = []
    for spec in args.models:
        model = make_model(spec)
        for task in tasks_under(args.leaf):
            items = dataio.load_task(task)[: args.limit or None]
            keep = {i.id for i in items}
            rows = [r for r in runner.latest_rows(model.slug, task.name) if r["item_id"] in keep]
            ids = {r["item_id"] for r in rows}
            result = metrics.task_metrics(task, [i for i in items if i.id in ids], rows)
            if result.get("n_questions") or result.get("unsupported_items"):
                table.append({"model": model.model_id, "task": task.name, "leaf": task.leaf, "status": task.status,
                              **result, "not_run": len(items) - len(ids)})  # e.g. media not fetched: not scored
    for row in table:
        if not row["n_questions"]:
            print(f"{row['model']:<28}  {row['task']:<44}  {row['status']:<8}  unsupported ({row['unsupported_items']} items)")
            continue
        cells = [f"{row['model']:<28}", f"{row['task']:<44}", f"{row['status']:<8}", f"n={row['n_questions']:<4}", f"cov={row['coverage']:.0%}"]
        cells += [f"acc={row['accuracy']:.3f}±{row['accuracy_se']:.3f}", f"maj={row['maj']:.3f}", f"brier={row['brier']:.3f}", f"ece={row['ece']:.3f}"]
        if row.get("latency_p50_ms") is not None:
            cells.append(f"p50={row['latency_p50_ms']:.0f}ms")
        if row.get("cost_per_1k_usd") is not None:
            cells.append(f"${row['cost_per_1k_usd']:.4f}/1k")
        if row.get("ndcg") is not None:
            cells.append(f"ndcg={row['ndcg']:.3f}")
        if row.get("pair_accuracy") is not None:
            cells.append(f"pair={row['pair_accuracy']:.3f}")
        cells.append(f"err={row['error_rate']:.1%}")
        if row["not_run"]:
            cells.append(f"not run={row['not_run']}")
        print("  ".join(cells))
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        path = SUMMARY / args.out
        path.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")


REDACTED = "[not shown: this source does not allow redistribution; run the loader locally to see the text]"


def cmd_examples(args: argparse.Namespace) -> None:
    """One full example item per task, for the website."""
    out = {}
    for task in tasks_under(args.leaf):
        items = dataio.load_task(task)
        if not items:
            continue
        short = [i for i in items if len(json.dumps(i.request.state, ensure_ascii=False)) <= args.max_chars]
        pool = short or items
        item = json.loads(pool[args.index % len(pool)].model_dump_json())
        if not task.redistributable:  # show the question structure, not the licensed text
            item["request"]["state"] = {k: REDACTED for k in item["request"]["state"]}
        out[task.name] = {"card": task.card(), "item": item}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {len(out)} examples to {args.out}")
    else:
        sys.stdout.write(text + "\n")


def manifest_rows(domain: str) -> dict[str, dict]:
    path = dataio.domain_dir(domain) / "manifest.json"
    return {row["name"]: row for row in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}


def cards_text(leaf: str) -> str:
    """Markdown table of task cards plus item and label counts. A task whose items are not on this machine (e.g. not
    redistributed) is described from its manifest entry, so the table is the same everywhere."""
    lines = [f"# Tasks under `{leaf or 'all'}`", "", "Generated by `jevarena-mm cards`. Status: released tasks are frozen and form the main tables; preview tasks may "
             "still change and are reported apart. Tier A: original instructions verbatim; B: adapted; C: newly written. Media: "
             "the kinds the items reference.", ""]
    lines += ["| Task | Status | Version | Source | License | Tier | Media | Items | Labels | Criteria source | Notes |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in tasks_under(leaf):
        c = task.card()
        items = dataio.load_task(task) if task.redistributable else []  # a local rebuild may be partial
        if items:
            count, labels = len(items), len({g.hard() for i in items for g in i.gold.values()})
            kinds = sorted({k for i in items for k in i.request.modalities()})
        else:
            row = manifest_rows(task.name.split("/")[0]).get(task.name, {})
            count, labels, kinds = row.get("items", 0), len(row.get("label_counts", {})), row.get("modalities", [])
        note = c["notes"] + ("" if c["redistributable"] else " Loader only: items are not redistributed.")
        source = f"[{c['source']}]({c['url']})" if c["url"] else c["source"]
        cells = [f"`{c['name']}`", c["status"], str(c["version"]), source, c["license"], c["tier"], " + ".join(kinds) or "text", str(count),
                 str(labels), c["criteria_source"], note.strip()]
        lines.append("| " + " | ".join(x.replace("|", "/") for x in cells) + " |")
    return "\n".join(lines) + "\n"


def cmd_cards(args: argparse.Namespace) -> None:
    """Markdown table of task cards plus item and label counts, for docs/."""
    text = cards_text(args.leaf)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


def cmd_compare(args: argparse.Namespace) -> None:
    """Per-leaf pivot of several models on the same items: released tasks in the main table (a leaf is the mean over
    its released tasks, the overall score the simple mean over leaves, median for latency), preview tasks apart."""
    from .release import leaderboard

    models = [make_model(spec) for spec in args.models]
    names = [m.model_id for m in models]
    board = leaderboard(models, tasks_under(args.leaf), args.limit, args.k or None)
    per_leaf, summary, preview = board["per_leaf"], board["overall"], board["preview"]
    print(f"{'leaf (released tasks)':<36}" + "".join(f"{n.split('/')[-1][:18]:>20}" for n in names))
    for leaf in sorted(per_leaf):
        cells = [f"{per_leaf[leaf][n]['accuracy']:.3f}" if "accuracy" in per_leaf[leaf].get(n, {})
                 else "unsup" if n in per_leaf[leaf] else "-" for n in names]
        print(f"{leaf.split('/', 1)[-1]:<36}" + "".join(f"{c:>20}" for c in cells))
    print()
    for n, s in summary.items():
        extra = "".join(f" {k}={s[k]:.3f}" for k in ("ndcg", "pair_accuracy") if k in s)
        print(f"{n:<32} acc={s['accuracy']:.3f} maj={s.get('maj', float('nan')):.3f} brier={s['brier']:.3f} ece={s['ece']:.3f} "
              f"p50={s.get('latency_p50_ms', float('nan')):.0f}ms cost/1k=${s.get('cost_per_1k_usd', float('nan')):.3f}{extra}  ({s['leaves']} leaves"
              + (f", {s['unsupported_leaves']} unsupported)" if s["unsupported_leaves"] else ")"))
    if preview:
        print(f"\n{'preview task (not in the overall)':<52}" + "".join(f"{n.split('/')[-1][:18]:>20}" for n in names))
        for task in sorted(preview):
            cells = [f"{preview[task][n]['accuracy']:.3f}" if "accuracy" in preview[task].get(n, {})
                     else "unsup" if n in preview[task] else "-" for n in names]
            print(f"{task.split('/', 1)[-1]:<52}" + "".join(f"{c:>20}" for c in cells))
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / args.out).write_text(json.dumps(board, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SUMMARY / args.out}")


def cmd_img_score(args: argparse.Namespace) -> None:
    """Score the image tasks with their own scoring (jevarena_mm.metrics_img): per leaf (items pooled over its tasks),
    per domain and overall, written to results/summary/img/<model>.json."""
    from . import metrics_img

    tasks = [task for task in tasks_under(args.leaf) if task.name.startswith("img.")]
    items = [metrics_img.as_item(item) for task in tasks for item in dataio.load_task(task)]
    out_dir = SUMMARY / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.models:
        model = make_model(spec)
        runs = {}
        for task in tasks:
            runs.update(metrics_img.runs_of(runner.latest_rows(model.slug, task.name)))
        out = {"model": model.model_id, **metrics_img.score_model(items, runs, args.k)}
        path = out_dir / f"{model.slug}.json"
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        m = out["overall"]
        if m[f"avg@{args.k}"] is None:
            print(f"{model.model_id}: no scored items -> {path}")
            continue
        print(f"{model.model_id}: avg@{args.k} {m[f'avg@{args.k}']:.3f}  brier {m['brier']:.3f}  ece {m['ece']:.3f}  "
              f"coverage {m['coverage']:.3f}  -> {path}")


def cmd_img_compare(args: argparse.Namespace) -> None:
    """Overall, per-domain and per-leaf tables of every model scored by img-score, written to
    results/summary/img/README.md."""
    from . import metrics_img

    out_dir = SUMMARY / "img"
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]
    rows = [r for r in rows if r["overall"].get(f"avg@{r['k']}") is not None]
    if not rows:
        raise SystemExit(f"no scored models in {out_dir}; run jevarena-mm img-score first")
    k = rows[0]["k"]
    rows.sort(key=lambda r: -(r["overall"][f"avg@{k}"] or 0))
    note = (f"Generated by `jevarena-mm img-compare` (the image tasks' scoring, jevarena_mm/metrics_img.py). avg@{k} "
            "(accuracy); coverage in brackets where a model was not run on every item; the best value of each column in "
            "bold.\n")
    lines = ["# Image results\n", note] + metrics_img.results_md(rows, k)
    lines += ["## Model specs\n"] + [f"- `{r['model']}`" for r in rows] + [""]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


def cmd_envs(args: argparse.Namespace) -> None:
    for cls in envs_under(args.env):
        print(f"{cls.name:<52} {cls.mode:<8} episodes {cls.episodes:<3} max decisions {cls.max_decisions:<5} {cls.source}")


def cmd_play(args: argparse.Namespace) -> None:
    """Play the benchmark episodes of closed-loop environments (cached per model, environment, seed, mode)."""
    from .envs import runner as env_runner

    load_env()
    model = make_model(args.model)
    index, count = (int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    if not 0 <= index < count:
        raise SystemExit("--shard is I/N with 0 <= I < N")
    fixed, kind = args.latency_ms, args.latency_gpu  # realtime: the latency charged per decision, resolved once
    for cls in envs_under(args.env)[index::count]:  # each environment writes its own episode file
        if getattr(cls, "streaming_only", False):
            cls.play(model, [], args.workers)
            continue
        mode = "realtime" if args.realtime or cls.mode == "realtime" else "paused"
        if mode == "realtime" and fixed is None and getattr(cls, "realtime_ok", True):
            try:
                fixed, kind = env_runner.realtime_latency(model, args.latency_gpu)
            except ValueError as error:
                raise SystemExit(str(error)) from None
        latency = (fixed or 0.0) if mode == "realtime" else 0.0
        seeds = list(range(args.episodes or cls.episodes))
        played = env_runner.run_env(model, cls, seeds, mode, latency, workers=args.workers, max_decisions=args.max_decisions or None,
                                    latency_gpu=kind if mode == "realtime" else None)
        print(f"{cls.name}: {played} episodes played ({mode}"
              + (f", {latency:.0f} ms per decision" + (f" measured on {kind}" if kind else "") if mode == "realtime" else "") + ")")
    from .envs import upstream

    upstream.close_servers()  # then drop the model, so that its engine shuts down now and the exit does not wait on it
    del model
    gc.collect()


def cmd_serve(args: argparse.Namespace) -> None:
    """Serve one model to the closed-loop bridges (jevarena_mm/bridges/) until interrupted."""
    from .serve import serve_forever

    load_env()
    serve_forever(make_model(args.model), args.port, Path(args.log) if args.log else None)


def cmd_env_score(args: argparse.Namespace) -> None:
    from .envs import runner as env_runner

    table = []
    for spec in args.models:
        model = make_model(spec)
        for cls in envs_under(args.env):
            seeds = set(range(args.episodes or cls.episodes))
            for mode in ("paused", "realtime"):
                cap = args.max_decisions or cls.max_decisions  # smoke runs with a lower cap are scored separately
                latency, kind = None, None
                if mode == "realtime":  # the model's measured latency unless another one is asked for
                    try:
                        latency, kind = ((args.latency_ms, args.latency_gpu) if args.latency_ms is not None
                                         else env_runner.realtime_latency(model, args.latency_gpu))
                    except ValueError:  # no measurement and no static runs to fix a latency from: any realtime episodes
                        latency = None
                rows = [r for r in env_runner.latest_episodes(model.slug, cls.name, mode, cap, latency, cls.version, kind) if r["seed"] in seeds]
                if rows:
                    table.append({"model": model.model_id, "env": cls.name, "leaf": cls.leaf, "mode": mode, "score_name": cls.score_name,
                                  **({"latency_ms": latency, "latency_gpu": kind} if mode == "realtime" else {}), **env_runner.summarize(rows)})
    for row in table:
        if not row["episodes"]:
            print(f"{row['model']:<28} {row['env']:<52} {row['mode']:<8} no completed episodes (crashed {row.get('crashed', 0)}, unsupported {row.get('unsupported', 0)})")
            continue
        success = f"success={row['success_rate']:.2f}" if row.get("success_rate") is not None else "success=-"
        score = f"{row['score_name']}={row['score_mean']:.2f}" + (f"±{row['score_se']:.2f}" if row.get("score_se") else "")
        extra = f"p50={row['latency_p50_ms']:.0f}ms" if row.get("latency_p50_ms") else ""
        print(f"{row['model']:<28} {row['env']:<52} {row['mode']:<8} n={row['episodes']:<3} {success}  {score}  decisions={row['decisions_mean']:.0f}  failed={row['failed_decision_rate'] or 0:.1%}  {extra}")
    if args.out:
        SUMMARY.mkdir(parents=True, exist_ok=True)
        (SUMMARY / args.out).write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {SUMMARY / args.out}")


def cmd_env_examples(args: argparse.Namespace) -> None:
    """The first decision request of seed 0 for each environment, in the same shape as task examples."""
    out = {}
    classes = envs_under(args.env)
    if args.per_leaf:  # a spread of at most N environments per leaf, e.g. for the site
        by_leaf: dict[str, list] = {}
        for cls in classes:
            by_leaf.setdefault(cls.leaf, []).append(cls)
        classes = [group[round(i * (len(group) - 1) / max(1, args.per_leaf - 1))] for group in by_leaf.values() for i in range(min(args.per_leaf, len(group)))]
        classes = list(dict.fromkeys(classes))
    for cls in classes:
        env = cls()
        try:
            env.reset(0)
            request = env.observe()
        except Exception as error:  # e.g. an environment whose simulator server is not running here
            print(f"skipped {cls.name}: {type(error).__name__}: {str(error)[:200]}", file=sys.stderr)
            continue
        finally:
            env.close()
        meta = {"source": cls.source, "license": cls.license, "tier": cls.tier, "closed_loop": True}
        out[cls.name] = {"card": env.card(), "item": {"id": cls.name, "leaf": cls.leaf, "task": cls.name, "request": json.loads(request.model_dump_json()), "gold": {}, "meta": meta}}
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {len(out)} environment examples to {args.out}")
    else:
        sys.stdout.write(text + "\n")


def env_cards_text(prefix: str) -> str:
    """Markdown table of the closed-loop environments."""
    lines = [f"# Closed-loop environments under `{prefix or 'all'}`", "",
             "Generated by `jevarena-mm env-cards`. Mode: paused (the world waits) or realtime (it keeps moving while the model decides). "
             "Scores are raw; success is shown where the task defines it.", "",
             "| Environment | Source | License | Tier | Mode | Episodes | Max decisions | Score | Notes |", "|---|---|---|---|---|---|---|---|---|"]
    for cls in envs_under(prefix):
        mode = cls.mode + (f" ({cls.control_dt:.3g} s/step)" if cls.mode == "realtime" and cls.control_dt else "")
        notes = cls.notes + "".join(f" **{label}:** {reason}" for label, reason in
                                    (("Unsupported", getattr(cls, "unsupported", None)), ("Blocked upstream", getattr(cls, "blocked", None))) if reason)
        cells = [f"`{cls.name}`", f"[{cls.source}]({cls.url})" if cls.url else cls.source, cls.license, cls.tier, mode, str(cls.episodes), str(cls.max_decisions), cls.score_name, notes]
        lines.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ") for x in cells) + " |")
    return "\n".join(lines) + "\n"


def cmd_env_cards(args: argparse.Namespace) -> None:
    """Markdown table of the closed-loop environments, for docs/."""
    text = env_cards_text(args.env)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run every model of a plan on its tasks and environments; cached, so it resumes where it stopped."""
    from . import sweep

    load_env()
    plan = sweep.load_plan(args.plan)
    if args.models:
        plan.models = [m for m in plan.models if m.spec in args.models]
    if args.envs is not None:
        plan.envs = args.envs
    for m in plan.models:
        m.static = m.static and args.only != "envs"
        m.envs = m.envs and args.only != "static"
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit(f"--shard {args.shard}: need 0 <= index < count")
        plan.shard = (index, count)
    sweep.run(plan, parallel_models=args.parallel_models)
    sweep.print_status(table := sweep.status(plan))
    if plan.shard == (0, 1):  # shards would overwrite each other's file; run sweep-status once they are done
        print(f"wrote {sweep.write_status(plan, table)}")


def cmd_sweep_submit(args: argparse.Namespace) -> None:
    """One Slurm job array per model of a plan (scripts/slurm/run_shards.slurm); prints the commands unless --go."""
    from . import slurm, sweep

    plan = sweep.load_plan(args.plan)
    if args.models:
        plan.models = [m for m in plan.models if m.spec in args.models]
    slurm.submit(plan, go=args.go, partition=args.partition, account=args.account, site_env=args.site_env,
                 log_dir=args.log_dir, only=args.only, extra=args.sbatch_args)


def cmd_results_page(args: argparse.Namespace) -> None:
    """Merge sweep reports into the results page (results/README.md, closed_loop.md, summary/results.json) and the
    README's leaderboard."""
    from . import results_page

    intro = Path(args.intro).read_text(encoding="utf-8") if args.intro else ""
    written = results_page.write([Path(p) for p in args.reports], Path(args.models), Path(args.out),
                                 None if args.no_readme else env.ROOT / "README.md", intro,
                                 Path(args.latency) if args.latency else None)
    print("wrote " + ", ".join(str(p) for p in written))


def cmd_sweep_status(args: argparse.Namespace) -> None:
    from . import sweep

    plan = sweep.load_plan(args.plan)
    sweep.print_status(table := sweep.status(plan))
    print(f"wrote {sweep.write_status(plan, table)}")


def cmd_sweep_report(args: argparse.Namespace) -> None:
    from . import sweep

    plan = sweep.load_plan(args.plan)
    rep = sweep.report(plan)
    sys.stdout.write(sweep.report_markdown(rep))
    print("wrote " + ", ".join(map(str, sweep.write_report(plan, rep))))


def cmd_latency_set(args: argparse.Namespace) -> None:
    """Draw a latency set (docs/latency.md): items per leaf, warm-up items, and closed-loop decision packets copied
    from recorded episodes to the media root."""
    from . import latency

    load_env()
    out = Path(args.out) if args.out else latency.set_path(args.version)
    if args.loop_only:  # keep the items of an existing set, (re)draw its packets
        latency_set = json.loads(out.read_text(encoding="utf-8"))
    else:
        tasks = [t for prefix in (args.leaf or ["vid", "lvid"]) for t in tasks_under(prefix)]
        latency_set = latency.build_set(tasks, per_leaf=args.per_leaf, av_per_leaf=args.av_per_leaf, warmup=args.warmup,
                                        seed=args.seed, version=args.version, statuses=tuple(args.status or ["released"]))
    if args.loop_attempts:
        quota = {k: int(v) for k, v in (part.split("=") for part in args.loop.split(","))} if args.loop else None
        folder = media_dir() / "latency" / latency_set["version"] / "loop"
        latency_set["loop"] = latency.sample_loop(Path(args.loop_attempts), folder, quota, args.seed, args.loop_model or None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(latency_set, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for entry in latency_set["items"] + latency_set["loop"]:
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
    print(f"wrote {_rel(out)}: {counts}, {len(latency_set['warmup'])} warm-up items, digest {latency.set_digest(latency_set)}")


def cmd_latency_stage(args: argparse.Namespace) -> None:
    """Copy the media of a latency set to the node's own disk; the job then points JEVARENA_MM_MEDIA there."""
    from . import latency

    load_env()
    latency_set = json.loads(Path(args.set).read_text(encoding="utf-8"))
    if args.kinds:  # the warm-up items always (latency answers them first), the others of these kinds
        kinds = set(args.kinds.split(","))
        latency_set = {**latency_set, **{part: [e for e in latency_set[part] if e["kind"] in kinds] for part in ("items", "loop")}}
    copied, size = latency.stage(latency_set, Path(args.to))
    print(f"copied {copied} files ({size / 2**30:.2f} GiB) to {args.to}")


def cmd_latency(args: argparse.Namespace) -> None:
    """Time one model's decisions on a latency set at batch size 1 (docs/latency.md)."""
    import os

    from . import latency

    load_env()
    latency_set = json.loads(Path(args.set).read_text(encoding="utf-8"))
    spec = os.path.expandvars(args.model)
    model = make_model(spec)
    gpus = args.gpus or len(latency.visible_gpus()) or 1
    gpu = args.gpu_label or latency.gpu_label()  # the kind of GPU, in the file names: h100, a100, ...
    kinds = set(args.kinds.split(",")) if args.kinds else None
    print(f"{model.model_id}: set {latency_set['version']} ({latency.set_digest(latency_set)}), {gpus} {gpu} GPU(s)"
          + (f", fit test on the {args.fit} longest requests per kind" if args.fit else ""), flush=True)
    try:
        rows, facts = latency.measure(model, latency_set, kinds, warmup=not args.no_warmup, repeat=args.repeat,
                                      fit=args.fit, seed=args.seed)
    finally:
        getattr(model, "close", lambda: None)()
    path = latency.rows_path(latency_set["version"], model.slug, fit=bool(args.fit), gpu=gpu)
    part = "." + "+".join(sorted(kinds)) if kinds else ""  # a run of some input kinds only is kept apart: never the measurement
    path = path.with_name(path.name.removesuffix(".jsonl") + part + ".jsonl") if part else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if args.fit:
        report = latency.fit_report(model, rows, facts)
        path.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"fit test: {report['ok']}/{report['requests']} answered, peak {report['peak_memory_mib']} MiB; "
              f"{len(report['failed_as_in_accuracy_run'])} failed as in the accuracy run; new failures: {report['new_failures'] or 'none'}")
        if not report["fits"]:
            raise SystemExit(3)
        return
    summary = latency.summarize(model, spec, latency_set, rows, facts, gpus, args.seed)
    out = Path(args.out) if args.out else latency.summary_path(model.slug, gpu)
    out = out.with_name(out.stem + part + out.suffix) if part and not args.out else out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for kind, s in summary["per_kind"].items():
        print(f"  {kind:6} n={s.get('n', 0):3} median {s.get('median_ms', '-')} ms (95% {s.get('median_ci95_ms', '-')}), "
              f"p90 {s.get('p90_ms', '-')} ms, {s.get('gpu_s_per_1k', '-')} GPU-s per 1,000; "
              f"{s['failed']} failed, {s['unsupported']} unsupported")
    print(f"  against the accuracy run: {summary['consistency']}")
    print(f"wrote {_rel(out)} and {_rel(path)}")


def cmd_latency_runs(args: argparse.Namespace) -> None:
    """Copy the accuracy rows a latency set compares with, for a plan's models, to a runs folder of their own."""
    import os
    import tomllib

    from . import latency

    load_env()
    latency_set = json.loads(Path(args.set).read_text(encoding="utf-8"))
    specs = args.models or [m["spec"] for m in tomllib.loads(Path(args.plan).read_text(encoding="utf-8"))["models"]]
    slugs = [make_model(os.path.expandvars(spec)).slug for spec in specs]
    for slug, n in latency.export_runs(latency_set, slugs, Path(args.out)).items():
        print(f"{slug}: {n} rows" + ("" if n else " (no cached accuracy run here)"))


def cmd_latency_plan(args: argparse.Namespace) -> None:
    """The sbatch command of each model of a latency plan (scripts/slurm/latency.slurm); submits them with --go."""
    import os
    import re
    import shlex
    import subprocess
    import tomllib

    plan = tomllib.loads(Path(args.plan).read_text(encoding="utf-8"))
    for m in plan["models"]:
        if args.models and m["spec"] not in args.models:
            continue
        kinds = args.kinds or plan.get("kinds", "")
        variables = {"JV_SPEC": m["spec"], "JV_GPUS": str(m.get("gpus", 1)), "JV_SET": args.set or plan.get("set", ""),
                     **({"JV_KINDS": kinds} if kinds else {}), **({"JV_ENV": m["env"]} if m.get("env") else {}),
                     **({"JV_GPU_LABEL": plan["gpu_label"]} if plan.get("gpu_label") else {}),
                     **({"JV_SITE_ENV": str(Path(args.site_env).resolve())} if args.site_env else {})}
        name = "jvlat-" + re.sub(r"[^A-Za-z0-9._-]+", "-", m["spec"])[:80]
        command = ["sbatch", "-p", args.partition, *(["-A", args.account] if args.account else []),
                   "-t", m.get("time", plan.get("time", "04:00:00")), "-J", name, *args.sbatch_args,
                   str(env.ROOT / "scripts" / "slurm" / "latency.slurm")]
        print(" ".join(f"{k}={shlex.quote(v)}" for k, v in variables.items()) + " " + shlex.join(command))
        if args.go:
            subprocess.run(command, check=True, env={**os.environ, **variables})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jevarena-mm")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tasks", help="list registered tasks")
    p.add_argument("--leaf", default="")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("build", help="build items from source datasets")
    p.add_argument("--leaf", default="")
    p.add_argument("--n", type=int, default=500, help="items per source dataset")
    p.add_argument("--per-leaf", type=int, default=0, help="items per leaf, split evenly across its sources (overrides --n)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("fetch", help="download the media of the selected tasks into $JEVARENA_MM_MEDIA")
    p.add_argument("--leaf", default="", help="task prefix; latency: only the latency sets' closed-loop packets")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("release", help="check the release, write the snapshot, commit and tag")
    p.add_argument("version", help="vMAJOR.MINOR.PATCH")
    p.add_argument("--dry-run", action="store_true", help="only check; write nothing")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("check", help="check committed items, manifests and generated docs (no media, no writes; run by CI)")
    p.add_argument("--leaf", default="", help="only these tasks (the docs are compared only for a full check)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("verify", help="check that every referenced media file is present and matches its sha256")
    p.add_argument("--leaf", default="")
    p.add_argument("--show", action="store_true", help="list each missing or changed file")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("run", help="query a model on built items")
    p.add_argument("--model", required=True, help="e.g. random, onejev@fps=1")
    p.add_argument("--leaf", default="")
    p.add_argument("--k", type=int, default=1, help="samples per item")
    p.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", default="", help="I/N: only the I-th of N disjoint parts of each task, written to its own file (parallel processes or nodes)")
    p.add_argument("--retry-rejected", action="store_true",
                   help="call again items whose cached row is a rejection (e.g. after an adapter change makes them fit)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("reparse", help="re-read stored replies of failed calls with the current parser (no new calls)")
    p.add_argument("--model", required=True)
    p.add_argument("--leaf", default="")
    p.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    p.set_defaults(func=cmd_reparse)

    p = sub.add_parser("envs", help="list closed-loop environments")
    p.add_argument("--env", default="")
    p.set_defaults(func=cmd_envs)

    p = sub.add_parser("play", help="play closed-loop episodes with a model")
    p.add_argument("--model", required=True)
    p.add_argument("--env", default="")
    p.add_argument("--episodes", type=int, default=0, help="seeds 0..N-1 (0 = the environment's default)")
    p.add_argument("--max-decisions", type=int, default=0, help="cap decisions per episode, e.g. for smoke tests (0 = default)")
    p.add_argument("--realtime", action="store_true", help="run paused environments in realtime mode too")
    p.add_argument("--latency-ms", type=float, default=None, help="fixed latency per decision in realtime mode (default: the model's measured one)")
    p.add_argument("--latency-gpu", default=None, help="realtime: the latency measured on this kind of GPU (h100, a100; docs/latency.md); "
                                                       "default: the H100 one, else the fixed table")
    p.add_argument("--workers", type=int, default=4, help="episodes played in parallel")
    p.add_argument("--shard", default="", help="I/N: play every N-th environment from the I-th (parallel workers)")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("serve", help="serve a model to the closed-loop suite bridges (GameWorld, BALROG, VS-Bench)")
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--log", default="", help="append one JSON line per decision to this file")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("env-score", help="summarize cached episodes")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--env", default="")
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--max-decisions", type=int, default=0)
    p.add_argument("--latency-ms", type=float, default=None, help="realtime episodes played at this latency (default: the model's measured one)")
    p.add_argument("--latency-gpu", default=None, help="realtime episodes at the latency measured on this kind of GPU (h100, a100)")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_env_score)

    p = sub.add_parser("env-examples", help="export each environment's first decision request")
    p.add_argument("--env", default="")
    p.add_argument("--per-leaf", type=int, default=0, help="at most N environments per leaf, spread over the list (0 = all)")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_env_examples)

    p = sub.add_parser("env-cards", help="write a markdown table of the closed-loop environments")
    p.add_argument("--env", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_env_cards)

    p = sub.add_parser("score", help="score cached runs")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--leaf", default="")
    p.add_argument("--limit", type=int, default=0, help="score only the first N items per task (0 = all)")
    p.add_argument("--out", default="", help="file name under results/summary/")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("examples", help="export one example per task")
    p.add_argument("--leaf", default="")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=1200, help="prefer items whose input is at most this long")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_examples)

    p = sub.add_parser("cards", help="write a markdown table of task cards")
    p.add_argument("--leaf", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_cards)

    p = sub.add_parser("compare", help="per-leaf comparison of several models on the same items")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--leaf", default="")
    p.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    p.add_argument("--k", type=int, default=0, help="use only the first k samples (0 = all)")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("img-score", help="score image runs with the image tasks' scoring (per leaf, domain and overall)")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--leaf", default="img")
    p.add_argument("-k", type=int, default=1, help="samples per item (avg@k, maj@k)")
    p.set_defaults(func=cmd_img_score)
    p = sub.add_parser("img-compare", help="tables of every model scored by img-score (results/summary/img/README.md)")
    p.set_defaults(func=cmd_img_compare)

    p = sub.add_parser("sweep", help="run all models of a plan on all its tasks and environments")
    p.add_argument("plan", help="TOML plan, e.g. sweeps/full.toml")
    p.add_argument("--models", nargs="*", default=[], help="only these model specs from the plan")
    p.add_argument("--parallel-models", type=int, default=0, help="models run at once (0 = all)")
    p.add_argument("--shard", default="", help="I/N: this worker takes every N-th task and environment from the I-th (0-based)")
    p.add_argument("--only", choices=["static", "envs"], help="only the static tasks or only the environments")
    p.add_argument("--envs", nargs="*", help="environment prefixes instead of the plan's (e.g. game/ for the BALROG image)")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("sweep-submit", help="submit a plan to Slurm, one job array per model (prints the commands unless --go)")
    p.add_argument("plan")
    p.add_argument("--partition", "-p", required=True)
    p.add_argument("--account", "-A", default="")
    p.add_argument("--site-env", default="", help="file each job sources first (scripts/slurm/site.env.example)")
    p.add_argument("--log-dir", default="", help="job and shard logs (default: results/slurm/)")
    p.add_argument("--models", nargs="*", default=[], help="only these model specs from the plan")
    p.add_argument("--only", choices=["static", "envs"], help="only the static tasks or only the environments")
    p.add_argument("--sbatch-args", nargs="*", default=[], help="more sbatch arguments, e.g. --sbatch-args=--qos=normal")
    p.add_argument("--go", action="store_true", help="submit (default: print the sbatch commands)")
    p.set_defaults(func=cmd_sweep_submit)

    p = sub.add_parser("results-page", help="merge sweep reports into results/README.md and the README leaderboard")
    p.add_argument("reports", nargs="+", help="sweep-report JSON files; a model is taken from the first that has it")
    p.add_argument("--models", default="results/models.toml", help="display names and types of the models")
    p.add_argument("--intro", default="results/intro.md", help="markdown put above the leaderboard")
    p.add_argument("--out", default="results")
    p.add_argument("--no-readme", action="store_true", help="do not update the README's results block")
    p.add_argument("--latency", default="results/summary/latency/h100",
                   help="the latency measurements to show, one kind of GPU (docs/latency.md); empty: none")
    p.set_defaults(func=cmd_results_page)

    p = sub.add_parser("sweep-status", help="coverage of a plan per model and domain (no calls)")
    p.add_argument("plan")
    p.set_defaults(func=cmd_sweep_status)

    p = sub.add_parser("sweep-report", help="result tables of a plan (markdown + JSON in results/summary/, no calls)")
    p.add_argument("plan")
    p.set_defaults(func=cmd_sweep_report)

    p = sub.add_parser("latency-set", help="draw the requests that every model is timed on (data/latency/<version>.json)")
    p.add_argument("--leaf", action="append", default=None, help="task prefixes (default: vid and lvid)")
    p.add_argument("--status", action="append", default=None, help="task statuses to draw from (default: released)")
    p.add_argument("--per-leaf", type=int, default=10)
    p.add_argument("--av-per-leaf", type=int, default=20, help="items per leaf whose tasks need audio")
    p.add_argument("--warmup", type=int, default=10, help="warm-up items without audio (and 2 with audio)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--version", default="v1")
    p.add_argument("--out", default="", help="default: data/latency/<version>.json")
    p.add_argument("--loop-attempts", default="", help="recorded episode attempts to copy decision packets from "
                                                         "($JEVARENA_MM_ATTEMPTS or results/episodes)")
    p.add_argument("--loop", default="", help="packets per suite, e.g. gameworld=50,balrog_pixels=30,vsbench=20")
    p.add_argument("--loop-model", default="", help="only packets of this model's episodes (its slug)")
    p.add_argument("--loop-only", action="store_true", help="keep the items of the existing set; draw its packets")
    p.set_defaults(func=cmd_latency_set)

    p = sub.add_parser("latency-stage", help="copy a latency set's media to a local folder (the node's own disk)")
    p.add_argument("--set", default="data/latency/v1.json")
    p.add_argument("--to", required=True)
    p.add_argument("--kinds", default="", help="only the media of these input kinds (and of the warm-up), e.g. loop")
    p.set_defaults(func=cmd_latency_stage)

    p = sub.add_parser("latency", help="time a model's decisions on a latency set at batch size 1")
    p.add_argument("--model", required=True, help="model spec; $VARIABLES are expanded")
    p.add_argument("--set", default="data/latency/v1.json")
    p.add_argument("--gpus", type=int, default=0, help="GPUs the spec uses, for GPU-seconds (default: the visible ones)")
    p.add_argument("--kinds", default="", help="only these input kinds, e.g. short,long: a check, written apart (<slug>.<kinds>.json), "
                                                "never the measurement that realtime charges")
    p.add_argument("--repeat", type=int, default=30, help="answered requests timed a second time (noise)")
    p.add_argument("--fit", type=int, default=0, help="fit test: only the N longest requests per kind, once")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-label", default="", help="the kind of GPU in the file names (default: from the visible GPUs, e.g. h100, a100)")
    p.add_argument("--out", default="", help="default: results/summary/latency/<gpu>/<slug>.json")
    p.set_defaults(func=cmd_latency)

    p = sub.add_parser("latency-runs", help="copy the accuracy rows a latency set compares with to a runs folder")
    p.add_argument("--set", default="data/latency/v1.json")
    p.add_argument("--plan", default="sweeps/latency_h100.toml", help="its models (or --models)")
    p.add_argument("--models", nargs="*", default=[], help="model specs instead of the plan's")
    p.add_argument("--out", required=True, help="a folder to use as JEVARENA_MM_RUNS where the caches are not")
    p.set_defaults(func=cmd_latency_runs)

    p = sub.add_parser("latency-plan", help="sbatch commands of a latency plan, one node per model (submits with --go)")
    p.add_argument("plan", help="e.g. sweeps/latency_h100.toml")
    p.add_argument("--partition", "-p", required=True)
    p.add_argument("--account", "-A", default="")
    p.add_argument("--site-env", default="", help="file each job sources first (scripts/slurm/site.env.example)")
    p.add_argument("--set", default="", help="default: the plan's set")
    p.add_argument("--kinds", default="", help="only these input kinds, e.g. loop (default: the plan's kinds, else all)")
    p.add_argument("--models", nargs="*", default=[], help="only these model specs from the plan")
    p.add_argument("--sbatch-args", nargs="*", default=[])
    p.add_argument("--go", action="store_true", help="submit (default: print the commands)")
    p.set_defaults(func=cmd_latency_plan)

    from .streaming.cli import add_commands

    add_commands(sub)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
