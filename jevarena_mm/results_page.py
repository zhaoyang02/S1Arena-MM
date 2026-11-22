"""The results page of a release: sweep reports (jevarena-mm sweep-report, one per team or run) merged into one
leaderboard (results/README.md), the per-environment tables (results/closed_loop.md), the merged numbers
(results/summary/results.json) and the leaderboard in the repository README.

    jevarena-mm results-page results/summary/sweep_video_baselines_report.json \\
        results/summary/sweep_video_dev_report.json results/summary/sweep_system1_fulleval_report.json \\
        results/summary/sweep_gameworld_realtime_report.json

Reports are read in the order given; a model in several reports is taken from the first one that has it. Names and
types come from results/models.toml. Scores, all in percent:
- Static: a leaf's accuracy is the mean over its released tasks, as sweep-report computes it. "Static" is the mean
  over the leaves every model of the page takes (those that need no audio), so all models are scored on the same
  leaves; Video, Long video and Benchmarks split those leaves (vid and lvid taxonomy leaves, the public benchmark
  subsets *.bench/*). "Audio-visual" is the mean over the leaves that need audio, for the models that take audio.
- Brier score (lower is better): the multi-class Brier score of the answer distributions, sum over options of
  (p - y)^2 (0-2; a question without an answer counts as a uniform distribution), per leaf as sweep-report computes it
  and averaged over the same leaves as Static. A model's Brier comes from the report that gives its other numbers; the
  leaderboard shows the column once every model run for the release has it (recorded runs may lack it).
- Closed loop, paused mode: per suite, the mean over its environments of each environment's mean episode score:
  GameWorld progress and BALROG progression (both 0-1). VS-Bench returns have a different scale per game, so each
  game's mean return is min-max normalized over the models of the page before the mean (games where every model
  scores the same are left out). Environments a model cannot play (free-text actions) or that are blocked upstream
  are left out.
- Latency (docs/latency.md): each model's batch-1 measurement on one kind of GPU (results/summary/latency/<gpu>/,
  --latency), the median per input kind; the leaderboard's column is the closed-loop decisions' median.
- GameWorld realtime: the reports' realtime episodes (sweep-report with realtime = true), at the latency each model's
  episodes were charged; per model the mean over the games it played in both modes, and the change from paused there.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from statistics import mean
from typing import Any

SUITES = {"gameworld": "GameWorld", "balrog_pixels": "BALROG", "vsbench": "VS-Bench"}
TYPES = {"system-one": "System-One", "mllm": "MLLM"}
START, END = "<!-- results:start -->", "<!-- results:end -->"


def merge(paths: list[Path]) -> dict[str, Any]:
    """One report from several: the first report that has a model gives all of that model's numbers. A model a report
    names without any result yet (a plan's model not run so far) is left out, so it does not empty the shared leaves."""
    merged: dict[str, Any] = {"models": [], "sources": {}, "overall": {}, "static": {}, "static_brier": {}, "envs": {},
                              "envs_realtime": {}, "realtime_sources": {}}
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        # realtime episodes: the first report that has any of a model's gives all of them (its static numbers may come
        # from another report, e.g. one team's realtime run of another team's models)
        realtime = report.get("envs_realtime", {})
        for model in sorted({m for cells in realtime.values() for m in cells} - set(merged["realtime_sources"])):
            merged["realtime_sources"][model] = Path(path).name
            for env, cells in realtime.items():
                if model in cells:
                    merged["envs_realtime"].setdefault(env, {})[model] = cells[model]
        for model in report["models"]:
            if model in merged["sources"] or not any(model in cells for cells in report["static"].values()):
                continue
            merged["models"].append(model)
            merged["sources"][model] = Path(path).name
            merged["overall"][model] = report["overall"].get(model, {})
            for leaf, cells in report["static"].items():
                if model in cells:
                    merged["static"].setdefault(leaf, {})[model] = cells[model]
            for leaf, cells in report.get("static_brier", {}).items():  # reports before 15 Oct 2026 have none
                if model in cells:
                    merged["static_brier"].setdefault(leaf, {})[model] = cells[model]
            for env, cells in report["envs"].items():
                if model in cells:
                    merged["envs"].setdefault(env, {})[model] = cells[model]
    return merged


def load_names(path: Path) -> dict[str, dict]:
    return {m["id"]: m for m in tomllib.loads(Path(path).read_text(encoding="utf-8"))["model"]}


def _slug(model_id: str) -> str:  # as models.base.ModelAdapter.slug names a model's files
    return model_id.replace("/", "__").replace(":", "_").replace("@", "_at_").replace(",", "_").replace("=", "-")


def load_latency(folder: Path | None) -> dict[str, dict]:
    """The latency measurements in a folder (results/summary/latency/<gpu>/), by model id: each model's whole-set summary
    (<slug>.json), not its fit test or runs of some input kinds (<slug>.fit.json, <slug>.<kinds>.json)."""
    out: dict[str, dict] = {}
    for path in sorted(Path(folder).glob("*.json")) if folder and Path(folder).is_dir() else []:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("model_id") and path.stem == _slug(summary["model_id"]):
            out[summary["model_id"]] = summary
    return out


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def _played(cell: dict | None) -> bool:
    return bool(cell) and (cell.get("episodes") or 0) > 0 and cell.get("score_mean") is not None


def scores(merged: dict[str, Any], models: list[str], latency: dict[str, dict] | None = None) -> dict[str, dict[str, Any]]:
    """Per model: the static groups, the per-suite closed-loop scores and success rates, coverage, its latency and its
    GameWorld realtime score."""
    leaves = sorted(merged["static"])
    shared = [leaf for leaf in leaves if all(m in merged["static"][leaf] for m in models)]
    audio = [leaf for leaf in leaves if leaf not in shared]
    groups = {"video": [x for x in shared if x.startswith("vid.") and ".bench/" not in x],
              "long_video": [x for x in shared if x.startswith("lvid.") and ".bench/" not in x],
              "benchmarks": [x for x in shared if ".bench/" in x]}
    suite_envs: dict[str, list[str]] = {s: [] for s in SUITES}
    for env in sorted(merged["envs"]):
        suite = env.split("/")[1] if env.count("/") >= 2 else ""
        if suite in suite_envs and any(_played(c) for c in merged["envs"][env].values()):  # no model plays it: left out
            suite_envs[suite].append(env)
    # VS-Bench: each game's mean return min-max normalized over the models that played it
    normalized: dict[str, dict[str, float]] = {}
    for env in suite_envs["vsbench"]:
        values = {m: c["score_mean"] for m, c in merged["envs"][env].items() if m in models and _played(c)}
        lo, hi = (min(values.values()), max(values.values())) if values else (0, 0)
        if hi > lo:
            normalized[env] = {m: (v - lo) / (hi - lo) for m, v in values.items()}
    out: dict[str, dict[str, Any]] = {}
    for m in models:
        static = {leaf: merged["static"][leaf][m] for leaf in leaves if m in merged["static"][leaf]}
        row: dict[str, Any] = {"static": _mean([static[x] for x in shared]),
                               **{g: _mean([static[x] for x in ls]) for g, ls in groups.items()},
                               "audio_visual": _mean([static[x] for x in audio]) if all(x in static for x in audio) else None,
                               "leaves": static}
        brier = {leaf: cells[m] for leaf, cells in merged["static_brier"].items() if m in cells}
        for key, ls in (("brier", shared), *((f"brier_{g}", x) for g, x in groups.items())):
            row[key] = _mean([brier[x] for x in ls]) if ls and all(x in brier for x in ls) else None
        row["leaves_brier"] = brier
        overall = merged["overall"].get(m, {})
        answers = overall.get("static_answers") or 0
        row["static_answered"] = (answers - (overall.get("static_unanswered") or 0), answers)
        played_envs, all_envs = 0, 0
        for suite, envs in suite_envs.items():
            cells = {e: merged["envs"][e].get(m) for e in envs}
            playable = [e for e, c in cells.items() if not (c and c.get("episodes", 0) == 0 and c.get("unsupported"))]
            done = [e for e in playable if _played(cells[e])]
            all_envs += len(playable)
            played_envs += len(done)
            if suite == "vsbench":
                row[suite] = _mean([normalized[e][m] for e in done if e in normalized and m in normalized[e]])
            else:
                row[suite] = _mean([cells[e]["score_mean"] for e in done])
            row[suite + "_success"] = _mean([cells[e].get("success_rate") for e in done])
            row[suite + "_envs"] = (len(done), len(playable))
        row["envs_played"] = (played_envs, all_envs)
        # environments other models played that this one could not take (its input limit): not counted, listed
        row["not_taken"] = [e for envs in suite_envs.values() for e in envs
                            if (c := merged["envs"][e].get(m)) and c.get("episodes", 0) == 0 and c.get("unsupported")
                            and any(_played(o) for o in merged["envs"][e].values())]
        # GameWorld realtime against paused, on the games the model played in both modes
        realtime = merged.get("envs_realtime", {})
        both = [e for e in suite_envs["gameworld"]
                if _played(merged["envs"][e].get(m)) and _played(realtime.get(e, {}).get(m))]
        rt_cells = [realtime[e][m] for e in both]
        row["gameworld_realtime"] = _mean([c["score_mean"] for c in rt_cells])
        row["gameworld_realtime_success"] = _mean([c.get("success_rate") for c in rt_cells])
        row["gameworld_paused_same"] = _mean([merged["envs"][e][m]["score_mean"] for e in both])
        row["gameworld_delta"] = (row["gameworld_realtime"] - row["gameworld_paused_same"]
                                  if row["gameworld_realtime"] is not None else None)
        row["gameworld_realtime_envs"] = len(both)
        row["realtime_latency_ms"] = rt_cells[0].get("latency_ms") if rt_cells else None
        measured = (latency or {}).get(m) or {}
        kinds = measured.get("per_kind") or {}
        row["latency_ms"] = measured.get("median_ms")  # the closed-loop decisions' median
        for kind in ("short", "long", "bench", "av"):
            row[f"latency_{kind}_ms"] = (kinds.get(kind) or {}).get("median_ms")
        row["latency_gpus"] = measured.get("gpus")
        row["latency_agreement"] = (measured.get("consistency") or {}).get("top_option_agreement")
        out[m] = row
    # static items still missing: fewer answers than the models that take the same leaves gave
    full: dict[int, int] = {}
    for m in models:
        n_leaves, answers = len(out[m]["leaves"]), out[m]["static_answered"][1]
        full[n_leaves] = max(full.get(n_leaves, 0), answers)
    for m in models:
        out[m]["static_partial"] = out[m]["static_answered"][1] < full[len(out[m]["leaves"])]
    out["_groups"] = {"shared": shared, "audio": audio, **groups, "suites": suite_envs}
    return out


def _pct(value: float | None, best: bool = False) -> str:
    if value is None:
        return "–"
    text = f"{100 * value:.1f}"
    return f"**{text}**" if best else text


def _ranked(models: list[str], rows: dict) -> list[str]:
    return sorted(models, key=lambda m: -(rows[m]["static"] or 0))


def _name(m: str, names: dict, link: bool = True) -> str:
    info = names.get(m, {})
    name = info.get("name", m.split("@")[0].split("/")[-1])
    return f"[{name}](https://huggingface.co/{info['repo']})" if link and info.get("repo") else name


def brier_complete(models: list[str], rows: dict, names: dict) -> bool:
    """Every model run for the release (recorded runs aside) has a Brier score."""
    return all(rows[m]["brier"] is not None for m in models if names.get(m, {}).get("source") != "recorded")


def _brier(value: float | None, best: bool = False) -> str:
    if value is None:
        return "–"
    return f"**{value:.3f}**" if best else f"{value:.3f}"


def _ms(value: float | None, best: bool = False) -> str:
    if value is None:
        return "–"
    return f"**{value:,.0f}**" if best else f"{value:,.0f}"


def _s(value: float | None) -> str:  # milliseconds shown in seconds
    return "–" if value is None else f"{value / 1000:.2f}"


def _delta(value: float | None) -> str:
    if value is None:
        return "–"
    points = round(100 * value, 1)
    return "0.0" if points == 0 else f"{points:+.1f}"


FORMAT = {"brier": _brier, "latency_ms": _ms}
LOOP = ("gameworld", "balrog_pixels", "vsbench", "gameworld_realtime")


def leaderboard(models: list[str], rows: dict, names: dict) -> str:
    columns = [("static", "Static"), *([("brier", "Brier ↓")] if brier_complete(models, rows, names) else []),
               ("video", "Video"), ("long_video", "Long video"), ("benchmarks", "Benchmarks"),
               ("audio_visual", "Audio-visual"), ("gameworld", "GameWorld"), ("balrog_pixels", "BALROG"),
               ("vsbench", "VS-Bench"),
               *([("gameworld_realtime", "GameWorld realtime")] if any(rows[m]["gameworld_realtime"] is not None for m in models) else []),
               *([("latency_ms", "Latency ↓")] if any(rows[m]["latency_ms"] is not None for m in models) else [])]
    pick = {k: min if k in ("brier", "latency_ms") else max for k, _ in columns}
    best = {k: pick[k]((rows[m][k] for m in models if rows[m][k] is not None), default=None) for k, _ in columns}
    lines = ["| # | Model | Type | " + " | ".join(t for _, t in columns) + " |",
             "|---:|---|---|" + "---:|" * len(columns)]
    for rank, m in enumerate(_ranked(models, rows), start=1):
        kind = TYPES.get(names.get(m, {}).get("type", ""), "")
        cells = [FORMAT.get(k, _pct)(rows[m][k], rows[m][k] is not None and rows[m][k] == best[k]) for k, _ in columns]
        played, total = rows[m]["envs_played"]
        partial = {k for k, _ in columns if k not in LOOP} if rows[m]["static_partial"] else set()  # static still running
        if played < total:  # a closed loop not yet complete: its scores are partial
            partial |= set(LOOP)
        cells = [c + " ‡" if k in partial and c != "–" else c for (k, _), c in zip(columns, cells)]
        lines.append(f"| {rank} | {_name(m, names)} | {kind} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _leaf_table(leaves: list[str], models: list[str], rows: dict, names: dict) -> str:
    head = "| Leaf | " + " | ".join(_name(m, names, link=False) for m in models) + " |"
    lines = [head, "|---|" + "---:|" * len(models)]
    for leaf in leaves:
        values = [rows[m]["leaves"].get(leaf) for m in models]
        top = max((v for v in values if v is not None), default=None)
        lines.append(f"| `{leaf}` | " + " | ".join(_pct(v, v is not None and v == top) for v in values) + " |")
    return "\n".join(lines) + "\n"


def loop_table(models: list[str], rows: dict, names: dict) -> str:
    lines = ["| Model | GameWorld progress | success | BALROG progression | success | VS-Bench normalized | Environments played |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in _ranked(models, rows):
        r = rows[m]
        played, total = r["envs_played"]
        lines.append(f"| {_name(m, names, link=False)} | {_pct(r['gameworld'])} | {_pct(r['gameworld_success'])} | "
                     f"{_pct(r['balrog_pixels'])} | {_pct(r['balrog_pixels_success'])} | {_pct(r['vsbench'])} | "
                     f"{played} / {total} |")
    return "\n".join(lines) + "\n"


def env_tables(merged: dict, models: list[str], rows: dict, names: dict) -> str:
    """Per environment: the mean episode score (success rate in brackets), one table per suite."""
    ranked = _ranked(models, rows)
    parts = ["# Closed-loop results per environment\n",
             "Mean episode score per environment, paused mode (success rate in brackets where the environment defines "
             "one): GameWorld progress and BALROG progression (0-1), VS-Bench raw return (the leaderboard normalizes it "
             "per game). `–`: not played (free-text actions, blocked upstream, over the model's input limit, or no episode recorded). Models in the "
             "order of the [leaderboard](README.md).\n"]
    for suite, title in SUITES.items():
        envs = rows["_groups"]["suites"][suite]
        parts.append(f"## {title}\n")
        parts.append("| Environment | " + " | ".join(_name(m, names, link=False) for m in ranked) + " |")
        parts.append("|---|" + "---:|" * len(ranked))
        for env in envs:
            cells = []
            for m in ranked:
                c = merged["envs"][env].get(m)
                if not _played(c):
                    cells.append("–")
                    continue
                s = c.get("success_rate")
                cells.append(f"{c['score_mean']:.2f}" + (f" ({s:.2f})" if s is not None else ""))
            parts.append(f"| `{env.split('/')[-1]}` | " + " | ".join(cells) + " |")
        parts.append("")
    parts.append(realtime_table(merged, models, rows, names))
    return "\n".join(parts) + "\n"


def page(merged: dict, rows: dict, names: dict, intro: str) -> str:
    models = merged["models"]
    g = rows["_groups"]
    system_one = [m for m in _ranked(models, rows) if names.get(m, {}).get("type") == "system-one"]
    mllm = [m for m in _ranked(models, rows) if m not in system_one]
    leaves = g["video"] + g["long_video"] + g["audio"] + g["benchmarks"]
    unanswered = [(m, rows[m]["static_answered"]) for m in _ranked(models, rows)
                  if rows[m]["static_answered"][0] < rows[m]["static_answered"][1]]
    board = leaderboard(models, rows, names)
    parts = ["# Results\n", intro.strip() + "\n", "## Leaderboard\n", board,
             notes(g, brier_complete(models, rows, names), merged.get("running", []), "‡" in board,
                   any(rows[m]["brier"] is None for m in models), "Latency ↓" in board or "GameWorld realtime" in board, page=""),
             "## Static accuracy per leaf\n",
             f"Accuracy per leaf (mean over its released tasks). {len(g['video'])} video and {len(g['long_video'])} "
             f"long-video taxonomy leaves, {len(g['audio'])} audio-visual leaves (models that take audio) and "
             f"{len(g['benchmarks'])} public benchmark subsets; the best score of each table's row in bold.\n",
             "### System-One decision models\n", _leaf_table(leaves, system_one, rows, names),
             "### Multimodal LLMs as classifiers\n", _leaf_table(leaves, mllm, rows, names),
             *calibration(models, rows, names),
             "## Closed loop\n",
             "Per suite (paused mode). Per-environment scores: [closed_loop.md](closed_loop.md).\n",
             loop_table(models, rows, names)]
    not_taken: dict[str, list[str]] = {}
    for m in _ranked(models, rows):
        for e in rows[m]["not_taken"]:
            not_taken.setdefault(e, []).append(_name(m, names, link=False))
    if not_taken:
        parts.append("Not counted for a model: environments whose decision prompt is over the model's input limit (every "
                     "call rejected; final, as rejected static items are): "
                     + "; ".join(f"`{e.split('/')[-1]}` ({', '.join(who)})" for e, who in not_taken.items()) + ".\n")
    parts += speed(models, rows, names)
    if unanswered:
        parts += ["## Items without an answer\n",
                  "Static items a model was given but did not answer: inputs over the model's context or memory "
                  "(rejected, final), or requests its server refuses (Xor: two-view videos). They count as wrong; items "
                  "needing a modality the model does not take are not given to it and do not count.\n",
                  "| Model | Answered |", "|---|---:|"]
        parts += [f"| {_name(m, names, link=False)} | {a:,} / {t:,} |" for m, (a, t) in unanswered]
        parts.append("")
    return "\n".join(parts) + "\n"


def speed(models: list[str], rows: dict, names: dict) -> list[str]:
    """Latency per input kind and GameWorld in realtime, fastest closed-loop decision first."""
    timed = [m for m in models if rows[m]["latency_ms"] is not None or rows[m]["gameworld_realtime"] is not None]
    if not timed:
        return []
    timed.sort(key=lambda m: (rows[m]["latency_ms"] is None, rows[m]["latency_ms"] or 0))
    best = min((rows[m]["latency_ms"] for m in timed if rows[m]["latency_ms"] is not None), default=None)
    lines = ["## Latency and realtime\n",
             "How long one decision takes at batch size 1 ([docs/latency.md](../docs/latency.md)): the median over the "
             "latency set (`data/latency/v1.json`) per input kind, on an H100 80 GB PCIe with the node to itself (two "
             "GPUs, tensor parallel, for the models that do not fit on one; flash attention first), with the model as "
             "in its accuracy run; *closed loop* is the median over its 100 "
             "decisions of recorded episodes (GameWorld, BALROG, VS-Bench). The same set on an H100 80 GB SXM machine: "
             "[summary/latency/h100-sxm/](summary/latency/h100-sxm/). GameWorld *realtime* plays the same games with "
             "the game running for the model's closed-loop latency (rounded to 10 ms) before each chosen action lands; "
             "*Δ* is realtime minus paused on the games the model played in both modes (percentage points).",
             "", "| # | Model | Type | Closed loop (ms) | Short video (s) | Long video (s) | Benchmarks (s) | "
                 "Audio-visual (s) | GameWorld paused | realtime | Δ |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, m in enumerate(timed, start=1):
        r = rows[m]
        kind = TYPES.get(names.get(m, {}).get("type", ""), "")
        lines.append(f"| {rank} | {_name(m, names, link=False)} | {kind} | "
                     f"{_ms(r['latency_ms'], r['latency_ms'] is not None and r['latency_ms'] == best)} | "
                     f"{_s(r['latency_short_ms'])} | {_s(r['latency_long_ms'])} | {_s(r['latency_bench_ms'])} | "
                     f"{_s(r['latency_av_ms'])} | {_pct(r['gameworld_paused_same'])} | {_pct(r['gameworld_realtime'])} | "
                     f"{_delta(r['gameworld_delta'])} |")
    return ["\n".join(lines) + "\n"]


def realtime_table(merged: dict, models: list[str], rows: dict, names: dict) -> str:
    """GameWorld per game in realtime: the mean progress (the change from paused in brackets)."""
    realtime = merged.get("envs_realtime", {})
    ranked = [m for m in _ranked(models, rows) if rows[m]["gameworld_realtime"] is not None]
    envs = [e for e in rows["_groups"]["suites"]["gameworld"] if e in realtime]
    if not ranked or not envs:
        return ""
    lines = ["## GameWorld in realtime\n",
             "Mean progress per game with the game running for the model's closed-loop latency before each action lands "
             "(the change from paused in brackets; [README.md](README.md#latency-and-realtime) gives the latencies).\n",
             "| Game | " + " | ".join(_name(m, names, link=False) for m in ranked) + " |", "|---|" + "---:|" * len(ranked)]
    for env in envs:
        cells = []
        for m in ranked:
            c, p = realtime[env].get(m), merged["envs"][env].get(m)
            if not _played(c):
                cells.append("–")
                continue
            cells.append(f"{c['score_mean']:.2f}" + (f" ({c['score_mean'] - p['score_mean']:+.2f})" if _played(p) else ""))
        lines.append(f"| `{env.split('/')[-1]}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def calibration(models: list[str], rows: dict, names: dict) -> list[str]:
    """The Brier scores per group, best first, for the models whose report has them."""
    scored = sorted((m for m in models if rows[m]["brier"] is not None), key=lambda m: rows[m]["brier"])
    if not scored:
        return []
    keys = ("brier", "brier_video", "brier_long_video", "brier_benchmarks")
    best = {k: min((rows[m][k] for m in scored if rows[m][k] is not None), default=None) for k in keys}
    g = rows["_groups"]
    lines = ["## Calibration\n",
             f"Brier score of the answer distributions (lower is better): sum over options of (p − y)², 0–2, a question "
             f"without an answer counted as uniform; mean over the {len(g['shared'])} leaves of Static and over its "
             "Video, Long video and Benchmarks groups.",
             "", "| # | Model | Type | Static | Video | Long video | Benchmarks |", "|---:|---|---|---:|---:|---:|---:|"]
    for rank, m in enumerate(scored, start=1):
        kind = TYPES.get(names.get(m, {}).get("type", ""), "")
        lines.append(f"| {rank} | {_name(m, names, link=False)} | {kind} | "
                     + " | ".join(_brier(rows[m][k], rows[m][k] is not None and rows[m][k] == best[k]) for k in keys) + " |")
    missing = [_name(m, names, link=False) for m in _ranked(models, rows) if rows[m]["brier"] is None]
    if missing:
        lines += ["", f"Not yet listed (their sweep reports predate Brier scores): {', '.join(missing)}."]
    return ["\n".join(lines) + "\n"]


def notes(g: dict, brier: bool = False, running: list[str] = (), partial: bool = False, brier_gaps: bool = False,
          speedy: bool = False, page: str = "results/README.md") -> str:
    """The notes under a leaderboard; page is the results page as linked from where the notes stand."""
    return (f"\n- **Static**: accuracy on the {len(g['shared'])} leaves every model takes (no audio); **Video** "
            f"({len(g['video'])} leaves), **Long video** ({len(g['long_video'])}) and **Benchmarks** "
            f"({len(g['benchmarks'])} public subsets) split them. **Audio-visual**: the {len(g['audio'])} leaves that "
            "need audio, for the models that take audio.\n"
            + ("- **Brier ↓**: Brier score of the answer distributions on the same leaves as Static, sum over options of "
               "(p − y)² (0–2, lower is better; a question without an answer counts as uniform)"
               + ("; – where the run's report predates Brier scores" if brier_gaps else "") + ".\n" if brier else "")
            + "- **Closed loop** (paused mode): GameWorld progress and BALROG progression, mean over environments; "
            "VS-Bench return, min-max normalized per game over the models above.\n"
            + ("- **GameWorld realtime**: the same games with the game running for the model's latency before each action "
               "lands; **Latency ↓**: median milliseconds per closed-loop decision at batch size 1 on H100 80 GB PCIe "
               f"([latency and realtime]({page}#latency-and-realtime)).\n" if speedy else "")
            + "- System-One: decision models with their own read-out (a decision head, a scoring server). MLLM: multimodal "
            "LLMs used as classifiers (label-token probabilities)."
            + (" ‡ still running: partial." if partial else "") + "\n"
            + (f"- Running, added when they finish: {', '.join(running)}.\n" if running else ""))


def readme_section(merged: dict, rows: dict, names: dict) -> str:
    models = merged["models"]
    board = leaderboard(models, rows, names)
    return (f"{START}\n## Leaderboard\n\nVideo: {len(models)} models on the 41 video tasks (17,500 items) and the 102 "
            "closed-loop games, one sample per item. Per-leaf and per-environment tables: "
            "[results/README.md](results/README.md).\n\n"
            + board + notes(rows["_groups"], brier_complete(models, rows, names), merged.get("running", []), "‡" in board,
                            any(rows[m]["brier"] is None for m in models), "Latency ↓" in board or "GameWorld realtime" in board)
            + f"\n{END}")


def write(paths: list[Path], names_path: Path, out_dir: Path, readme: Path | None, intro: str,
          latency_dir: Path | None = None) -> list[Path]:
    merged = merge(paths)
    names = load_names(names_path)
    missing = [m for m in merged["models"] if m not in names]
    if missing:
        raise SystemExit(f"no entry in {names_path} for: {', '.join(missing)}")
    # status = "running" in the names file: off the tables until its runs finish (a part-run model would narrow the
    # shared leaves to the ones it has done), listed as running
    merged["models"] = [m for m in merged["models"] if names[m].get("status") != "running"]
    merged["running"] = [n["name"] for n in names.values() if n.get("status") == "running"]
    rows = scores(merged, merged["models"], load_latency(latency_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "README.md", out_dir / "closed_loop.md", out_dir / "summary" / "results.json"]
    written[0].write_text(page(merged, rows, names, intro), encoding="utf-8")
    written[1].write_text(env_tables(merged, merged["models"], rows, names), encoding="utf-8")
    written[2].parent.mkdir(parents=True, exist_ok=True)
    table = {m: {k: v for k, v in rows[m].items() if k not in ("leaves", "leaves_brier")} | {"name": names[m]["name"], "source": merged["sources"][m]}
             for m in merged["models"]}
    written[2].write_text(json.dumps({"groups": rows["_groups"], "models": table, "merged": merged}, indent=1), encoding="utf-8")
    if readme is not None:
        text = readme.read_text(encoding="utf-8")
        section = readme_section(merged, rows, names)
        if START in text and END in text:
            text = text[:text.index(START)] + section + text[text.index(END) + len(END):]
        else:
            raise SystemExit(f"{readme} has no {START} ... {END} block for the results")
        readme.write_text(text, encoding="utf-8")
        written.append(readme)
    return written
