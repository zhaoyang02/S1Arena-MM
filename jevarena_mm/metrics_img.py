"""Scoring of the image tasks, as the image rows of the paper were computed, so that those results are reproduced.

score_model() is `jevarena-mm img-score`, results_md() the tables of `jevarena-mm img-compare`. It differs from
jevarena_mm.metrics, which scores the video and audio tasks, in three ways:
- an item's questions are averaged first and a leaf pools the items of all its tasks (metrics.py averages questions,
  and a leaf there is the mean over its tasks);
- a failed call or a missing answer scores the Brier of an all-zero distribution (1 for a one-hot target) and
  confidence 0 (metrics.py: a uniform distribution);
- a reply's distribution is renormalized before scoring, and a boolean answer is "true" when P(true) >= 0.5.

The metrics:

accuracy  argmax is gold (or in gold.accept); boolean: P(true) >= 0.5 is "true"; avg@k over samples, maj@k by vote
          an item's questions are averaged first, so every item weighs the same in its leaf (the 64-label privacy
          item counts once); a domain, and the overall score, is the mean of its leaves
brier     sum over options of (p_k - t_k)^2, as qev.calibrate.brier (boolean = the two options yes/no). The target t is
          the annotators' distribution where the item has one (gold.dist: KonIQ, ArtEmis, New Yorker), else one-hot
          on the gold. Keys in gold.accept count as one outcome: their probabilities are summed before scoring.
brier_hard the same with a one-hot target everywhere (reported alongside)
ece       10 equal-mass bins of top-1 confidence vs correctness
Failures (failed call, missing question, unusable reply) score accuracy 0 and the Brier of an all-zero distribution
(sum of t_k^2: 1 for a one-hot target).
Items not run, and items the model cannot take (status "unsupported"), are left out of that model's scores; coverage =
share of the leaf's items the model answered or failed, so a model cannot gain by declining a modality.
A reply's distribution is renormalized to sum to 1 before scoring (some servers round their probabilities).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

BENCHMARK_DOMAINS = ("std",)   # standard benchmarks: own domain and leaf tables, not in the overall score


def _top(dist: dict) -> tuple[str, float]:
    k = max(dist, key=dist.get)
    return k, dist[k]


def target(q: dict, gold: dict, keys: list[str], soft: bool = True) -> dict[str, float]:
    """Target distribution over `keys`: gold.dist (normalized) when soft and present, else one-hot on the gold."""
    d = gold.get("dist") if soft else None
    if d is not None:
        d = {str(i): float(v) for i, v in enumerate(d)} if isinstance(d, list) else {str(k): float(v) for k, v in d.items()}
        tot = sum(d.values())
        if tot > 0:
            return {k: d.get(k, 0.0) / tot for k in keys}
    return {k: float(k == str(gold["label"])) for k in keys}


def brier(p: dict[str, float], t: dict[str, float], merge: set[str] | None = None) -> float:
    """sum_k (p_k - t_k)^2; keys in `merge` (gold.accept with the gold) are one outcome."""
    if merge and len(merge) > 1:
        p = {**{k: v for k, v in p.items() if k not in merge}, "@accept": sum(p.get(k, 0.0) for k in merge)}
        t = {**{k: v for k, v in t.items() if k not in merge}, "@accept": sum(t.get(k, 0.0) for k in merge)}
    return sum((p.get(k, 0.0) - t.get(k, 0.0)) ** 2 for k in set(p) | set(t))


def decision(q: dict, gold: dict, dist: dict | None) -> dict:
    """One question of one sample -> {correct, brier, brier_hard, conf, pred}."""
    goldkey = str(gold["label"])
    ok_keys = {goldkey, *map(str, gold.get("accept") or [])}
    if q["type"] == "boolean":
        keys = ["true", "false"]
    elif q["type"] == "score":
        keys = [str(i) for i in range(len(q["criteria"]))]
    else:
        keys = list(q["criteria"])
    if not dist:
        t = target(q, gold, keys)
        return {"correct": 0.0, "brier": brier({}, t, ok_keys), "brier_hard": brier({}, target(q, gold, keys, False), ok_keys),
                "conf": 0.0, "pred": None}
    dist = {str(k): float(v) for k, v in dist.items()}
    if (tot := sum(dist.values())) > 0:
        dist = {k: v / tot for k, v in dist.items()}
    if q["type"] == "boolean":
        pred = "true" if dist.get("true", 0.0) >= 0.5 else "false"
        conf = max(dist.get("true", 0.0), dist.get("false", 0.0))
    else:
        pred, conf = _top(dist)
    keys = keys + [k for k in dist if k not in keys]
    return {"correct": float(pred in ok_keys), "brier": brier(dist, target(q, gold, keys), ok_keys),
            "brier_hard": brier(dist, target(q, gold, keys, False), ok_keys), "conf": conf, "pred": pred}


def ece(pairs: list[tuple[float, float]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    pairs = sorted(pairs)
    n, tot = len(pairs), 0.0
    for b in range(bins):
        chunk = pairs[b * n // bins:(b + 1) * n // bins]
        if chunk:
            tot += len(chunk) / n * abs(sum(c for c, _ in chunk) / len(chunk) - sum(y for _, y in chunk) / len(chunk))
    return tot


def mean_of(leaves: list[dict], k: int) -> dict:
    """Domain or overall = unweighted mean of its leaves; coverage = covered items / items."""
    keys = (f"avg@{k}", f"maj@{k}", "brier", "brier_hard", "ece", "latency_ms_mean")
    out = {key: (sum(v) / len(v) if (v := [m[key] for m in leaves if m[key] is not None]) else None) for key in keys}
    out.update(leaves=len(leaves), items=sum(m["items"] for m in leaves), covered=sum(m["covered"] for m in leaves))
    out["coverage"] = out["covered"] / out["items"] if out["items"] else None
    return out


def score_items(items: list[dict], runs: dict[tuple[str, int], dict], k: int = 1) -> dict:
    """items of one leaf (or task) + cached runs -> metrics with coverage and per-question accuracy."""
    acc, bri, bri_h, lat, cal, per_q = [], [], [], [], [], defaultdict(list)
    maj = []
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    covered = 0
    for it in items:
        samples = [runs.get((it["id"], s)) for s in range(k)]
        if all(r is None or r.get("status") == "unsupported" for r in samples):
            continue   # not run, media missing, or a modality the model does not take: neither covered nor failed
        covered += 1
        votes = defaultdict(Counter)
        i_acc, i_brier, i_brier_h = [], [], []
        for r in samples:
            ans = (r or {}).get("answers", {}) if (r or {}).get("status", "failed") == "ok" else {}
            if r and r.get("latency_ms"):
                lat.append(r["latency_ms"])
            for name, q in it["request"]["questions"].items():
                d = decision(q, it["gold"][name], ans.get(name))
                i_acc.append(d["correct"])
                i_brier.append(d["brier"])
                i_brier_h.append(d["brier_hard"])
                cal.append((d["conf"], d["correct"]))
                per_q[name].append(d["correct"])
                votes[name][d["pred"]] += 1
        acc.append(mean(i_acc))
        bri.append(mean(i_brier))
        bri_h.append(mean(i_brier_h))
        i_maj = []
        for name, c in votes.items():
            g = it["gold"][name]
            top = c.most_common(1)[0][0]
            i_maj.append(float(top is not None and top in {str(g["label"]), *map(str, g.get("accept") or [])}))
        maj.append(mean(i_maj))
    return {"items": len(items), "covered": covered, "coverage": covered / len(items) if items else None,
            f"avg@{k}": mean(acc), f"maj@{k}": mean(maj), "brier": mean(bri), "brier_hard": mean(bri_h), "ece": ece(cal),
            "latency_ms_mean": mean(lat), "per_question": {q: mean(v) for q, v in sorted(per_q.items())}}


# The harness side: its items and run rows in the form the functions above read.

def parts(leaf: str) -> tuple[str, str, str]:
    mod, rest = leaf.split(".", 1)
    dom, name = rest.split("/")
    return mod, dom, name


def as_item(item) -> dict:
    """An Item as this scoring reads it (request, gold and id as stored)."""
    return json.loads(item.model_dump_json())


def runs_of(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """Run rows of the harness -> the run cache this scoring reads: (item id, sample) -> response, with each answer as
    its distribution {option: p}. Later rows replace earlier ones."""
    out = {}
    for row in rows:
        response = dict(row["response"])
        response["answers"] = {name: answer["probs"] for name, answer in (response.get("answers") or {}).items()}
        out[(row["item_id"], row.get("sample", 0))] = response
    return out


def score_model(items: list[dict], runs: dict[tuple[str, int], dict], k: int = 1) -> dict:
    """One model's scores (`jevarena-mm img-score`): per leaf (its items pooled over tasks), per domain (mean of its
    leaves) and overall (mean of the taxonomy leaves; the standard benchmarks, img.std, are reported on their own)."""
    by = defaultdict(list)
    for it in items:
        by[it["leaf"]].append(it)
    out = {"k": k, "overall": {}, "domain": {}, "leaf": {}}
    for leaf, its in sorted(by.items()):
        out["leaf"][leaf] = score_items(its, runs, k)
    groups = defaultdict(list)
    for leaf, m in out["leaf"].items():
        groups[".".join(parts(leaf)[:2])].append(m)   # a leaf not run lowers coverage; its scores are absent, not 0
    out["domain"] = {g: mean_of(ms, k) for g, ms in sorted(groups.items())}
    out["overall"] = mean_of([m for leaf, m in out["leaf"].items() if parts(leaf)[1] not in BENCHMARK_DOMAINS], k)
    return out


def model_name(spec: str) -> str:
    return dict(kv.split("=", 1) for kv in spec.partition("@")[2].split(",") if "=" in kv).get("model", spec)


def results_md(rows: list[dict], k: int, h: str = "##") -> list[str]:
    """Overall, per-domain and per-leaf tables (one leaf table per domain); the best value of each column in bold
    (`jevarena-mm img-compare`)."""
    acc = f"avg@{k}"

    def cell(m, best):
        if not m or m[acc] is None:
            return ""
        v = f"{m[acc]:.3f}"
        v = f"**{v}**" if best is not None and round(m[acc], 3) == best else v
        return v + ("" if m["coverage"] == 1 else f" ({m['coverage']:.2f})")

    def table(cols: list[tuple[str, callable]]) -> list[str]:
        best = [max((round(m[acc], 3) for r in rows if (m := get(r)) and m[acc] is not None), default=None)
                for _, get in cols]
        out = ["| model | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
        out += [f"| {model_name(r['model'])} | " + " | ".join(cell(get(r), b) for (_, get), b in zip(cols, best)) + " |"
                for r in rows]
        return out + [""]

    L = [f"{h} Overall\n", f"| # | model | {acc} | maj@{k} | Brier | Brier (one-hot) | ECE | coverage |",
         "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        m = r["overall"]
        L.append(f"| {i} | {model_name(r['model'])} | {m[acc]:.3f} | {m[f'maj@{k}']:.3f} | {m['brier']:.3f} | "
                 f"{m['brier_hard']:.3f} | {m['ece']:.3f} | {m['coverage']:.3f} |")
    doms = sorted({x for r in rows for x in r["domain"]})
    L += ["", "Overall is the mean of the taxonomy leaves; the standard benchmarks ("
          + ", ".join(f"img.{d}" for d in BENCHMARK_DOMAINS) + ") are reported in their own tables.\n"]
    L += [f"{h} By domain\n", "Each domain is the unweighted mean of its leaves.\n"]
    L += table([(d.split(".", 1)[1], lambda r, d=d: r["domain"].get(d)) for d in doms])
    L += [f"{h} By leaf\n", "One table per domain: the domain mean, then each leaf.\n"]
    for d in doms:
        leaves = sorted({x for r in rows for x in r["leaf"] if x.startswith(d + "/")})
        L += [f"{h}# {d}\n"]
        L += table([("**mean**", lambda r, d=d: r["domain"].get(d))] +
                   [(x.split("/", 1)[1], lambda r, x=x: r["leaf"].get(x)) for x in leaves])
    return L
