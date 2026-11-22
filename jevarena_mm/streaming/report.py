"""Aggregate streaming diagnostics without folding them into offline accuracy."""

from copy import deepcopy
import json
from pathlib import Path

from .aggregate import aggregate
from .catalog import PROTOCOL, TASKS
from .core import summarize
from .manifest import digest, load_manifest, write_json


def rescore(manifest, progress, run_dir):
    """Read raw traces, excluding cleanup after the observation deadline; never edit originals."""
    progress = deepcopy(progress)
    cases = {c["id"]: c for c in manifest["cases"]}
    seen = set()
    for result in progress["results"]:
        ident = result["id"]
        if ident not in cases or ident in seen:
            raise ValueError("Unknown or duplicate result ID")
        seen.add(ident)
        # Historical Gemini suite logs were grouped into category directories.
        directory = run_dir / ident
        if not directory.exists():
            directory = run_dir / cases[ident]["category"] / ident
        trace = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if (
            not any(e["event"] == "observation_end" for e in trace)
            and result["status"] == "completed"
        ):
            raise ValueError(f"Completed episode has no observation deadline: {ident}")
        result["metrics"] = summarize(cases[ident], trace, result["playback_intervals"])
        end = next((e for e in trace if e['event'] == 'observation_end'), {})
        if progress.get('backend') in ('qwen', 'minicpm', 'jev', 'controller'):
            active = end.get('active_responses', 0) > 0
            result['metrics']['active_generation_at_deadline'] = active
            result['metrics']['observation_censored'] |= active
        if progress.get('backend') in ('jev', 'controller'):
            pending = bool(end.get('controller_pending') or end.get('controller_queued'))
            result['metrics']['controller_pending_at_deadline'] = pending
            result['metrics']['observation_censored'] |= pending
    return progress


def public_report(manifest, progress):
    """Explicit public fields only: no PCM, transcripts, event timelines or private paths."""
    selected = progress.get("case_ids", [c["id"] for c in manifest["cases"]])
    known = {c["id"]: c for c in manifest["cases"]}
    if len(set(selected)) != len(selected) or not set(selected) <= known.keys():
        raise ValueError("Unknown or duplicate selected case ID")
    scoped = {**manifest, "cases": [known[i] for i in selected]}
    summary = aggregate(scoped, progress)
    for row in summary["categories"]:
        row["env"] = TASKS[row["category"]]
    rows = [
        {
            k: r[k]
            for k in (
                "id",
                "category",
                "mode",
                "status",
                "error_type",
                "model",
                "metrics",
            )
            if k in r
        }
        for r in progress["results"]
    ]
    for row in rows:
        row["env"] = TASKS[row["category"]]
    report = {
        "report_schema": PROTOCOL,
        "model": progress["model"],
        "backend": progress.get("backend"),
        "run_protocol": progress.get("protocol", "audio-live-v1"),
        "state": progress.get("state", progress.get("suite_state")),
        "workers": progress["workers"],
        "manifest_sha256": progress["manifest_sha256"],
        "code_sha256": progress.get("code_sha256", {}),
        "runtime": progress.get("runtime", {}),
        "selection": manifest["selection"],
        "analysis": progress.get("analysis", {}),
        "aggregate": summary,
        "results": rows,
        "input_fingerprints": [
            {
                "id": c["id"],
                "audio_sha256": c["audio"]["sha256"],
                **(
                    {"user_event_sha256": c["user_event"]["sha256"]}
                    if c["mode"] == "reactive"
                    else {}
                ),
            }
            for c in scoped["cases"]
        ],
        "interpretation": [
            "Completed means the observation finished, not that the model passed the behavior.",
            "Timing statistics are conditional on observed events; failures, silence and missing triggers remain visible.",
            "Backchannel cancellation is undesirable; interruption cancellation is expected.",
            "Scripted virtual playback diagnostics, not official FDB scores, physical device latency or semantic success.",
            "Development sample: the first eligible source IDs of each task, in numeric order.",
        ],
    }

    strata = progress.get('analysis', {}).get('reused_runtime_strata', [])
    new_ids = progress.get('analysis', {}).get('new_runtime_case_ids', [])
    report['runtime_strata'] = []
    for entry in [*strata, {'case_ids': new_ids, 'runtime': progress.get('runtime', {}),
                            'code_sha256': progress.get('code_sha256', {}), 'source': 'current'}]:
        ids = set(entry['case_ids'])
        if not ids:
            continue
        scoped_manifest = {**manifest, 'cases': [c for c in manifest['cases'] if c['id'] in ids]}
        scoped_progress = {**progress, 'results': [r for r in progress['results'] if r['id'] in ids]}
        report['runtime_strata'].append({**entry, 'aggregate': aggregate(scoped_manifest, scoped_progress)})
    report["controller_service"] = {k: v for k, v in (progress.get("controller_service") or {}).items() if k != "model_spec"}
    if progress.get("backend") == "controller":
        report["interpretation"].append(
            "Decision model with shared external Qwen speech and endpoint VAD; not native streaming ability. "
            "Controller and voice runtimes jointly determine latency; report separately from native-voice systems."
        )
    if progress.get("backend") == "jev":
        report["interpretation"].append(
            "Jev-Omni uses causal local VAD and an external Qwen voice service; not native Jev speech. "
            "Automatic backend VAD/cancellation is disabled; Jev acts at detected endpoints. "
            "Known binary option-position sensitivity remains; no semantic recovery score is claimed."
        )
    return report


def render(report):
    a = report["aggregate"]
    lines = [
        f"# Audio streaming: {report['model']}",
        "",
        f"{a['completed']}/{a['planned']} completed; {a['failed']} failed; {a['pending']} pending. "
        f"Workers: {report['workers']}. Completed episodes are not behavioral successes.",
        "",
        "| Environment | Completed / planned | Output observed | Diagnostic |",
        "|---|---:|---:|---|",
    ]
    for row in a["categories"]:
        if row["category"] == "candor_turn_taking":
            t = row["first_audio_after_boundary_ms"]
            detail = f"First audio median {t['median']} ms (n={t['n']}); {row['events_with_early_playback']} early boundaries"
        elif row["category"] == "candor_pause_handling":
            detail = f"Overlap at {row['pause_events_with_overlap']}/{row['scorable_events']} marked pauses"
        else:
            t = row["provider_interrupt_after_trigger_ms"]
            detail = (
                f"{row['triggered']} triggers; {row['provider_interruption_signals']} cancellation signals; "
                f"dispatch-to-signal median {t['median']} ms (n={t['n']})"
            )
            if row["category"] == "user_backchannel":
                detail = "Unwanted cancellation: " + detail
        lines.append(
            f"| `{row['env']}` | {row['completed']}/{row['planned']} | {row['output_observed']} | {detail} |"
        )
    lines += [
        "",
        f"Right-censored episodes: {sum(r['right_censored'] for r in a['categories'])}.",
        "",
    ]
    lines += ["- " + note for note in report["interpretation"]]
    return "\n".join(lines) + "\n"


def score(args):
    manifest = load_manifest(args.manifest)
    path = args.run / "progress.json"
    progress = json.loads(path.read_text())
    if digest(args.manifest) != progress["manifest_sha256"]:
        raise ValueError("Run/manifest fingerprint mismatch")
    progress = rescore(manifest, progress, args.run)
    progress["analysis"] = {
        "raw_progress_sha256": digest(path),
        "scorer_sha256": digest(Path(__file__).with_name("core.py")),
        "rule": "Clip traces and audible playback at observation_end; preserve raw results",
    }
    report = public_report(manifest, progress)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "report.json", report)
    (args.out / "report.md").write_text(render(report), encoding="utf-8")
    print(render(report))
