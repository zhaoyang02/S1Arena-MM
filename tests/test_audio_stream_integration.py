"""Exercise the migrated CLI and historical reports without model calls."""

from copy import deepcopy
import hashlib
import json

import pytest

from conftest import make_wav
from jevarena_mm import cli, registry
from jevarena_mm.streaming import qwen, runner
from jevarena_mm.streaming.aggregate import aggregate
from jevarena_mm.streaming.catalog import TASKS
from jevarena_mm.streaming.core import summarize
from jevarena_mm.streaming.manifest import load_manifest
from jevarena_mm.streaming.report import public_report


@pytest.fixture
def manifest(tmp_path):
    audio = make_wav(tmp_path / "media" / "input.wav", seconds=0.1, rate=16000)
    spec = {
        "path": "media/input.wav",
        "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "duration_s": 0.1,
    }
    cases = []
    for category in TASKS:
        for index in range(2):
            stream = category.startswith("candor_")
            cases.append(
                {
                    "id": f"{category}-{index}",
                    "category": category,
                    "mode": "stream" if stream else "reactive",
                    "audio": spec,
                    "gold_events": [
                        {
                            "type": "turn"
                            if category == "candor_turn_taking"
                            else "pause",
                            "interval_s": [0, 0.1],
                        }
                    ]
                    if stream
                    else [],
                    **(
                        {}
                        if stream
                        else {"user_event": spec, "trigger_after_audible_s": 1.0}
                    ),
                }
            )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {"version": "audio-live-v1", "selection": {"per_task": 2}, "cases": cases}
        )
    )
    return path


@pytest.fixture
def fake_transport(monkeypatch):
    calls = []

    async def fake_run(self, url):
        calls.append(self.case["id"])
        self.out.mkdir(parents=True)
        trace = [
            {"event": "observation_end", "t": 1, "buffered_s": 0, "active_responses": 1}
        ]
        (self.out / "events.jsonl").write_text("\n".join(json.dumps(e) for e in trace))
        # Cleanup-only output must not turn a silent observation into a success.
        return {
            "id": self.case["id"],
            "category": self.case["category"],
            "mode": self.case["mode"],
            "model": qwen.MODEL,
            "status": "completed",
            "error_type": None,
            "metrics": summarize(self.case, trace, [[2, 3]]),
            "playback_intervals": [[2, 3]],
        }

    monkeypatch.setattr(qwen.Episode, "run", fake_run)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "test")
    return calls


def test_stream_cli_round_trip(manifest, tmp_path, fake_transport):
    run_dir = tmp_path / "run"
    cli.main(
        [
            "stream-run",
            "--manifest",
            str(manifest),
            "--out",
            str(run_dir),
            "--backend",
            "qwen",
            "--limit",
            "1",
        ]
    )
    assert len(fake_transport) == 4
    raw = (run_dir / "progress.json").read_bytes()
    assert json.loads(raw)["state"] == "completed"
    report_dir = tmp_path / "report"
    cli.main(
        [
            "stream-score",
            "--manifest",
            str(manifest),
            "--run",
            str(run_dir),
            "--out",
            str(report_dir),
        ]
    )
    report = json.loads((report_dir / "report.json").read_text())
    assert report["aggregate"]["planned"] == report["aggregate"]["completed"] == 4
    assert all(
        row["output_observed"] == 0 and row["right_censored"] == 1
        for row in report["aggregate"]["categories"]
    )
    assert (run_dir / "progress.json").read_bytes() == raw
    assert report["analysis"]["raw_progress_sha256"] == hashlib.sha256(raw).hexdigest()
    assert all("playback_intervals" not in r for r in report["results"])
    with pytest.raises(ValueError, match="new output directory"):
        cli.main(
            [
                "stream-run",
                "--manifest",
                str(manifest),
                "--out",
                str(run_dir),
                "--backend",
                "qwen",
            ]
        )
    assert len(fake_transport) == 4


def test_failed_transport_stops_without_retry(
    manifest, tmp_path, fake_transport, monkeypatch
):
    original = qwen.Episode.run

    async def fail(self, url):
        row = await original(self, url)
        return {**row, "status": "failed", "error_type": "ConnectionError"}

    monkeypatch.setattr(qwen.Episode, "run", fail)
    path = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="episode failed"):
        cli.main(
            [
                "stream-run",
                "--manifest",
                str(manifest),
                "--out",
                str(path),
                "--backend",
                "qwen",
            ]
        )
    progress = json.loads((path / "progress.json").read_text())
    assert progress["state"] == "failed"
    assert len(fake_transport) == 1
    assert len(progress["pending_ids"]) == 7
    a = aggregate(load_manifest(manifest), progress)
    assert (a["completed"], a["failed"], a["pending"]) == (0, 1, 7)


@pytest.mark.parametrize(
    "change", ["hash", "escape", "duplicate", "mode", "trigger", "duration"]
)
def test_invalid_input_rejected_before_transport(
    manifest, tmp_path, fake_transport, change
):
    data = json.loads(manifest.read_text())
    c = data["cases"][0]
    if change == "hash":
        c["audio"]["sha256"] = "0" * 64
    if change == "escape":
        c["audio"]["path"] = "../input.wav"
    if change == "duplicate":
        data["cases"].append(deepcopy(c))
    if change == "mode":
        c["mode"] = "reactive"
    if change == "trigger":
        data["cases"][-1]["trigger_after_audible_s"] = 0
    if change == "duration":
        c["audio"]["duration_s"] = 0.2
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        cli.main(
            [
                "stream-run",
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / "invalid"),
                "--backend",
                "qwen",
            ]
        )
    assert not fake_transport
    assert not (tmp_path / "invalid").exists()


def test_streaming_registered_but_not_offline_tasks(clean_registry, capsys):
    classes = registry.envs_under("st.audio")
    assert {c.name for c in classes} == set(TASKS.values())
    assert not registry.tasks_under("st.audio")
    assert len(registry.tasks_under("aud")) == 21
    cli.main(["play", "--env", "st.audio", "--model", "random"])
    assert "use stream-run" in capsys.readouterr().out


def test_unknown_status_not_hidden_as_finished(manifest):
    m = load_manifest(manifest)
    case = m["cases"][0]
    with pytest.raises(ValueError, match="status"):
        aggregate(
            m,
            {
                "results": [
                    {
                        "id": case["id"],
                        "category": case["category"],
                        "status": "unknown",
                    }
                ]
            },
        )


def test_observation_deadline_clips_crossing_audio():
    trace = [{"event": "observation_end", "t": 1, "buffered_s": 0}]
    metrics = summarize({"mode": "reactive"}, trace, [[0.5, 2]])
    assert metrics["audible_playback_s"] == 0.5
    assert not metrics["triggered"]


def test_public_export_excludes_private_fields(manifest):
    m = load_manifest(manifest)
    report = public_report(
        m,
        {
            "model": "test",
            "workers": 1,
            "manifest_sha256": "hash",
            "results": [],
            "transcript": "PRIVATE",
            "resume_token": "PRIVATE",
            "config": {"key": "PRIVATE"},
        },
    )
    assert "PRIVATE" not in json.dumps(report)
