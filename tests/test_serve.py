"""The decision service the closed-loop bridges call, and the upstream suites' play path (no suite installed)."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from jevarena_mm.bridges import common
from jevarena_mm.envs import runner as env_runner
from jevarena_mm.envs.upstream import GameWorldEnvironment
from jevarena_mm.registry import make_model, register_env
from jevarena_mm.serve import PROTOCOL, DecisionServer, request_of

from conftest import make_png

pytest.importorskip("PIL")


def packet(paths, state=None, candidates=("left", "right", "jump")):
    return {"protocol": PROTOCOL, "question": "Which action?",
            "state": state if state is not None else "screen <image:1> then <image:2> now",
            "candidates": [{"id": f"a{i}", "text": c} for i, c in enumerate(candidates)],
            "images": [{"path": str(p), "sha256": common.sha256_file(p)} for p in paths]}


def test_request_places_images_in_order(workspace):
    paths = [make_png(workspace / "s1.png", value=10), make_png(workspace / "s2.png", value=20)]
    request = request_of(packet(paths))
    assert request.state["text"].startswith("screen <image:1>")
    assert len(request.state["images"]) == 2
    assert request.modalities() == {"image"}
    assert list(request.questions["action"].criteria) == ["a0", "a1", "a2"]
    with pytest.raises(ValueError, match="does not place <image:2>"):
        request_of(packet(paths, state="<image:2> before <image:1>"))
    changed = packet(paths)
    make_png(paths[0], value=99)
    with pytest.raises(ValueError, match="changed"):
        request_of(changed)


def test_bridge_round_trip(workspace):
    paths = [make_png(workspace / "s1.png"), make_png(workspace / "s2.png", value=50)]
    log = workspace / "calls.jsonl"
    with DecisionServer(make_model("random"), log=log) as url:
        p = packet(paths)
        result = common.decide(url, p["state"], p["question"], p["candidates"], paths, {"step": 0})
        assert result["selected"] in {"a0", "a1", "a2"}
        assert result["selected"] == max(result["probabilities"], key=result["probabilities"].get)
        forced = common.decide(url, "only one <image:1> <image:2>", "Which?", [{"id": "x", "text": "noop"}], paths)
        assert forced["selected"] == "x" and forced["forced"]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["context"] for r in rows] == [{"step": 0}, {}] and all(r["status"] == "ok" for r in rows)


def test_bridge_records_concurrent_calls(workspace, monkeypatch):
    """Players asked at once (VS-Bench: asyncio.to_thread) each get their own record, also when writes are slow."""
    paths = [make_png(workspace / "s1.png"), make_png(workspace / "s2.png", value=50)]
    write = common.write_json
    monkeypatch.setattr(common, "write_json", lambda path, value: (time.sleep(0.05), write(path, value)))
    monkeypatch.setattr(common, "_RECORD", {"dir": None, "count": 0})
    common.record_to(workspace / "attempt")
    with DecisionServer(make_model("random")) as url, ThreadPoolExecutor(8) as pool:
        p = packet(paths)
        list(pool.map(lambda _: common.decide(url, p["state"], p["question"], p["candidates"], paths), range(8)))
    records = sorted(path.name for path in (workspace / "attempt" / "decisions").iterdir())
    assert records == [f"{n:05d}.json" for n in range(8)]


def test_bridge_unsupported_modality(workspace):
    paths = [make_png(workspace / "s1.png"), make_png(workspace / "s2.png")]
    with DecisionServer(make_model("random@modalities=video")) as url:
        p = packet(paths)
        with pytest.raises(common.Unsupported, match="image"):
            common.decide(url, p["state"], p["question"], p["candidates"], paths)


def test_outcome_file(tmp_path):
    common.outcome(tmp_path, "completed", success=True, score=1.0, decisions=3)
    assert json.loads((tmp_path / "outcome.json").read_text()) == {"status": "completed", "success": True, "score": 1.0, "decisions": 3}


def test_upstream_play_without_installation(workspace, clean_registry, monkeypatch):
    for key in ("JEVARENA_MM_GAMEWORLD", "JEVARENA_MM_GAMEWORLD_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("jevarena_mm.env._loaded", True)  # ignore the developer's .env

    @register_env
    class Toy(GameWorldEnvironment):
        name, leaf, episodes, max_decisions = "loop.game/gameworld/toy", "loop.game/gameworld", 2, 100
        details = {"tasks": ["t1", "t2"]}

    @register_env
    class Wordy(GameWorldEnvironment):
        name, leaf, episodes, max_decisions = "loop.game/gameworld/wordy", "loop.game/gameworld", 2, 100
        unsupported = "free-text actions"

    @register_env
    class Broken(GameWorldEnvironment):
        name, leaf, episodes, max_decisions = "loop.game/gameworld/broken", "loop.game/gameworld", 2, 100
        blocked = "upstream cannot execute its actions"

    model = make_model("random")
    assert env_runner.run_env(model, Broken, [0, 1], "paused", 0.0) == 0
    assert env_runner.read_episodes(model.slug, Broken.name) == []
    assert Toy.installation() is None
    assert env_runner.run_env(model, Toy, [0, 1], "paused", 0.0) == 0
    assert env_runner.read_episodes(model.slug, Toy.name) == []
    assert Toy().bridge_args(1) == ["--game", "toy", "--task", "t2", "--seed", "42"]
    assert env_runner.run_env(model, Wordy, [0, 1], "paused", 0.0) == 2
    assert env_runner.run_env(model, Wordy, [0, 1], "paused", 0.0) == 0  # recorded once
    rows = env_runner.read_episodes(model.slug, Wordy.name)
    assert [r["unsupported"] for r in rows] == ["free-text actions"] * 2


def test_play_shards_and_sweeps_cover_upstream_suites(workspace, clean_registry, monkeypatch):
    from jevarena_mm import cli, sweep

    monkeypatch.setattr("jevarena_mm.env._loaded", True)  # ignore the developer's .env
    for name in ("a", "b", "c"):  # unsupported: recorded without the suite installed
        register_env(type(name, (GameWorldEnvironment,), {
            "name": f"loop.game/gameworld/{name}", "leaf": "loop.game/gameworld", "episodes": 1, "max_decisions": 100,
            "unsupported": "free-text actions"}))
    model = make_model("random")
    cli.main(["play", "--model", "random", "--env", "loop.game/gameworld", "--shard", "1/2"])
    assert [bool(env_runner.read_episodes(model.slug, f"loop.game/gameworld/{n}")) for n in "abc"] == [False, True, False]
    # sweep plays upstream suites through play() (their reset() is not a probe of the installation)
    plan = sweep.Plan(name="t", tasks=[], envs=["loop.game/gameworld"], models=[sweep.ModelPlan(spec="random", static=False)])
    assert sweep.run_model(plan, plan.models[0])["episodes"] == 2
    assert all(env_runner.read_episodes(model.slug, f"loop.game/gameworld/{n}") for n in "abc")
    rep = sweep.report(plan)  # environments the model cannot take stay in the report, apart from the played ones
    assert rep["envs"]["loop.game/gameworld/a"][model.model_id]["unsupported"] == 1 and rep["overall"][model.model_id]["envs"] == 0
    assert "| loop.game/gameworld/a | - |" in sweep.report_markdown(rep)


def test_realtime_play_charges_gameworld_and_skips_suites_without_it(workspace, clean_registry, monkeypatch):
    from jevarena_mm import cli
    from jevarena_mm.envs.upstream import BalrogEnvironment

    monkeypatch.setattr("jevarena_mm.env._loaded", True)  # ignore the developer's .env
    fields = {"episodes": 1, "max_decisions": 100, "unsupported": "free-text actions"}
    register_env(type("g", (GameWorldEnvironment,), {"name": "loop.game/gameworld/g", "leaf": "loop.game/gameworld", **fields}))
    register_env(type("b", (BalrogEnvironment,), {"name": "loop.game/balrog_pixels/b", "leaf": "loop.game/balrog_pixels", **fields}))
    model = make_model("random")
    for name in ("loop.game/gameworld/g", "loop.game/balrog_pixels/b"):
        cli.main(["play", "--model", "random", "--env", name, "--realtime", "--latency-ms", "500"])
    rows = env_runner.read_episodes(model.slug, "loop.game/gameworld/g")
    assert [(r["mode"], r["assumed_latency_ms"]) for r in rows] == [("realtime", 500.0)]
    assert not env_runner.read_episodes(model.slug, "loop.game/balrog_pixels/b")  # no realtime protocol: not played
    assert env_runner.latest_episodes(model.slug, "loop.game/gameworld/g", "paused") == []


def test_a_realtime_episode_gets_its_latency_on_top_of_the_timeout(clean_registry):
    game = type("g", (GameWorldEnvironment,), {"name": "loop.game/gameworld/g", "leaf": "loop.game/gameworld", "max_decisions": 100})
    assert game.episode_timeout_s() == 2700
    assert game.episode_timeout_s(1490.0) == 2700 + 149


def test_a_fixed_latency_is_kept(workspace, monkeypatch):
    monkeypatch.setattr(env_runner, "LATENCY_TABLE", workspace / "latency.json")
    assert env_runner._store_latency("m", 120.0) == 120.0
    assert env_runner._store_latency("m", 300.0) == 120.0


def test_realtime_charges_the_loop_median_measured_on_the_gpu_asked_for(workspace, monkeypatch):
    from jevarena_mm import latency

    model = make_model("random")
    monkeypatch.setattr(env_runner, "LATENCY_TABLE", workspace / "latency.json")
    for gpu, median in (("a100", 287.4), ("h100", 151.0)):
        path = latency.summary_path(model.slug, gpu)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"median_ms": median}))
    assert env_runner.realtime_latency(model, "a100") == (290.0, "a100")
    assert env_runner.realtime_latency(model) == (150.0, "h100")  # none asked for: the H100 measurement
    with pytest.raises(ValueError):
        env_runner.realtime_latency(model, "l40s")
    latency.summary_path(model.slug, "h100").unlink()
    env_runner._store_latency(model.model_id, 400.0)
    assert env_runner.realtime_latency(model) == (400.0, None)  # no H100 measurement: the fixed table


def test_realtime_episodes_of_two_gpus_stay_apart_at_the_same_latency(workspace, clean_registry, monkeypatch):
    from jevarena_mm import cli, latency

    monkeypatch.setattr("jevarena_mm.env._loaded", True)
    register_env(type("g", (GameWorldEnvironment,), {"name": "loop.game/gameworld/g", "leaf": "loop.game/gameworld",
                                                     "episodes": 1, "max_decisions": 100, "unsupported": "free-text actions"}))
    model = make_model("random")
    for gpu, median in (("a100", 301.0), ("h100", 299.0)):  # both round to 300 ms
        path = latency.summary_path(model.slug, gpu)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"median_ms": median}))
        cli.main(["play", "--model", "random", "--env", "loop.game/gameworld/g", "--realtime", "--latency-gpu", gpu])
    rows = env_runner.read_episodes(model.slug, "loop.game/gameworld/g")
    assert [(r["assumed_latency_ms"], r["latency_gpu"]) for r in rows] == [(300.0, "a100"), (300.0, "h100")]
    assert len(env_runner.latest_episodes(model.slug, "loop.game/gameworld/g", "realtime", 100, 300.0, None, "h100")) == 1
    with pytest.raises(SystemExit):  # a GPU without a measurement is an error, not a silent fallback
        cli.main(["play", "--model", "random", "--env", "loop.game/gameworld/g", "--realtime", "--latency-gpu", "l40s"])


def test_a_model_exit_is_a_failed_decision_not_a_dropped_connection(workspace):
    from jevarena_mm.serve import answer

    class Dying:
        model_id, modalities = "dying", frozenset({"image"})

        def decide(self, request, sample=0):
            raise SystemExit("engine died")

    paths = [make_png(workspace / "s1.png", value=10), make_png(workspace / "s2.png", value=20)]
    reply = answer(Dying(), packet(paths))
    assert reply["status"] == "failed" and "engine died" in reply["error"]


def test_bridge_path_survives_a_concurrent_maker(workspace, monkeypatch):
    from pathlib import Path

    from jevarena_mm.envs import upstream

    first = upstream.bridge_path()
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if self.name == "jevarena_mm" else real_exists(self))
    assert upstream.bridge_path() == first  # the link is there already: no FileExistsError


def test_attempt_files_can_live_elsewhere(workspace, clean_registry, monkeypatch, tmp_path):
    from jevarena_mm import env as jenv

    monkeypatch.setenv("JEVARENA_MM_ATTEMPTS", str(tmp_path / "attempts"))
    monkeypatch.setattr("jevarena_mm.env._loaded", True)
    assert jenv.attempts_dir() == tmp_path / "attempts"
    monkeypatch.delenv("JEVARENA_MM_ATTEMPTS")
    assert jenv.attempts_dir() is None


def test_a_stalled_bridge_is_stopped(tmp_path, monkeypatch):
    import sys

    from jevarena_mm.envs import upstream

    monkeypatch.setattr(upstream, "STALL_S", 2)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    with (attempt / "bridge.log").open("w") as log:
        started = __import__("time").time()
        code = upstream.run_watched([sys.executable, "-c", "import time; time.sleep(600)"], attempt, {}, log)
    assert code == "stopped after 2 s without progress" and __import__("time").time() - started < 120
    with (attempt / "bridge.log").open("w") as log:
        assert upstream.run_watched([sys.executable, "-c", "print('done')"], attempt, {}, log) == 0


def test_last_change_skips_files_that_vanish(tmp_path, monkeypatch):
    import os

    from jevarena_mm.envs import upstream

    (tmp_path / "a.log").write_text("x")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "gone").write_text("y")
    real_stat = os.stat

    def flaky(path, *a, **k):
        if str(path).endswith("gone"):
            raise FileNotFoundError(path)
        return real_stat(path, *a, **k)

    monkeypatch.setattr(upstream.os, "stat", flaky)
    assert upstream.last_change(tmp_path) >= real_stat(tmp_path / "a.log").st_mtime


def test_an_input_the_model_rejects_is_unsupported_not_a_failure(workspace):
    from jevarena_mm.schema import ModelResponse
    from jevarena_mm.serve import answer

    class Short:
        model_id, modalities = "short", frozenset({"image"})

        def decide(self, request, sample=0):
            return ModelResponse(error="rejected: schema requires 16955 tokens before state; maximum is 16384")

    paths = [make_png(workspace / "s1.png", value=10), make_png(workspace / "s2.png", value=20)]
    reply = answer(Short(), packet(paths))
    assert reply["status"] == "unsupported" and "16955" in reply["error"]


def test_a_realtime_plan_reports_the_episodes_at_the_measured_latency(workspace, clean_registry, monkeypatch):
    import json as _json

    from jevarena_mm import latency, sweep

    monkeypatch.setattr("jevarena_mm.env._loaded", True)  # ignore the developer's .env

    @register_env
    class Toy(GameWorldEnvironment):
        name, leaf, episodes, max_decisions = "loop.game/gameworld/toy", "loop.game/gameworld", 2, 100
        details = {"tasks": ["t1", "t2"]}

    model = make_model("random")
    path = latency.summary_path(model.slug, "h100")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps({"model_id": model.model_id, "median_ms": 304.0}))  # charged rounded: 300 ms
    for seed, (paused, realtime, other) in enumerate([(0.8, 0.5, 0.0), (0.6, 0.3, 0.0)]):
        Toy.record(model, seed, outcome={"score": paused, "success": False}, decisions=10, failed_decisions=0)
        Toy.record(model, seed, outcome={"score": realtime, "success": False}, decisions=10, failed_decisions=0, mode="realtime", assumed_latency_ms=300.0,
                   latency_gpu="h100")
        Toy.record(model, seed, outcome={"score": other, "success": False}, decisions=10, failed_decisions=0, mode="realtime", assumed_latency_ms=500.0,
                   latency_gpu="h100")  # played at another latency: not this measurement's
    plan = sweep.Plan(name="t", tasks=[], envs=["loop.game/gameworld"], models=[sweep.ModelPlan(spec="random", static=False)],
                      realtime=True)
    rep = sweep.report(plan)
    cell = rep["envs_realtime"]["loop.game/gameworld/toy"][model.model_id]
    assert cell["episodes"] == 2 and cell["score_mean"] == pytest.approx(0.4) and cell["latency_ms"] == 300.0
    assert cell["latency_gpu"] == "h100" and rep["envs"]["loop.game/gameworld/toy"][model.model_id]["score_mean"] == pytest.approx(0.7)
    assert "## Environments in realtime" in sweep.report_markdown(rep)
    assert "envs_realtime" not in sweep.report(sweep.Plan(name="t", tasks=[], envs=["loop.game/gameworld"],
                                                          models=[sweep.ModelPlan(spec="random", static=False)]))
