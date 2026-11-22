"""The results page: sweep reports merged, models scored on the leaves they share, VS-Bench normalized per game."""

import json

import pytest

from jevarena_mm import results_page


def report(models, static, envs, overall=None, brier=None):
    return {"plan": "t", "models": models, "overall": overall or {m: {"static_answers": 10, "static_unanswered": 0} for m in models},
            "static": static, **({"static_brier": brier} if brier else {}), "preview": {}, "envs": envs}


def env(score, success=None, episodes=2):
    return {"mode": "paused", "episodes": episodes, "unsupported": 0, "score_mean": score, "success_rate": success}


@pytest.fixture
def files(tmp_path):
    a = report(["a"], {"vid.x/one": {"a": 0.8}, "vid.bench/b": {"a": 0.6}, "lvid.dialog/talk": {"a": 0.5}},
               {"loop.game/vsbench/poker": {"a": env(10.0, 1.0)}, "loop.game/gameworld/g1": {"a": env(0.4)}})
    b = report(["b", "a"], {"vid.x/one": {"b": 0.4, "a": 0.1}, "vid.bench/b": {"b": 0.2, "a": 0.1}},
               {"loop.game/vsbench/poker": {"b": env(-10.0, 0.0), "a": env(0.0)}, "loop.game/gameworld/g1": {"b": env(0.2)}})
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, rep in zip(paths, (a, b)):
        path.write_text(json.dumps(rep))
    names = tmp_path / "models.toml"
    names.write_text('[[model]]\nid = "a"\nname = "Model A"\nrepo = "org/a"\ntype = "system-one"\n'
                     '[[model]]\nid = "b"\nname = "Model B"\ntype = "mllm"\nsource = "recorded"\n')
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\n<!-- results:start -->\nold\n<!-- results:end -->\n\n## Next\n")
    return paths, names, readme


def test_merge_takes_each_model_from_the_first_report(files):
    paths, _, _ = files
    merged = results_page.merge(paths)
    assert merged["models"] == ["a", "b"] and merged["sources"]["a"] == "a.json"
    assert merged["static"]["vid.x/one"] == {"a": 0.8, "b": 0.4}  # a's re-run in b.json is not used

    rep = json.loads(paths[1].read_text())
    rep["models"].append("c")  # in the plan, nothing run yet
    paths[1].write_text(json.dumps(rep))
    assert results_page.merge(paths)["models"] == ["a", "b"]


def test_scores_use_shared_leaves_and_normalize_vsbench(files):
    paths, _, _ = files
    merged = results_page.merge(paths)
    rows = results_page.scores(merged, merged["models"])
    assert rows["_groups"]["shared"] == ["vid.bench/b", "vid.x/one"] and rows["_groups"]["audio"] == ["lvid.dialog/talk"]
    assert rows["a"]["static"] == pytest.approx(0.7) and rows["a"]["audio_visual"] == pytest.approx(0.5)
    assert rows["b"]["audio_visual"] is None  # does not take the audio leaf
    assert rows["a"]["vsbench"] == 1.0 and rows["b"]["vsbench"] == 0.0  # min-max over the models of the page
    assert rows["a"]["gameworld"] == pytest.approx(0.4) and rows["a"]["vsbench_success"] == 1.0


def test_an_environment_over_a_models_input_is_listed_not_partial(files, tmp_path):
    paths, names, readme = files
    for path, cell in zip(paths, ({"a": env(0.6)}, {"b": {"mode": "paused", "episodes": 0, "crashed": 0, "unsupported": 5}})):
        rep = json.loads(path.read_text())
        rep["envs"]["loop.game/gameworld/g2"] = cell
        path.write_text(json.dumps(rep))
    merged = results_page.merge(paths)
    rows = results_page.scores(merged, merged["models"])
    assert rows["b"]["not_taken"] == ["loop.game/gameworld/g2"] and rows["b"]["envs_played"] == (2, 2)
    page = results_page.write(paths, names, tmp_path / "results", None, "")[0].read_text()
    assert "`g2` (Model B)" in page


def test_page_and_readme_block(files, tmp_path):
    paths, names, readme = files
    written = results_page.write(paths, names, tmp_path / "results", readme, "Intro text.")
    page = written[0].read_text()
    assert "| 1 | [Model A](https://huggingface.co/org/a) | System-One | **70.0** |" in page
    assert "Model B" in page and "Intro text." in page
    text = readme.read_text()
    assert "old" not in text and text.count("<!-- results:start -->") == 1 and text.endswith("## Next\n")
    assert "| 2 | Model B | MLLM | 30.0 |" in text


def test_brier_from_the_reports_that_have_it(files, tmp_path):
    paths, names, readme = files
    rep = json.loads(paths[0].read_text())  # a's report has Brier scores; b's (recorded) predates them
    rep["static_brier"] = {"vid.x/one": {"a": 0.3}, "vid.bench/b": {"a": 0.5}, "lvid.dialog/talk": {"a": 0.9}}
    paths[0].write_text(json.dumps(rep))
    merged = results_page.merge(paths)
    rows = results_page.scores(merged, merged["models"])
    assert rows["a"]["brier"] == pytest.approx(0.4) and rows["a"]["brier_benchmarks"] == pytest.approx(0.5)
    assert rows["b"]["brier"] is None
    page = results_page.write(paths, names, tmp_path / "results", readme, "")[0].read_text()
    assert "| Static | Brier ↓ |" in page and "| **0.400** |" in page  # every run model has it: a column (b is recorded)
    assert "## Calibration" in page and "Not yet listed (their sweep reports predate Brier scores): Model B." in page
    names.write_text(names.read_text().replace('source = "recorded"\n', ""))
    page = results_page.write(paths, names, tmp_path / "results", readme, "")[0].read_text()
    assert "Brier ↓" not in page and "## Calibration" in page  # b was run for the release: no column until it has one


def test_a_running_model_stays_off_the_tables(files, tmp_path):
    paths, names, readme = files
    names.write_text(names.read_text().replace('source = "recorded"\n', 'source = "recorded"\nstatus = "running"\n'))
    page = results_page.write(paths, names, tmp_path / "results", readme, "")[0].read_text()
    assert "| Model B" not in page and "Running, added when they finish: Model B." in page
    assert "Running, added when they finish: Model B." in readme.read_text()


def test_latency_and_realtime(files, tmp_path):
    paths, names, readme = files
    realtime = report([], {}, {}) | {"models": ["a", "b"], "envs_realtime": {
        "loop.game/gameworld/g1": {"a": env(0.6) | {"latency_ms": 120.0}, "b": env(0.1) | {"latency_ms": 300.0}}}}
    paths.append(tmp_path / "realtime.json")
    paths[-1].write_text(json.dumps(realtime))
    folder = tmp_path / "latency"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps({"model_id": "a", "median_ms": 120.4, "gpus": 1,
                                               "per_kind": {"short": {"median_ms": 300.0}}}))
    (folder / "a.fit.json").write_text(json.dumps({"model_id": "a", "median_ms": 999.0}))  # a fit test: not the measurement
    (folder / "a.loop.json").write_text(json.dumps({"model_id": "a", "median_ms": 555.0}))  # some kinds only: neither
    merged = results_page.merge(paths)
    assert merged["models"] == ["a", "b"] and merged["realtime_sources"] == {"a": "realtime.json", "b": "realtime.json"}
    rows = results_page.scores(merged, merged["models"], results_page.load_latency(folder))
    assert rows["a"]["gameworld_realtime"] == pytest.approx(0.6) and rows["a"]["gameworld_delta"] == pytest.approx(0.2)
    assert rows["b"]["gameworld_delta"] == pytest.approx(-0.1) and rows["a"]["latency_ms"] == 120.4
    assert rows["a"]["latency_short_ms"] == 300.0 and rows["b"]["latency_ms"] is None
    out = results_page.write(paths, names, tmp_path / "results", readme, "", folder)
    page, loop = out[0].read_text(), out[1].read_text()
    assert "| GameWorld realtime | Latency ↓ |" in page and "## Latency and realtime" in page
    assert "| 1 | Model A | System-One | **120** | 0.30 |" in page  # fastest first; short video in seconds
    assert "## GameWorld in realtime" in loop and "0.60 (+0.20)" in loop and "0.10 (-0.10)" in loop
    assert "Latency ↓" not in results_page.write(paths[:2], names, tmp_path / "plain", None, "")[0].read_text()
