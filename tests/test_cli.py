import json

import pytest

from conftest import make_png
from jevarena_mm import cli, dataio, env, media, runner
from jevarena_mm.registry import register_task
from jevarena_mm.schema import DecisionRequest, Gold, Item, Question
from jevarena_mm.tasks.base import TaskAdapter

pytest.importorskip("PIL")


class FrozenToy(TaskAdapter):
    name = "img.basic/toy/frozen"
    leaf = "img.basic/toy"
    source = "Toy pictures (generated in the tests)"
    license = "CC0"
    tier = "C"
    media_root = "toy"
    frozen = True


def write_frozen_items(workspace) -> list[Item]:
    items = []
    for n in range(3):
        make_png(workspace / "media" / "toy" / f"{n}.png", value=40 * n)
        ref = media.make_ref("image", f"toy/{n}.png")
        items.append(Item(
            leaf=FrozenToy.leaf, task=FrozenToy.name,
            request=DecisionRequest(state={"image": ref}, questions={"bright": Question(type="boolean", instructions="Is it bright?")}),
            gold={"bright": Gold(label="true" if n == 2 else "false")},
            meta={"source": FrozenToy.source, "license": FrozenToy.license, "tier": FrozenToy.tier},
        ))
    dataio.write_items(dataio.task_path(FrozenToy.name), items)
    return items


def test_paths_carry_the_modality(workspace):
    assert dataio.task_path("vid.basic/temporal/ssv2") == env.DATA / "vid" / "basic" / "temporal" / "ssv2.jsonl"
    assert dataio.domain_dir("lvid.know") == env.DATA / "lvid" / "know"
    with pytest.raises(ValueError):
        dataio.task_path("basic/temporal/ssv2")


def test_build_verify_run_score_cards(workspace, clean_registry, capsys):
    register_task(FrozenToy)
    write_frozen_items(workspace)

    cli.main(["build", "--leaf", "img.basic"])
    manifest = json.loads((env.DATA / "img" / "basic" / "manifest.json").read_text())
    assert manifest[0]["name"] == FrozenToy.name and manifest[0]["items"] == 3
    assert manifest[0]["modalities"] == ["image"] and len(manifest[0]["sha256"]) == 64 and manifest[0]["frozen"]

    cli.main(["verify", "--leaf", "img.basic"])
    assert "3 items, 3 media files: ok" in capsys.readouterr().out

    cli.main(["run", "--model", "random@modalities=video", "--leaf", "img.basic"])
    cli.main(["run", "--model", "random", "--leaf", "img.basic"])
    capsys.readouterr()
    cli.main(["score", "--models", "random", "random@modalities=video", "--leaf", "img.basic"])
    out = capsys.readouterr().out
    assert "cov=100%" in out and "unsupported (3 items)" in out

    cli.main(["cards", "--leaf", "img.basic"])
    assert "| image | 3 | 2 |" in capsys.readouterr().out

    (workspace / "media" / "toy" / "1.png").unlink()
    with pytest.raises(SystemExit):
        cli.main(["verify", "--leaf", "img.basic"])


def test_check_finds_inconsistent_data_and_docs(workspace, clean_registry, monkeypatch, capsys):
    clean_registry.TASKS.clear()
    clean_registry.ENVS.clear()
    register_task(FrozenToy)
    monkeypatch.setattr(env, "ROOT", workspace)
    (workspace / "docs").mkdir()
    items = write_frozen_items(workspace)
    cli.main(["build", "--leaf", "img.basic"])
    cli.main(["cards", "--leaf", "img.basic", "--out", str(workspace / "docs" / "tasks_img_basic.md")])
    cli.main(["check"])
    assert "1 item files checked" in capsys.readouterr().out

    path = dataio.task_path(FrozenToy.name)
    edited = items[0].model_copy(deep=True)
    edited.gold["bright"] = Gold(label="true")  # gold changed but id kept: no longer the content id
    dataio.write_items(path, [edited, *items[1:]])
    (workspace / "docs" / "tasks_img_basic.md").write_text("stale\n")
    with pytest.raises(SystemExit):
        cli.main(["check"])
    out = capsys.readouterr().out
    assert "1 item ids are not content ids" in out
    assert "file sha256 differs from the manifest" in out
    assert "docs/tasks_img_basic.md is out of date" in out


def test_sharded_runs_split_the_items(workspace, clean_registry):
    register_task(FrozenToy)
    write_frozen_items(workspace)
    cli.main(["build", "--leaf", "img.basic"])
    cli.main(["run", "--model", "random", "--leaf", "img.basic", "--shard", "0/2"])
    cli.main(["run", "--model", "random", "--leaf", "img.basic", "--shard", "1/2"])
    folder = env.RUNS / "test__random"
    assert sorted(p.name for p in folder.iterdir()) == ["img.basic__toy__frozen.shard-0of2.jsonl",
                                                        "img.basic__toy__frozen.shard-1of2.jsonl"]  # one file per shard
    rows = runner.read_runs("test__random", FrozenToy.name)
    assert sorted(r["item_id"] for r in rows) == sorted(i.id for i in dataio.load_task(FrozenToy()))  # each item once
    cli.main(["run", "--model", "random", "--leaf", "img.basic"])  # the cached shards count as done
    assert len(runner.read_runs("test__random", FrozenToy.name)) == len(rows)


def test_sweep_submit_prints_one_array_per_model(tmp_path, capsys):
    plan = tmp_path / "plan.toml"
    plan.write_text('name = "t"\ntasks = ["vid"]\nenvs = ["loop"]\n[defaults]\nk = 1\nslurm = { time = "12:00:00" }\n'
                    '[[models]]\nspec = "random"\nworkers = 2\nslurm = { array = 3, name = "t-random" }\n'
                    '[[models]]\nspec = "random@modalities=image+video"\nstatic = false\nslurm = { gpus_per_shard = 2, fill = "random@modalities=audio" }\n'
                    '[[models]]\nspec = "random@modalities=audio"\nslurm = { array = 0 }\n')
    cli.main(["sweep-submit", str(plan), "-p", "gpu", "-A", "acct"])
    first, second = capsys.readouterr().out.strip().splitlines()
    assert "JV_MODEL=random " in first and "JV_LEAVES=vid JV_ENVS=loop" in first and "--array=0-2" in first
    assert "JV_RUN_ARGS='--workers 2'" in first and "JV_RUN_ARGS=''" in second
    assert "-J t-random" in first and "-t 12:00:00" in first and "-A acct" in first and first.endswith("scripts/slurm/run_shards.slurm")
    assert "JV_MODEL=random@modalities=image+video" in second and "JV_LEAVES='' JV_ENVS=loop" in second
    assert "JV_GPUS_PER_SHARD=2 JV_FILL_MODEL=random@modalities=audio" in second and "--array=0-0" in second


def test_sweep_report_has_brier_scores_per_leaf(workspace, clean_registry):
    from jevarena_mm import metrics, sweep

    class Released(FrozenToy):
        status = "released"

    register_task(Released)
    items = write_frozen_items(workspace)
    cli.main(["build", "--leaf", "img.basic"])
    cli.main(["run", "--model", "random", "--leaf", "img.basic"])
    plan = sweep.Plan(name="t", tasks=["img.basic"], envs=[], models=[sweep.ModelPlan(spec="random")])
    rep = sweep.report(plan)
    (model,) = rep["models"]
    expected = metrics.task_metrics(Released(), items, runner.read_runs("test__random", Released.name))
    assert rep["static_brier"]["img.basic/toy"][model] == pytest.approx(expected["brier"])
    assert rep["static_ece"]["img.basic/toy"][model] == pytest.approx(expected["ece"])
    assert rep["overall"][model]["static_brier"] == pytest.approx(expected["brier"])
    assert "## Static Brier score per leaf" in sweep.report_markdown(rep)
