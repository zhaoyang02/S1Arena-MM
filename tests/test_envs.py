import json
import re

import numpy as np
import pytest

from jevarena_mm import media
from jevarena_mm.envs import runner as env_runner
from jevarena_mm.envs.base import Environment
from jevarena_mm.registry import make_model, register_env
from jevarena_mm.schema import DecisionRequest, Question

pytest.importorskip("PIL")


class Corridor(Environment):
    """Walk right to the end of a corridor; the observation is a picture of the position."""

    name = "loop.game/toy_pixels/corridor"
    leaf = "loop.game/toy_pixels"
    modalities = frozenset({"image"})
    episodes = 2
    max_decisions = 6

    def reset(self, seed: int) -> None:
        self.position, self.steps = 0, 0

    def observe(self) -> DecisionRequest:
        picture = np.zeros((4, 8, 3), dtype=np.uint8)
        picture[:, self.position] = 255
        return DecisionRequest(
            state={"frame": media.inline_image(picture)},
            questions={"move": Question(type="choice", instructions="Where to?", criteria={"left": "left", "right": "right"})},
        )

    def act(self, choice: dict[str, str]) -> None:
        self.position = max(0, min(7, self.position + (1 if choice["move"] == "right" else -1)))
        self.steps += 1

    def done(self) -> bool:
        return self.position == 7

    def outcome(self) -> dict:
        return {"success": self.position == 7, "score": float(self.position)}


def test_environment_with_image_observations(workspace, clean_registry):
    register_env(Corridor)
    played = env_runner.run_env(make_model("random"), Corridor, [0, 1], "paused", 0.0, workers=1)
    assert played == 2
    rows = env_runner.latest_episodes(make_model("random").slug, Corridor.name)
    assert len(rows) == 2 and all("outcome" in r and r["decisions"] >= 1 for r in rows)
    assert Corridor().card()["modalities"] == ["image"]


def test_environment_is_not_played_by_a_model_without_its_modality(workspace, clean_registry):
    register_env(Corridor)
    model = make_model("random@modalities=video")
    assert env_runner.run_env(model, Corridor, [0, 1], "paused", 0.0, workers=1) == 0
    assert env_runner.latest_episodes(model.slug, Corridor.name) == []


def test_recorded_suites_are_not_played_and_failures_are_not_losses(workspace, clean_registry):
    from jevarena_mm.envs.upstream import UpstreamEnvironment

    class Recorded(UpstreamEnvironment):
        name = "loop.game/toy_suite/level_1"
        leaf = "loop.game/toy_suite"
        upstream = "Toy @ 0000000"
        episodes = 3

    register_env(Recorded)
    model = make_model("random")
    assert env_runner.run_env(model, Recorded, [0, 1, 2], "paused", 0.0, workers=1) == 0
    rows = [
        {"seed": 0, "outcome": {"success": True, "score": 1.0}, "decisions": 4, "failed_decisions": 0},
        {"seed": 1, "error": "upstream blocked"},
        {"seed": 2, "unsupported": "free-text actions"},
    ]
    summary = env_runner.summarize(rows)
    assert summary["episodes"] == 1 and summary["crashed"] == 1 and summary["unsupported"] == 1
    assert summary["success_rate"] == 1.0


def test_loop_installer_pins_the_recorded_revisions():
    """scripts/loop/install.sh checks out each suite at the revision data/loop/suites.json names."""
    from jevarena_mm import env

    script = (env.ROOT / "scripts" / "loop" / "install.sh").read_text()
    pins = dict(re.findall(r"^([A-Z_]+)=([0-9a-f]{40})", script, re.M))
    suites = json.loads((env.DATA / "loop" / "suites.json").read_text())
    expected = {"gameworld": "GAMEWORLD", "balrog_pixels": "BALROG", "vsbench": "VSBENCH"}
    for suite, name in expected.items():
        short = suites[suite]["upstream"].rsplit("@ ", 1)[1]
        assert pins[name].startswith(short), (suite, pins[name], short)
    assert "GAMEWORLD_GAMES" in pins
