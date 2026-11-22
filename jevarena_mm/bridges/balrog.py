"""Play one BALROG episode with an S1Arena-MM model (run with BALROG's own Python).

    python -m jevarena_mm.bridges.balrog --upstream BALROG_DIR --env babyai --task BabyAI-MixedTrainLocal-v0/goto \
        --seed 42 --episode-idx 0 --endpoint http://127.0.0.1:PORT/ --out DIR

BALROG's own Evaluator.run_episode runs the environment, the naive agent's history (16 text turns, 1 image) and the
progression score; this bridge supplies the client: each step sends the agent's messages with the image in place and
every action of the environment's language action space as an option, and returns the chosen action unchanged (the
upstream answer sanitizer, which strips digits from NetHack actions, is bypassed). Seeding is that of the recorded
episodes: Python/NumPy and the environment are seeded from --seed, BabyAI's task-kind draw uses one seeded RNG,
BabaIsAI's native RNG and Crafter's constructor take the seed. An action outside the action space is a technical
failure, never a default action.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import random
import time
import traceback
from pathlib import Path

from .common import Unsupported, decide, outcome, record_to

QUESTION = "Which of the available actions should you take next to achieve the task?"
BINARY = ("babyai", "babaisai", "minihack")  # progression is 0 or 1: success is defined


def configuration(upstream: Path, env: str, seed: int, label: str):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(upstream / "balrog/config/config.yaml")
    cfg.envs.names = env
    cfg.envs.env_kwargs.seed = seed
    cfg.agent.type = "naive"
    cfg.agent.max_image_history = 0 if env == "textworld" else 1
    cfg.agent.max_text_history = 16
    cfg.agent.remember_cot = False
    cfg.agent.max_cot_history = 0
    cfg.eval.save_images = True
    cfg.eval.save_trajectories = True
    cfg.eval.num_workers = 1
    cfg.client.client_name = "jevarena_mm"
    cfg.client.model_id = label
    cfg.client.generate_kwargs = {}
    cfg.client.max_retries = 0
    if env == "crafter":
        cfg.envs.crafter_kwargs.seed = seed
    return cfg


@contextlib.contextmanager
def babyai_constructor_seed(seed: int):
    """BabyAI's mixed level draws its task kind before __init__ ends: give that rejection sampler one seeded RNG."""
    import numpy as np
    from minigrid.envs.babyai.mixed_seq_levels import Level_MixedTrainLocal

    original = Level_MixedTrainLocal.__init__
    rng = np.random.default_rng(seed)

    def seeded_init(self, *args, **kwargs):
        self._np_random = rng
        original(self, *args, **kwargs)

    Level_MixedTrainLocal.__init__ = seeded_init
    try:
        yield
    finally:
        Level_MixedTrainLocal.__init__ = original


def make_seeded(env_name: str, task: str, cfg):
    import numpy as np
    from balrog.environments import make_env

    seed = int(cfg.envs.env_kwargs.seed)
    random.seed(seed)
    np.random.seed(seed)
    if env_name == "babyai":
        with babyai_constructor_seed(seed):
            return make_env("babyai", task, cfg)
    if env_name == "crafter":
        cfg.envs.crafter_kwargs.seed = seed
    env = make_env(env_name, task, cfg)
    if env_name == "babaisai":
        env.gym_env.unwrapped._np_random = np.random.default_rng(seed)
    return env


class Tracked:
    """The native environment, refusing actions outside its language action space and counting steps."""

    def __init__(self, env, client):
        self.env, self.steps, self.last = env, 0, None
        self.actions = list(env.language_action_space)
        client.actions = self.actions

    def __getattr__(self, key):
        return getattr(self.env, key)

    def check_action_validity(self, action):
        if action not in self.actions:
            raise ValueError("classified action outside the action space; no default action is executed")
        return self.env.check_action_validity(action)

    def step(self, action):
        self.check_action_validity(action)
        result = self.env.step(action)
        self.steps += 1
        self.last = result
        return result


class Client:
    """A BALROG client that classifies over the action list instead of generating text."""

    def __init__(self, endpoint: str, out: Path, label: str):
        self.endpoint, self.out, self.label = endpoint, out, label
        self.actions: list[str] = []
        self.calls = 0

    def generate(self, messages):
        from balrog.client import LLMResponse

        state, images = [], []
        for message in messages:
            if message.role == "assistant" and message.content not in self.actions:
                raise ValueError("non-action assistant history is outside the naive protocol")
            state.append(f"{message.role.upper()}:\n{message.content}")
            if message.attachment is not None:
                buffer = io.BytesIO()
                message.attachment.save(buffer, format="PNG")
                data = buffer.getvalue()
                path = self.out / "inputs" / f"{hashlib.sha256(data).hexdigest()}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
                images.append(str(path))
                state.append(f"<image:{len(images)}>")
        options = [{"id": f"a{i}", "text": action} for i, action in enumerate(self.actions)]
        result = decide(self.endpoint, "\n\n".join(state), QUESTION, options, images, {"suite": "balrog", "step": self.calls})
        self.calls += 1
        chosen = next(o["text"] for o in options if o["id"] == result["selected"])
        return LLMResponse(model_id=self.label, completion=chosen, stop_reason="classification_argmax",
                           input_tokens=0, output_tokens=0, reasoning=None)


def run(args) -> dict:
    upstream, out = Path(args.upstream).resolve(), Path(args.out).resolve()
    record_to(out)
    from balrog import evaluator as native
    from balrog.agents.naive import NaiveAgent
    from balrog.prompt_builder import create_prompt_builder

    cfg = configuration(upstream, args.env, args.seed, args.label)
    client = Client(args.endpoint, out, args.label)
    tracked: list[Tracked] = []

    def make_env(env_name, task, config):
        env = Tracked(make_seeded(env_name, task, config), client)
        tracked.append(env)
        return env

    class ClassifiedNaiveAgent(NaiveAgent):
        def _extract_final_answer(self, answer):
            return answer  # already one exact action; the upstream sanitizer would corrupt actions such as "86"

    native.make_env = make_env
    evaluator = native.Evaluator(args.env, cfg, original_cwd=str(upstream), output_dir=str(out / "native"))
    agent = ClassifiedNaiveAgent(lambda: client, create_prompt_builder(cfg.agent))
    started = time.time()
    try:
        result = evaluator.run_episode(args.task, agent, episode_idx=args.episode_idx)
    except Unsupported as error:
        return outcome(out, "unsupported", error=str(error), decisions=client.calls)
    progression = float(result.get("progression") or 0.0)
    return outcome(out, "completed", success=(progression == 1) if args.env in BINARY else None, score=progression,
                   decisions=tracked[0].steps if tracked else client.calls, failed_decisions=0,
                   wall_s=round(time.time() - started, 1),
                   details={"task": args.task, "seed": args.seed, "episode_return": result.get("episode_return"),
                            "max_steps": int(getattr(tracked[0], "max_steps", 0)) if tracked else None})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="jevarena-mm")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        outcome(args.out, "technical_failure", error=f"{type(error).__name__}: {error}"[:500], traceback=traceback.format_exc()[-4000:])


if __name__ == "__main__":
    main()
