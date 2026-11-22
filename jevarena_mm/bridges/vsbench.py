"""Play one VS-Bench episode with an S1Arena-MM model (run with VS-Bench's own Python).

    python -m jevarena_mm.bridges.vsbench --upstream VSBENCH_DIR --env kuhn_poker --episode 0 \
        --endpoint http://127.0.0.1:PORT/ --out DIR [--rng legacy|isolated]

VS-Bench's own run_episode runs the environment, its fixed opponents and the returns; every role the official config
gives to a prompt agent is played by the model. Each decision sends the official system prompt, the role's images and
observation text, with all legal actions as options; nothing hidden (serialized state, opponent records) is sent.
--rng legacy keeps the official seeding (environment version 1, as recorded first); --rng isolated gives the
environment and each fixed opponent their own seeded Python/NumPy streams (version 2, the RNG fix).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import random
import sys
import textwrap
import time
import traceback
import types
from copy import deepcopy
from pathlib import Path

from .common import Unsupported, decide, outcome, record_to

QUESTION = "Which legal action should you take now? Select exactly one complete action."


def bootstrap(upstream: Path, environment: str) -> None:
    """Import only the modules this environment uses (the official package files eagerly import every backend)."""
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    for name in ("envs", "agents", "models"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(upstream / name)]
            sys.modules[name] = package
    importlib.import_module(f"envs.{environment}")
    for name in ("builtin_agent", "mcts_agent", "cfr_agent"):
        importlib.import_module(f"agents.{name}")


def visible(observation, environment: str) -> dict:
    """The official prompt pieces of one player's observation, and its legal actions."""
    from prompts.observation import get_observation_prompt
    from prompts.system import get_system_prompt

    prompt = get_observation_prompt(environment, observation)
    return {"system": get_system_prompt(environment), "observation": prompt.text,
            "images": [str(Path(p).resolve()) for p in prompt.image_paths],
            "candidates": [{"id": f"action_{i}", "native_id": int(k), "text": v} for i, (k, v) in enumerate(observation.legal_actions.items())]}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Stream:
    """A saved Python/NumPy RNG state that native code runs inside."""

    def __init__(self, seed: int):
        import numpy as np

        self.python = random.Random(seed).getstate()
        self.numpy = np.random.RandomState(seed).get_state()

    @contextlib.contextmanager
    def active(self):
        import numpy as np

        outer = random.getstate(), np.random.get_state()
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        try:
            yield
        finally:
            self.python, self.numpy = random.getstate(), np.random.get_state()
            random.setstate(outer[0])
            np.random.set_state(outer[1])


@contextlib.contextmanager
def isolated_rng(environment: str, seed: int):
    """Version 2: the environment and each fixed opponent draw from their own seeded streams, the model from none."""
    from agents.base_agent import BaseAgent
    from envs.base_env import BaseEnv

    cls = BaseEnv._registry[environment]
    originals = {name: getattr(cls, name) for name in ("__init__", "reset", "step")}
    holder = {"stream": Stream(seed)}

    def init(env, *args, **kwargs):
        with holder["stream"].active():
            originals["__init__"](env, *args, **kwargs)

    def reset(env, seed=0):
        holder["stream"] = Stream(seed)
        with holder["stream"].active():
            return originals["reset"](env, seed=seed)

    def step(env, *args, **kwargs):
        with holder["stream"].active():
            return originals["step"](env, *args, **kwargs)

    cls.__init__, cls.reset, cls.step = init, reset, step
    factory = BaseAgent.__dict__["from_config"]
    count = {"roles": 0}

    def from_config(agent_cls, spec):
        role = count["roles"]
        count["roles"] += 1
        if spec["type"] in ("prompt_agent", "jevarena_mm"):
            return factory.__func__(agent_cls, spec)
        opponent = Stream(int(digest([seed, role, spec["type"]])[:8], 16))
        with opponent.active():
            agent = factory.__func__(agent_cls, spec)
        original_act = agent._act
        tree = ast.parse(textwrap.dedent(inspect.getsource(original_act)))
        if any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom)) for n in ast.walk(tree)):
            raise ValueError("a fixed opponent now suspends inside _act; RNG isolation must be reviewed")

        async def act(observation):
            with opponent.active():
                return await original_act(observation)

        agent._act = act
        return agent

    BaseAgent.from_config = classmethod(from_config)
    try:
        yield
    finally:
        for name, method in originals.items():
            setattr(cls, name, method)
        BaseAgent.from_config = factory


def register_agent(environment: str, endpoint: str, counter: dict):
    from agents.base_agent import BaseAgent

    class JevArenaAgent(BaseAgent, agent_type="jevarena_mm"):
        def __init__(self, role):
            self.role, self.calls = role, 0

        async def _act(self, observation):
            if observation.agent_id != self.role:
                raise ValueError("observation of another player")
            seen = visible(observation, environment)
            state = seen["system"] + "\n\n" + "\n".join(f"<image:{i + 1}>" for i in range(len(seen["images"]))) + "\n\n" + seen["observation"]
            context = {"suite": "vsbench", "role": self.role, "decision": self.calls}
            result = await asyncio.to_thread(decide, endpoint, state, QUESTION, seen["candidates"], seen["images"], context)
            self.calls += 1
            counter["decisions"] += 1
            chosen = next(c for c in seen["candidates"] if c["id"] == result["selected"])
            return chosen["native_id"], {"action": chosen["native_id"], "action_string": chosen["text"],
                                         "probabilities": result["probabilities"], "token_info": None}

    return JevArenaAgent


def run(args) -> dict:
    upstream, out = Path(args.upstream).resolve(), Path(args.out).resolve()
    bootstrap(upstream, args.env)
    record_to(out)
    counter = {"decisions": 0}
    register_agent(args.env, args.endpoint, counter)
    from eval.decision_making import run_episode
    from utils.helper import load_decision_config
    from utils.logger import Logger, convert_numpy_types

    config = deepcopy(load_decision_config(args.env, None))  # also performs the official base seeding
    config["experiment"]["results_dir"] = str(out / "native")
    config["experiment"]["name"] = "jevarena-mm"
    roles = []
    for role, agent in enumerate(config["agents"]):
        if agent["type"] == "prompt_agent":
            config["agents"][role] = {"type": "jevarena_mm", "params": {"role": role}}
            roles.append(role)
    record: dict = {}

    class RecordingLogger(Logger):
        def episode_end(self, episode_id, env_info):
            super().episode_end(episode_id, env_info)
            record.update(returns=convert_numpy_types(self.returns[episode_id]), native_result=convert_numpy_types(env_info),
                          steps=int(self.steps[episode_id]))

    logger = RecordingLogger(config)
    started = time.time()
    scope = isolated_rng(args.env, config["experiment"].get("seed", 0) + args.episode) if args.rng == "isolated" else contextlib.nullcontext()
    try:
        with scope:
            asyncio.run(run_episode(args.episode, config, logger))
    except Unsupported as error:
        return outcome(out, "unsupported", error=str(error), decisions=counter["decisions"])
    finally:
        logger.close()
    if not record:
        raise RuntimeError("VS-Bench returned without a final result")
    returns = [float(r) for r in record["returns"]]
    winner = (record["native_result"] or {}).get("winner") if isinstance(record["native_result"], dict) else None
    return outcome(out, "completed", success=(winner in roles) if winner is not None else None,
                   score=sum(returns[r] for r in roles) / len(roles), decisions=counter["decisions"], failed_decisions=0,
                   wall_s=round(time.time() - started, 1),
                   details={"returns": returns, "winner": winner, "model_roles": roles, "steps": record["steps"], "rng": args.rng})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rng", choices=["legacy", "isolated"], default="legacy")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        outcome(args.out, "technical_failure", error=f"{type(error).__name__}: {error}"[:500], traceback=traceback.format_exc()[-4000:])


if __name__ == "__main__":
    main()
