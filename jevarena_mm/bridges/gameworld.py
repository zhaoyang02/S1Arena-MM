"""Play one GameWorld task with an S1Arena-MM model (run with GameWorld's own Python).

    python -m jevarena_mm.bridges.gameworld --upstream GAMEWORLD_DIR --game 01_2048 --task 01_01 --seed 42 \
        --endpoint http://127.0.0.1:PORT/ --out DIR

GameWorld's own main() runs the browser game, its memory (the current screenshot and the previous ones) and its
evaluator; this bridge only replaces the model client. Each step sends the native system and user prompt with the
screenshots in place, and every complete semantic control (each enumerable parameter value expanded) as an option;
the service's choice is executed as that control. A game whose controls are not enumerable (Wordle's free word) is
unsupported. The protocol is the one of the recorded episodes (main, paused, 100 steps). With --latency-ms
(realtime) the game still pauses while the model decides, then runs for that long before the chosen action lands: each
decision costs the model's fixed latency in game time, whatever the inference took here. GameWorld runs
from its checkout and keeps its temporary screenshots there (.screenshots_temp/, ignored by its git and removed after
the episode); the logs and step screenshots go to --out.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import itertools
import json
import math
import os
import socket
import sys
import time
import traceback
import types
from pathlib import Path

from .common import Unsupported, decide, outcome, record_to

MODEL_NAME = "jevarena-mm"
QUESTION = "Which one of the registered actions should you execute next to achieve the task goal?"


class UnsupportedActionSpace(ValueError):
    pass


def candidates(specs, controls, max_options: int = 255) -> list[dict]:
    """Every complete native call of the role's semantic controls, with its normalized execution."""
    from agents.harness.semantic_controls import inspect_semantic_controls_output
    from env.action_executor import ActionExecutor

    mapping = {s["id"]: s["binding"] for s in specs}
    if len(mapping) != len(specs) or not mapping:
        raise UnsupportedActionSpace("missing or duplicate native semantic controls")
    executor = ActionExecutor(None, controls=controls)
    result = []
    for spec in specs:
        params = spec.get("parameters") or {}
        required = spec.get("required") or []
        if not isinstance(params, dict) or set(required) - set(params):
            raise UnsupportedActionSpace("native parameter schema is not an enumerable mapping")
        domains = []
        for key, schema in params.items():
            values = schema.get("enum") if isinstance(schema, dict) else None
            if values is None and key == "cell" and spec["binding"].get("cell_param"):
                values = list(spec["binding"].get("cell_bindings", {}))  # Minesweeper publishes its complete cell table
            if not isinstance(values, list) or not values:
                raise UnsupportedActionSpace(f"{spec['id']}.{key}: non-enumerable action parameter")
            domains.append([(key, value) for value in values] + ([] if key in required else [(key, None)]))
        if len(result) + math.prod(len(domain) for domain in domains) > max_options:
            raise UnsupportedActionSpace(f"the complete action space exceeds {max_options} options")
        for combination in itertools.product(*domains):
            call = {"tool_name": spec["id"], "arguments": {k: v for k, v in combination if v is not None}}
            mapped = inspect_semantic_controls_output(call, mapping)
            checked = executor.inspect_action(mapped.get("mapped_action"))
            if not mapped["is_valid"] or not checked["is_valid"]:
                raise UnsupportedActionSpace(f"incomplete native binding: {call}")
            binding = dict(checked["normalized_action"])
            if binding["action"] in ("press_key", "press_keys"):
                binding["duration"] = executor._resolve_key_hold_duration(checked["normalized_action"])
            result.append({"id": f"a{len(result):03}", "text": spec["description"] + " Execution: " + json.dumps(binding, sort_keys=True),
                           "native_call": call})
    return result


def adapter_module(endpoint: str, counter: dict) -> types.ModuleType:
    """A GameWorld agent adapter (Client, Config, PROMPT_TEMPLATE, OUTPUT_FORMAT) that asks the service."""
    from agents.mm_agents.base import BaseClientConfig, GeneralAgent

    @dataclasses.dataclass
    class Config(BaseClientConfig):
        pass

    class Client(GeneralAgent):
        def __init__(self, config, semantic_controls_specs=None):
            super().__init__(config, semantic_controls_specs)
            self.controls = None  # set from the parsed native role (see run())
            self.step = 0
            self._options: dict[str, dict] = {}

        def build_request_payload(self, *, system_prompt, user_prompt, memory_entries, tools, screenshot_path):
            if self.controls is None:
                raise ValueError("parsed native role controls were not provided")
            for entry in memory_entries:  # a missing history image must fail, never shrink the history
                if entry.type == "image" and (not entry.image_path or not Path(entry.image_path).is_file()):
                    raise FileNotFoundError("native image history is missing")
            content = self._build_user_content(memory_entries, append_user_text=lambda t: {"type": "text", "text": t},
                                               append_user_image=lambda p: {"type": "image", "path": str(p)},
                                               user_prompt=user_prompt, screenshot_path=screenshot_path)
            state, images = [system_prompt or ""], []
            for part in content:
                if part["type"] == "text":
                    state.append(part["text"])
                else:
                    images.append(part["path"])
                    state.append(f"<image:{len(images)}>")
            options = candidates(self._semantic_controls_specs, self.controls)
            self._options = {o["id"]: o for o in options}
            return {"state": "\n".join(state), "candidates": options, "images": images,
                    "context": {"suite": "gameworld", "agent": getattr(self, "agent_id", "agent_0"), "step": self.step}}

        def send_request(self, request_payload):
            result = decide(endpoint, request_payload["state"], QUESTION, request_payload["candidates"],
                            request_payload["images"], request_payload["context"])
            self.step += 1
            counter["decisions"] += 1
            return result

        def extract_tool_call(self, response):
            return self._options[response["selected"]]["native_call"]

    module = types.ModuleType("agents.mm_agents.jevarena_mm")
    module.Client, module.Config = Client, Config
    module.PROMPT_TEMPLATE, module.OUTPUT_FORMAT = "game_agent.j2", None
    return module


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run(args) -> dict:
    upstream = Path(args.upstream).resolve()
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    out = Path(args.out).resolve()
    record_to(out)
    counter = {"decisions": 0}
    from agents.mm_agents import loader

    sys.modules["agents.mm_agents.jevarena_mm"] = adapter_module(args.endpoint, counter)
    loader._model_index()[MODEL_NAME] = loader.ModelProfile(model_id=MODEL_NAME, adapter_id="jevarena_mm", source_path=Path(__file__),
                                                            enable_memory=True, config_overrides={"model": MODEL_NAME, "endpoint": args.endpoint})
    import main as native

    build_config, build_clients = native.build_runtime_config, native.build_agent_clients

    def build(preset):
        config = build_config(preset)
        config.random_seed = args.seed
        return config

    def clients(config, *rest, **kwargs):
        values = build_clients(config, *rest, **kwargs)
        for i, client in enumerate(values):
            client.controls = config.role_controls_maps[i]
            if len(values) > 1:
                client.agent_id = f"agent_{i}"
        return values

    native.build_runtime_config, native.build_agent_clients = build, clients
    if args.latency_ms > 0:  # realtime: the coordinator resumes the game before the action; it runs for the latency first
        from runtime.env import GameEnv

        execute = GameEnv.execute_action

        async def late_action(self, agent, action, **kwargs):
            await asyncio.sleep(args.latency_ms / 1000)
            await execute(self, agent, action, **kwargs)

        GameEnv.execute_action = late_action
    preset = f"{args.game}+{args.task}+{MODEL_NAME}"
    config = build(preset)
    try:
        options = [len(candidates(s, c)) for s, c in zip(config.semantic_controls_specs, config.role_controls_maps, strict=True)]
    except UnsupportedActionSpace as error:
        return outcome(out, "unsupported", error=str(error))
    started = time.time()
    try:
        asyncio.run(native.main(preset, headless=True, port=free_port(), log_root=str(out / "native"), model_endpoint=args.endpoint))
    except Unsupported as error:
        return outcome(out, "unsupported", error=str(error), decisions=counter["decisions"], wall_s=round(time.time() - started, 1))
    evaluation_path = out / "native" / "agent_0" / "evaluation" / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    if evaluation.get("task_status") not in ("success", "fail") or not evaluation.get("finalized"):
        raise RuntimeError("GameWorld returned without a finalized task outcome")
    return outcome(out, "completed", success=evaluation["task_status"] == "success", score=float(evaluation["progress"]),
                   decisions=counter["decisions"], failed_decisions=0, wall_s=round(time.time() - started, 1),
                   details={"task": args.task, "step": evaluation.get("step"), "max_steps": evaluation.get("max_steps"),
                            "stop_reason": evaluation.get("stop_reason"), "options_per_role": options,
                            "game_completion_progress": evaluation.get("game_completion_progress"),
                            **({"latency_ms": args.latency_ms} if args.latency_ms > 0 else {})})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--game", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--latency-ms", type=float, default=0.0, help="realtime: game time each decision costs (0 = paused)")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        outcome(args.out, "technical_failure", error=f"{type(error).__name__}: {error}"[:500], traceback=traceback.format_exc()[-4000:])


if __name__ == "__main__":
    main()
