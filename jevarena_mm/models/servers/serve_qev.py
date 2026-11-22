"""Serve OneJev-27B (BF16) sharded over several GPUs with qev.

qev's MMDecisionEngine loads with device_map={"": device} (one GPU). On 3x A100-40GB the 54.7 GB BF16 checkpoint
needs sharding, and the FP8 path (torch._scaled_mm) needs sm_89+, so this wrapper patches from_pretrained to use
device_map="auto" with per-GPU memory caps, then runs the unmodified qev server. Everything else (prompt, media,
readout, calibration) is qev's own code.

    python -m jevarena_mm.models.servers.serve_qev --model <path> --port 8000 [--max-memory 0=20GiB,1=36GiB,2=36GiB]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-memory", default="0=20GiB,1=36GiB,2=36GiB")
    p.add_argument("--name", default="OneJev-27B")
    p.add_argument("--meta", default=None, help="write run_metadata.json here")
    args, rest = p.parse_known_args()

    import torch
    import transformers

    max_memory = {int(k): v for k, v in (kv.split("=") for kv in args.max_memory.split(","))}
    cls = transformers.Qwen3_5ForConditionalGeneration
    orig = cls.from_pretrained.__func__

    def patched(klass, *a, **kw):
        kw["device_map"] = "auto"
        kw["max_memory"] = max_memory
        kw.setdefault("low_cpu_mem_usage", True)
        t0 = time.time()
        m = orig(klass, *a, **kw)
        dm = getattr(m, "hf_device_map", {})
        devs = sorted({str(v) for v in dm.values()})
        print(f"[serve] loaded in {time.time()-t0:.0f}s over devices {devs}; "
              f"{sum(1 for v in dm.values() if str(v) == '0')}/{len(dm)} modules on cuda:0", flush=True)
        if args.meta:
            meta = {"server_command": " ".join(sys.argv), "model_name": args.name, "model_path": args.model,
                    "dtype": "bfloat16", "quantize": None, "device_map": {k: str(v) for k, v in dm.items()},
                    "max_memory": args.max_memory, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "calibration": "none shipped with the checkpoint (temperature 1.0)",
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            Path(args.meta).write_text(json.dumps(meta, indent=1))
        return m

    cls.from_pretrained = classmethod(patched)

    # the slot head rows live on the lm_head's device; hidden states come from the last decoder device
    from qev import mm_engine
    orig_readout = mm_engine.MMDecisionEngine._readout_rows

    def readout_rows(self, hidden, n_slots):
        return orig_readout(self, hidden.to(self._slot_w.device), n_slots)

    mm_engine.MMDecisionEngine._readout_rows = readout_rows

    from qev import cli
    sys.argv = ["qev", "serve", "--model", args.model, "--multimodal", "--no-cuda-graphs", "--port", str(args.port),
                "--name", args.name, "--host", "0.0.0.0", *rest]
    cli.main()


if __name__ == "__main__":
    main()
