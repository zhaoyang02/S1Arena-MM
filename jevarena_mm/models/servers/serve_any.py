"""Serve one comparison model behind the qev System One API with the general adapter (models/servers/readout_engines.py).

Same endpoints and response JSON as the OneJev server, so the `qev_onejev` and other `qev_*` adapters (models/qev.py) call it unchanged.
Checkpoints with their own modeling code (MiniCPM-o, Phi-4-multimodal, Nemotron-3-Nano-Omni) go through readout_remote.py.
qwen3_5 / qwen3_5_moe checkpoints go through serve.py (qev's own engine) instead.

    python -m jevarena_mm.models.servers.serve_any --model <path> --port 8011 --name Qwen2.5-Omni-7B [--max-memory 0=36GiB,1=36GiB]
           [--meta run_metadata.json] [--batch-tokens 24000]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--name", required=True)
    p.add_argument("--max-memory", default=None)
    p.add_argument("--meta", default=None)
    p.add_argument("--batch-tokens", type=int, default=24000)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    import transformers
    import uvicorn
    from qev.server import create_app
    from jevarena_mm.models.servers.readout_engines import DiffusionReadoutEngine, HFReadoutEngine

    mm = {int(k): v for k, v in (kv.split("=") for kv in args.max_memory.split(","))} if args.max_memory else None
    from jevarena_mm.models.servers import readout_remote
    cfg = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cls = readout_remote.ENGINES.get(cfg.model_type) or (DiffusionReadoutEngine if cfg.model_type == "diffusion_gemma" else HFReadoutEngine)
    engine = cls(args.model, dtype=args.dtype, max_memory=mm, batch_tokens=args.batch_tokens)
    if args.meta:
        dm = getattr(engine.model, "hf_device_map", {}) or {}
        meta = {"server_command": " ".join(sys.argv), "model_name": args.name, "model_path": args.model,
                "dtype": args.dtype, "quantize": engine.quant_note, "backend": f"hf ({engine.info.extra['adapter']})",
                "assistant_prefix": engine.prefix_policy, "max_slots": engine.max_slots,
                "device_map": {k: str(v) for k, v in dm.items()}, "max_memory": args.max_memory,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "torch": torch.__version__, "transformers": transformers.__version__,
                "calibration": "none (temperature 1.0)", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta).write_text(json.dumps(meta, indent=1))
    app = create_app(engine, model_name=args.name)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
