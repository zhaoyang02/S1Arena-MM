"""Dequantize a GPTQ checkpoint (4-bit, symmetric, GPTQ v1 storage, any act-order g_idx) into plain 16-bit linear weights.

Each quantized linear becomes `weight[j, i] = (q[i, j] - 2^(bits - 1)) * scale[g_idx[i], j]` (symmetric: the zero point
vLLM's kernels use, never reading qzeros), rounded once to the dtype the GPTQ kernel dequantizes to (the model's
activation dtype), which is what the kernel multiplies with: a weight-only (W4A16) model runs the same from these
weights. Everything else (embeddings, norms, encoders) is copied.

    python scripts/gptq_dequant.py <snapshot dir> <output dir>

vLLM >= 0.30 refuses GPTQ checkpoints quantized with desc_act=True (e.g. Qwen/Qwen2.5-Omni-7B-GPTQ-Int4); such a model
runs as lm:<its Hugging Face id>@dequant=<output dir>, which keeps the checkpoint's id (with -dequant) for its results.
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PARTS = ("qweight", "qzeros", "scales", "g_idx")


def unpack(packed: torch.Tensor, bits: int, along_rows: bool) -> torch.Tensor:
    """int32 words of 32/bits values each -> the values, unpacked along rows ([in/8, out]) or columns ([g, out/8])."""
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    mask = (1 << bits) - 1
    if along_rows:
        values = (packed.unsqueeze(1) >> shifts.view(1, -1, 1)) & mask
        return values.reshape(-1, packed.shape[1])
    values = (packed.unsqueeze(-1) >> shifts.view(1, 1, -1)) & mask
    return values.reshape(packed.shape[0], -1)


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    quant = config.pop("quantization_config")
    bits, group = quant["bits"], quant["group_size"]
    if bits != 4 or quant.get("checkpoint_format", "gptq") != "gptq" or not quant.get("sym", True):
        raise SystemExit(f"unsupported quantization: {quant}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config.get("torch_dtype", "float16")]
    print(f"dequantizing to {dtype} (group {group}, desc_act {quant.get('desc_act')})", flush=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    weight_map, done = {}, 0
    for name in files:
        out = {}
        with safe_open(src / name, "pt") as f:
            keys = list(f.keys())
            groups = defaultdict(dict)
            for key in keys:
                prefix, _, part = key.rpartition(".")
                if part in PARTS:
                    groups[prefix][part] = f.get_tensor(key)
                else:
                    out[key] = f.get_tensor(key)
            for prefix, parts in groups.items():
                q = unpack(parts["qweight"], bits, along_rows=True)              # [in, out]
                scales = parts["scales"].to(dtype)                               # [groups, out]
                g_idx = parts.get("g_idx")
                if g_idx is None:
                    g_idx = torch.arange(q.shape[0], dtype=torch.int32) // group
                g_idx = g_idx.long()
                # Symmetric: the zero point is 2^(bits-1), as vLLM's kernels take it (they never read qzeros then).
                centered = (q - (1 << (bits - 1))).to(dtype)                     # exact: small integers
                weight = (centered.float() * scales[g_idx].float()).to(dtype)    # one rounding, as the kernel's product
                out[f"{prefix}.weight"] = weight.t().contiguous()
                done += 1
        save_file(out, dst / name, metadata={"format": "pt"})
        weight_map.update({key: name for key in out})
        print(f"{name}: {len(out)} tensors", flush=True)

    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    skip = ("config.json", "model.safetensors.index.json", "quantize_config.json", "quant_log.csv")  # vLLM reads quantize_config.json
    for path in src.iterdir():  # tokenizer, processor, chat template, generation config
        if path.suffix != ".safetensors" and path.name not in skip:
            shutil.copy(path.resolve(), dst / path.name)
    print(f"{done} quantized linears dequantized into {dst}", flush=True)


if __name__ == "__main__":
    main()
