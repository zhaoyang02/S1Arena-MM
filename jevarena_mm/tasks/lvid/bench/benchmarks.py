"""lvid.bench: public long-video benchmarks, original questions."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask


@register_task
class EgoSchema(FrozenTask):
    name = "lvid.bench/egoschema/public_500"
    leaf = "lvid.bench/egoschema"
    status = 'released'
    source = "EgoSchema, public answer subset (Mangalam et al., 2023)"
    url = "https://github.com/egoschema/EgoSchema"
    license = "Questions: published by the EgoSchema authors, no license stated; clips cut from Ego4D (Ego4D license)"
    tier = "A"
    criteria_source = "Original EgoSchema questions and options"
    pipeline = "s1-arena standalone benchmark stores"
    notes = ("The 500 questions with public answers, on three-minute Ego4D clips; 5 options. The questions are as "
             "EgoSchema publishes them; the clips come from EgoSchema's release (scripts/fetch/egoschema.py). Closest "
             "leaf: lvid.know/long_qa.")


@register_task
class VSIBench(FrozenTask):
    name = "lvid.bench/vsi_bench/test"
    leaf = "lvid.bench/vsi_bench"
    status = 'released'
    source = "VSI-Bench (Yang et al., 2025)"
    url = "https://huggingface.co/datasets/nyu-visionx/VSI-Bench"
    license = "Apache-2.0 (annotations); videos under the ARKitScenes, ScanNet and ScanNet++ terms"
    tier = "A"
    criteria_source = "Original VSI-Bench questions and options"
    pipeline = "s1-arena release v0.1"
    notes = ("243 indoor walkthrough videos, 500 spatial and route-planning questions with 2-4 options. Closest leaf: "
             "lvid.embodied/ego_plan.")
