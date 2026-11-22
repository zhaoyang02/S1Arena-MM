"""lvid.embodied/ego_plan: Which action to take next toward a goal, from an egocentric task history."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask


@register_task
class EgoPlanBench2(FrozenTask):
    name = "lvid.embodied/ego_plan/egoplan_bench2"
    leaf = "lvid.embodied/ego_plan"
    status = 'released'
    source = "EgoPlan-Bench2 (Qiu et al., 2024)"
    url = "https://github.com/qiulu66/EgoPlan-Bench2"
    license = "Questions: published by the EgoPlan-Bench2 authors, no license stated; clips cut from Ego4D (Ego4D license)"
    tier = "A"
    criteria_source = "Original EgoPlan-Bench2 questions and options"
    pipeline = "s1-arena standalone benchmark stores"
    notes = ("500 next-step planning questions on Ego4D task histories; the video ends at the current observation; "
             "4 options. The questions are as EgoPlan-Bench2 publishes them; the clips come from its reference "
             "release (scripts/fetch/egoplan-bench2.py).")
