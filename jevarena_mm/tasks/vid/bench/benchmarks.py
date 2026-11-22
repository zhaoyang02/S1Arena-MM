"""vid.bench: public short-video benchmarks, original questions."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena standalone benchmark stores"


@register_task
class TempCompass(FrozenTask):
    name = "vid.bench/tempcompass/mc"
    leaf = "vid.bench/tempcompass"
    status = 'released'
    source = "TempCompass, multiple-choice questions (Liu et al., 2024)"
    url = "https://github.com/llyx97/TempCompass"
    license = "CC BY-NC 4.0"
    tier = "A"
    criteria_source = "Original TempCompass questions and options"
    pipeline = PIPELINE
    notes = "500 questions on temporal aspects (action, speed, direction, order, attribute change); 2-4 options. Closest leaf: vid.basic/temporal."


@register_task
class NExTQA(FrozenTask):
    name = "vid.bench/nextqa/val"
    leaf = "vid.bench/nextqa"
    status = 'released'
    source = "NExT-QA, multiple choice, validation split (Xiao et al., 2021)"
    url = "https://github.com/doc-doc/NExT-QA"
    license = "MIT (annotations); VidOR videos from YFCC100m under their own Creative Commons terms"
    tier = "A"
    criteria_source = "Original NExT-QA questions and options"
    pipeline = PIPELINE
    notes = "500 causal, temporal and descriptive questions; 5 options. Closest leaf: vid.know/causal_qa."


@register_task
class MVBench(FrozenTask):
    name = "vid.bench/mvbench/test"
    leaf = "vid.bench/mvbench"
    status = 'released'
    source = "MVBench (Li et al., 2024)"
    url = "https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md"
    license = "MIT (dataset card); source videos keep their own licenses"
    tier = "A"
    criteria_source = "Original MVBench questions and options"
    pipeline = PIPELINE
    notes = "500 questions over MVBench's temporal subtasks; 2-5 options. Closest leaf: vid.basic/temporal."
