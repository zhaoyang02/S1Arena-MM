"""vid.judge/hallucination: Is the claimed object, relation or event in the video?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class VideoHallucer(FrozenTask):
    name = 'vid.judge/hallucination/videohallucer'
    leaf = 'vid.judge/hallucination'
    status = 'released'
    source = 'VideoHallucer, object-relation and temporal subsets (Wang et al., 2024)'
    url = 'https://huggingface.co/datasets/bigai-nlco/VideoHallucer'
    license = 'MIT (dataset card), CC BY-NC-SA (its paper)'
    tier = 'A'
    criteria_source = 'Original VideoHallucer yes/no questions'
    pipeline = PIPELINE
    notes = (
        "250 videos, each with VideoHallucer's basic and hallucinated question (250 yes, 250 no); the two "
        "items of a video share meta['group']."
    )
