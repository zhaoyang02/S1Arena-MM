"""vid.judge/t2v: How well a generated video matches its prompt."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class VideoFeedbackAlignment(FrozenTask):
    name = 'vid.judge/t2v/videofeedback'
    leaf = 'vid.judge/t2v'
    status = 'released'
    source = 'VideoFeedback, text-to-video alignment, test split (He et al., 2024)'
    url = 'https://huggingface.co/datasets/TIGER-Lab/VideoFeedback'
    license = 'Apache-2.0 (dataset card)'
    tier = 'B'
    criteria_source = "Adapted from VideoFeedback's four-level text-to-video alignment scale"
    pipeline = PIPELINE
    notes = (
        '500 generated videos with their generation prompt quoted verbatim; gold is the annotated alignment '
        "level. The ratings are VideoFeedback's human labels on its test split and the items a seeded random draw, so "
        "they keep the dataset's distribution: 24 / 33 / 43 / 0.6 % rated 1 / 2 / 3 / 4 (the test split holds only 15 "
        "'Perfect' ratings)."
    )
