"""vid.know/causal_qa: Cause of an object's motion."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class PerceptionTestCausal(FrozenTask):
    name = 'vid.know/causal_qa/perception_test'
    leaf = 'vid.know/causal_qa'
    status = 'released'
    source = 'Perception Test, multiple-choice video QA, valid split (Patraucean et al., 2023)'
    url = 'https://github.com/google-deepmind/perception_test'
    license = 'CC BY 4.0'
    tier = 'A'
    criteria_source = 'Original Perception Test questions and options'
    pipeline = PIPELINE
    notes = (
        '500 videos, one question each, from the explanatory (cause-of-motion) questions; 3 options.'
    )
