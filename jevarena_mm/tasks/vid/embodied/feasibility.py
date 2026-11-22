"""vid.embodied/feasibility: What will the person do next / what is feasible."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class STARPredictionFeasibility(FrozenTask):
    name = 'vid.embodied/feasibility/star'
    leaf = 'vid.embodied/feasibility'
    status = 'released'
    source = 'STAR, prediction and feasibility questions, validation split (Wu et al., 2021)'
    url = 'https://bobbywu.com/STAR/'
    license = 'Charades license (non-commercial research)'
    tier = 'A'
    criteria_source = 'Original STAR questions and options'
    pipeline = PIPELINE
    notes = (
        "362 Charades videos, 500 questions; for prediction questions the video ends at the question's "
        'cutoff.'
    )
