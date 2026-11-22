"""vid.know/physics: Will A touch B in the unseen remainder?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class PhysionContact(FrozenTask):
    name = 'vid.know/physics/physion'
    leaf = 'vid.know/physics'
    status = 'released'
    source = 'Physion object contact prediction, test set (Bear et al., 2021)'
    url = 'https://github.com/cogtoolslab/physics-benchmarking-neurips2021'
    license = 'MIT (code and data)'
    tier = 'B'
    criteria_source = "Adapted from Physion's object contact prediction task (will the red agent touch the yellow target)"
    pipeline = PIPELINE
    notes = (
        "500 stimuli; the video ends before the outcome, as in Physion's readout protocol (254 yes, 246 no). Known "
        'limit: hard; OneJev-27B scores at chance (51 %).'
    )
