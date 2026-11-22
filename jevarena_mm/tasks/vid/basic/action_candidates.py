"""vid.basic/action_candidates: Which of a few candidate actions is shown."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask


@register_task
class KineticsActionCandidates(FrozenTask):
    name = "vid.basic/action_candidates/kinetics700"
    leaf = "vid.basic/action_candidates"
    status = 'released'
    source = "Kinetics-700-2020, validation split (Smaira et al., 2020)"
    url = "https://github.com/cvdfoundation/kinetics-dataset"
    license = "CC BY 4.0 (annotations); YouTube videos keep their own terms"
    tier = "B"
    criteria_source = "Adapted: the gold class and three other Kinetics-700 class names are the options"
    pipeline = "s1-arena standalone benchmark stores"
    notes = "500 ten-second clips of everyday human actions; 4 candidate actions per question."
