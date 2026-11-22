"""vid.fore/pedestrian: Will the marked pedestrian start crossing within 2 s?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class PIECrossing(FrozenTask):
    name = 'vid.fore/pedestrian/pie'
    leaf = 'vid.fore/pedestrian'
    status = 'released'
    source = 'PIE, pedestrian crossing intention (Rasouli et al., 2019)'
    url = 'https://github.com/aras62/PIE'
    license = 'MIT (videos and annotations)'
    tier = 'B'
    criteria_source = "Adapted from PIE's crossing annotations: will the target start crossing within 2 s of the cutoff"
    pipeline = PIPELINE
    notes = (
        '49 videos, 325 cutoffs (150 yes, 175 no). The video runs from its start to the cutoff (median 302 s); the '
        'target pedestrian is outlined in red (marks) wherever PIE annotates it. Known limit: hard for OneJev-27B, and not '
        'because of the long window: on 100 items (majority 54 %) it scored 56 / 59 / 55 % with the full window, the last '
        '10 s and the last 5 s (no difference significant).'
        " Leaf score: the mean of its sources, so these 325 items weigh as much as JAAD's 175."
    )


@register_task
class JAADCrossing(FrozenTask):
    name = 'vid.fore/pedestrian/jaad'
    leaf = 'vid.fore/pedestrian'
    status = 'released'
    source = 'JAAD, pedestrian crossing behaviour (Rasouli et al., 2017)'
    url = 'https://github.com/ykotseruba/JAAD'
    license = 'CC BY 4.0 (videos); MIT (annotations)'
    tier = 'B'
    criteria_source = "Adapted from JAAD's crossing annotations: will the target start crossing within 2 s of the cutoff"
    pipeline = PIPELINE
    notes = (
        '130 videos, 175 cutoffs (100 yes, 75 no). The video runs from its start to the cutoff; the target '
        'pedestrian is outlined in red (marks) wherever JAAD annotates it.'
        " Leaf score: the mean of its sources, so these 175 items weigh as much as PIE's 325."
    )
