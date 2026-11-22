"""vid.basic/state_change: Did an object change state?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class Ego4DObjectStateChange(FrozenTask):
    name = 'vid.basic/state_change/ego4d_oscc'
    leaf = 'vid.basic/state_change'
    status = 'released'
    source = 'Ego4D object state change classification (OSCC), validation split (Grauman et al., 2022)'
    url = 'https://ego4d-data.org/'
    license = 'Ego4D license agreement'
    tier = 'B'
    criteria_source = 'Adapted from the OSCC task: does an object undergo a state change in the clip'
    redistributable = False
    pipeline = PIPELINE
    notes = (
        '419 clips, one each; gold is the OSCC state-change label (201 yes, 218 no). The labels ship as a recipe; '
        '`jevarena-mm build` fills them in from your copy of the Ego4D annotations (Ego4D license; fetched with the '
        'videos).'
        " Leaf score: the mean of its sources, so these 419 items weigh as much as ChangeIt's 81."
    )


@register_task
class ChangeItStateDirection(FrozenTask):
    name = 'vid.basic/state_change/changeit'
    leaf = 'vid.basic/state_change'
    status = 'released'
    source = 'ChangeIt, test split (Soucek et al., 2022)'
    url = 'https://github.com/soCzech/ChangeIt'
    license = 'MIT (annotations); video rights not established'
    tier = 'C'
    criteria_source = "Written from ChangeIt's annotated initial and end states of each object category"
    pipeline = PIPELINE
    notes = (
        '81 web videos, one each; asks whether the main demonstrated change goes from the initial to the end '
        'state, with the direction swapped in about half of the items (43 yes, 38 no).'
        " Leaf score: the mean of its sources, so these 81 items weigh as much as Ego4D-OSCC's 419."
    )
