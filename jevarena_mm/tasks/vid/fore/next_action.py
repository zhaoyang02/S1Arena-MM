"""vid.fore/next_action: Which action starts next."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class EpicNextAction(FrozenTask):
    name = 'vid.fore/next_action/epic_kitchens_100'
    leaf = 'vid.fore/next_action'
    status = 'released'
    source = 'EPIC-KITCHENS-100, action anticipation, validation split (Damen et al., 2022)'
    url = 'https://epic-kitchens.github.io/'
    license = 'CC BY-NC 4.0'
    tier = 'B'
    criteria_source = 'Adapted from EPIC-KITCHENS-100 action anticipation; options are verb-noun actions observed in the validation set'
    pipeline = PIPELINE
    notes = (
        '135 videos, 500 cutoffs one second before the next annotated action; 4 options, all verb-noun '
        'actions observed in the validation set.'
    )
