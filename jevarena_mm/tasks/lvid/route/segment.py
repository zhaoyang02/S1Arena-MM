"""lvid.route/segment: Highlight quality of a 2 s clip for a query."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class QVHighlightsSaliency(FrozenTask):
    name = 'lvid.route/segment/qvhighlights'
    leaf = 'lvid.route/segment'
    status = 'released'
    source = 'QVHighlights saliency scores, validation split (Lei et al., 2021)'
    url = 'https://github.com/jayleicn/moment_detr'
    license = 'CC BY-NC-SA 4.0 (annotations); YouTube videos, no license stated'
    tier = 'B'
    criteria_source = 'Original five-level saliency scale (0 Very Bad to 4 Very Good)'
    pipeline = PIPELINE
    notes = (
        "482 videos, 500 two-second clips; gold is the distribution of the three annotators' ratings."
    )
