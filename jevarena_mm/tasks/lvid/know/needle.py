"""lvid.know/needle: Does a target appear, and in which quarter?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class NeedleShapes(FrozenTask):
    name = 'lvid.know/needle/generated'
    leaf = 'lvid.know/needle'
    status = 'released'
    source = 'Generated needle videos (built for this benchmark)'
    url = ''
    license = 'CC BY 4.0 (generated for this benchmark; no third-party assets)'
    tier = 'C'
    criteria_source = "Written; gold comes from the generator's placement log"
    pipeline = PIPELINE
    notes = (
        '250 generated videos of 25 coloured-shape targets inside light panels; each video is asked whether a '
        'target appears (125 yes, 125 no) and in which quarter it appears (5 options).'
    )
