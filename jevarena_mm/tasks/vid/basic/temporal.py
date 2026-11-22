"""vid.basic/temporal: Direction of a fine-grained action."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class SSv2PushDirection(FrozenTask):
    name = 'vid.basic/temporal/ssv2'
    leaf = 'vid.basic/temporal'
    status = 'released'
    source = 'Something-Something v2, validation split (Goyal et al., 2017)'
    url = 'https://www.qualcomm.com/developer/software/something-something-v-2-dataset'
    license = 'Something-Something v2 license (research use)'
    tier = 'C'
    criteria_source = "Written; gold is the direction in the SSv2 template label ('Pushing [something] from left to right')"
    redistributable = False
    pipeline = PIPELINE
    notes = (
        '250 pushing videos, each asked twice: a two-option direction choice and a yes/no question about one '
        'direction (125 yes, 125 no). The scenes, questions, options and answers depend on the video\'s label, so they '
        'ship as a recipe; `jevarena-mm build` fills them in from the validation labels of your Something-Something v2 '
        'download (fetched with the videos).'
    )
