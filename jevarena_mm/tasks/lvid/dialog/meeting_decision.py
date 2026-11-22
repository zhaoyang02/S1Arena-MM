"""lvid.dialog/meeting_decision: Was this decision made in the meeting?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class AMIMeetingDecision(FrozenTask):
    name = 'lvid.dialog/meeting_decision/ami'
    leaf = 'lvid.dialog/meeting_decision'
    status = 'released'
    source = 'AMI Meeting Corpus, decision annotations (Carletta et al., 2005)'
    url = 'https://groups.inf.ed.ac.uk/ami/corpus/'
    license = 'CC BY 4.0'
    tier = 'B'
    criteria_source = 'Adapted from the AMI decision annotations'
    pipeline = PIPELINE
    notes = (
        '133 meetings, 500 decisions (250 made, 250 not, each asked as made or as not made); the full meeting '
        "audio and a close-up video are the input. 71 % of the meetings are from AMI's training split. Models that do "
        "not take audio (OneJev-27B) are unsupported."
    )
