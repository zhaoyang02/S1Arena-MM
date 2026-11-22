"""lvid.route/escalation: Should this question go to the stronger model?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class JevPairEscalation(FrozenTask):
    name = 'lvid.route/escalation/jev_pair'
    leaf = 'lvid.route/escalation'
    status = 'released'
    source = 'Escalation between Jev-Omni and OneJev-27B on other long-video leaves (built for this benchmark)'
    url = ''
    license = 'Inherits the licenses of the quoted source questions'
    tier = 'C'
    criteria_source = 'Written; gold = the stronger system B answers the original question correctly and A does not'
    redistributable = False
    pipeline = PIPELINE
    notes = (
        '500 questions from Video-MME, VSI-Bench, VideoMMMU, Ego4D-NLQ, UCF-Crime, QVHighlights and '
        "Charades-STA, each with system A's actual answer quoted (250 escalate, 250 not). A (Jev-Omni) and B "
        '(OneJev-27B) are frozen, fixed systems whose answers are recorded: the item asks whether, for this fixed pair, '
        'the question should go from A to B, so any model can be scored on it. The 70 items that quote Ego4D queries '
        "and the 113 that quote Video-MME ship as a recipe; `jevarena-mm build` fills them in from those sources' "
        "annotations (fetched with the videos; Ego4D needs its license)."
    )
