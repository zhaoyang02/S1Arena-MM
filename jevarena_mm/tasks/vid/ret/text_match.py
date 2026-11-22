"""vid.ret/text_match: Which caption matches the video."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class MSRVTTMultipleChoice(FrozenTask):
    name = 'vid.ret/text_match/msrvtt_mc'
    leaf = 'vid.ret/text_match'
    status = 'released'
    source = 'MSR-VTT multiple-choice test (Xu et al., 2016; Yu et al., 2018)'
    url = 'https://github.com/yj-yu/lsmdc'
    license = 'No license stated; video rights not established'
    tier = 'B'
    criteria_source = 'Original five candidate captions; the question is written'
    pipeline = PIPELINE
    notes = (
        '500 test videos, one each; the options are the five captions of the JSFusion multiple-choice test.'
    )
