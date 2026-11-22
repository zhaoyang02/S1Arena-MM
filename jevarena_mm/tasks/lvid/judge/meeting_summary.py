"""lvid.judge/meeting_summary: Is this summary sentence faithful to the meeting?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class TofuEvalMeetingSummary(FrozenTask):
    name = 'lvid.judge/meeting_summary/tofueval_meetingbank'
    leaf = 'lvid.judge/meeting_summary'
    status = 'released'
    source = 'TofuEval, MeetingBank topic summaries (Tang et al., 2024; Hu et al., 2023)'
    url = 'https://github.com/amazon-science/tofueval'
    license = 'MeetingBank: CC BY-NC-ND 4.0 (the strictest of its stated licenses); TofuEval: MIT-0'
    tier = 'B'
    criteria_source = "Adapted from TofuEval's sentence-level factual consistency labels"
    pipeline = PIPELINE
    notes = (
        "50 meetings, 500 summary sentences (250 faithful, 250 not); the agenda segment's audio and video are "
        'the input.'
    )
