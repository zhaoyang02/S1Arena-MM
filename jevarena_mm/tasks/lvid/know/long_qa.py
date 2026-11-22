"""lvid.know/long_qa: Long-video multiple choice."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class VideoMMELong(FrozenTask):
    name = 'lvid.know/long_qa/video_mme'
    leaf = 'lvid.know/long_qa'
    status = 'released'
    source = 'Video-MME, long videos (Fu et al., 2025)'
    url = 'https://video-mme.github.io/'
    license = 'Video-MME terms: academic research only, no distribution without the authors\' approval'
    tier = 'A'
    criteria_source = 'Original Video-MME questions and options'
    redistributable = False
    pipeline = PIPELINE
    notes = (
        '299 long videos, 500 questions; video only, without subtitles or audio. Video-MME may not be redistributed, '
        'so the questions, options and answers ship as a recipe and `jevarena-mm build` fills them in from Video-MME\'s '
        'own release (fetched with the videos).'
    )
