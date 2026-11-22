"""lvid.know/lecture: Lecture content multiple choice."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class VideoMMMULecture(FrozenTask):
    name = 'lvid.know/lecture/videommmu'
    leaf = 'lvid.know/lecture'
    status = 'released'
    source = 'VideoMMMU (Hu et al., 2025)'
    url = 'https://github.com/EvolvingLMMs-Lab/VideoMMMU'
    license = 'CC BY-NC-SA 4.0 (dataset); videos remain their creators\''
    tier = 'A'
    criteria_source = 'Original VideoMMMU questions and options'
    pipeline = PIPELINE
    version = 2
    notes = (
        '255 lecture videos, 500 questions, most with 10 options. The input is the video with its complete audio track '
        '(median 458 s, longest 2,028 s); models that do not take audio are unsupported on it. Known limit of the '
        'reference run: Jev-Omni with the full audio scores 26.4 % (61 % on items with under 3 minutes of audio, but '
        '11-12 % beyond 7.5 minutes, where its answers collapse onto the seventh option); the 2 items with 2,028 s of '
        'audio exceed a 48 GB card and are recorded as rejected.'
    )
