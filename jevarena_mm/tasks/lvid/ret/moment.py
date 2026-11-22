"""lvid.ret/moment: Which quarter holds the queried moment."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class Ego4DNLQMoment(FrozenTask):
    name = 'lvid.ret/moment/ego4d_nlq'
    leaf = 'lvid.ret/moment'
    status = 'released'
    source = 'Ego4D natural language queries, validation split (Grauman et al., 2022)'
    url = 'https://ego4d-data.org/'
    license = 'Ego4D license agreement'
    tier = 'C'
    criteria_source = 'Written; gold is the quarter that holds all annotated response windows; the query is quoted verbatim'
    redistributable = False
    pipeline = PIPELINE
    version = 2
    notes = (
        '247 NLQ validation queries from 247 parent videos, 62 / 62 / 62 / 61 per quarter, the correct option equally '
        'often in each position; every annotation of the query lies in the answer quarter of the official clip, and the '
        'selection is spread over query templates and clip lengths. OneJev-27B scores 51.4 % and Jev-Omni 29.6 % (chance '
        '25 %). '
        'The queries and gold quarters ship as a recipe; `jevarena-mm build` fills them in from your copy of the '
        'Ego4D annotations (Ego4D license; fetched with the videos).'
        " Leaf score: the mean of its sources, so these 247 items weigh as much as Charades-STA's 114 and QVHighlights's 139."
    )


@register_task
class QVHighlightsMoment(FrozenTask):
    name = 'lvid.ret/moment/qvhighlights'
    leaf = 'lvid.ret/moment'
    status = 'released'
    source = 'QVHighlights, validation split (Lei et al., 2021)'
    url = 'https://github.com/jayleicn/moment_detr'
    license = 'CC BY-NC-SA 4.0 (annotations); YouTube videos, no license stated'
    tier = 'C'
    criteria_source = 'Written; gold is the quarter that holds the annotated moment; the query is quoted verbatim'
    pipeline = PIPELINE
    notes = (
        '137 videos, 139 queries with a single moment.'
        " Leaf score: the mean of its sources, so these 139 items weigh as much as Charades-STA's 114 and Ego4D-NLQ's 247."
    )


@register_task
class CharadesSTAMoment(FrozenTask):
    name = 'lvid.ret/moment/charades_sta'
    leaf = 'lvid.ret/moment'
    status = 'released'
    source = 'Charades-STA (Gao et al., 2017)'
    url = 'https://github.com/jiyanggao/TALL'
    license = 'Charades license (non-commercial research)'
    tier = 'C'
    criteria_source = 'Written; gold is the quarter that holds every interval of the query; the query is quoted verbatim'
    redistributable = False
    pipeline = PIPELINE
    version = 2
    notes = (
        '114 Charades-STA test events, one per video, 29 / 28 / 29 / 28 per quarter, the correct option equally often in '
        'each position; every annotated window of the event lies in the answer quarter of the whole video (decoded '
        'duration). OneJev-27B scores 43.0 % and Jev-Omni 37.7 % (chance 25 %). Media: the official Charades license '
        'allows local evaluation only, no redistribution. The queries, '
        'windows and gold quarters ship as a recipe; `jevarena-mm build` fills them in from the Charades-STA test '
        'annotations, fetched with the videos.'
        " Leaf score: the mean of its sources, so these 114 items weigh as much as Ego4D-NLQ's 247 and QVHighlights's 139."
    )
