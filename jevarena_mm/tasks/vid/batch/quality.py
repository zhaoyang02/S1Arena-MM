"""vid.batch/quality: Overall visual quality."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class KoNViDQuality(FrozenTask):
    name = 'vid.batch/quality/konvid_1k'
    leaf = 'vid.batch/quality'
    status = 'released'
    source = 'KoNViD-1k (Hosu et al., 2017)'
    url = 'https://database.mmsp-kn.de/konvid-1k-database.html'
    license = 'Free for research, no license stated; Flickr (YFCC100m) videos under their own Creative Commons terms'
    tier = 'B'
    criteria_source = 'Adapted from the ACR scale of the KoNViD-1k study (1 Bad to 5 Excellent)'
    pipeline = PIPELINE
    notes = (
        '500 videos. Gold is the level nearest to the mean opinion score, which is kept in '
        "meta['mean_score']."
    )
