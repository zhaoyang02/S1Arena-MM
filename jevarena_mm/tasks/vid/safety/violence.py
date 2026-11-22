"""vid.safety/violence: Does the video show event X?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class XDViolenceEvents(FrozenTask):
    name = 'vid.safety/violence/xd_violence'
    leaf = 'vid.safety/violence'
    status = 'released'
    source = 'XD-Violence, test split (Wu et al., 2020)'
    url = 'https://roc-ng.github.io/XD-Violence/'
    license = 'No license stated; video rights not established'
    tier = 'C'
    criteria_source = 'Written; gold is the video-level event label'
    pipeline = PIPELINE
    notes = (
        '338 videos, one named event each (185 yes, 153 no); audio is not part of the input.'
        " Leaf score: the mean of its sources, so these 338 items weigh as much as UCF-Crime's 162."
    )


@register_task
class UCFCrimeEvents(FrozenTask):
    name = 'vid.safety/violence/ucf_crime'
    leaf = 'vid.safety/violence'
    status = 'released'
    source = 'UCF-Crime, test split (Sultani et al., 2018)'
    url = 'https://github.com/WaqasSultani/AnomalyDetectionCVPR2018'
    license = 'No license stated; video rights not established'
    tier = 'C'
    criteria_source = 'Written; gold is the video-level event label'
    pipeline = PIPELINE
    notes = (
        '162 videos, one named event each (65 yes, 97 no).'
        " Leaf score: the mean of its sources, so these 162 items weigh as much as XD-Violence's 338."
    )
