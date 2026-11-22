"""lvid.pro/surveillance: Is there an anomaly, and when does it start?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class UCFCrimeSurveillance(FrozenTask):
    name = 'lvid.pro/surveillance/ucf_crime'
    leaf = 'lvid.pro/surveillance'
    status = 'released'
    source = 'UCF-Crime, untrimmed test videos (Sultani et al., 2018)'
    url = 'https://github.com/WaqasSultani/AnomalyDetectionCVPR2018'
    license = 'No license stated; video rights not established'
    tier = 'C'
    criteria_source = 'Written; gold comes from the temporal anomaly annotations of the test set'
    pipeline = PIPELINE
    notes = (
        '250 recordings, each asked whether it contains an anomaly (125 yes, 125 no) and in which quarter the '
        'first anomaly begins (5 options, including none).'
    )
