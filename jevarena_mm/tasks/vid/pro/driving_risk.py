"""vid.pro/driving_risk: does the dashcam scene call for immediate action?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "built from the source's own labels"


@register_task
class NexarCollision(FrozenTask):
    name = 'vid.pro/driving_risk/nexar'
    leaf = 'vid.pro/driving_risk'
    status = 'released'
    source = 'Nexar Collision Prediction, test set (Moura and Zvitia, 2025)'
    url = 'https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction'
    license = 'Nexar Open Data License (attribution; no resale)'
    tier = 'B'
    criteria_source = "Adapted from Nexar's labels: a test clip of an event ends 0.5, 1.0 or 1.5 s before the collision or near-miss"
    pipeline = PIPELINE
    media_root = 'nexar'
    notes = (
        '500 dashcam clips of about 10 s (median 9.97 s in both classes), one per original video: 250 end 0.5, 1.0 or 1.5 s '
        'before a collision or near-miss (133, 69 and 48 clips; many videos are cut at 0.5 s only), 250 are normal drives '
        'cut at the same points (136, 68 and 46), so neither clip timing nor length gives the answer.'
    )
