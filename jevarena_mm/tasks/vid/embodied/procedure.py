"""vid.embodied/procedure: Was an error made in this step?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class CaptainCookStepError(FrozenTask):
    name = 'vid.embodied/procedure/captaincook4d'
    leaf = 'vid.embodied/procedure'
    status = 'released'
    version = 2
    source = 'CaptainCook4D, step error annotations (Peddi et al., 2024)'
    url = 'https://github.com/CaptainCook4D/annotations'
    license = 'Apache-2.0 (data); MIT (annotation repository)'
    tier = 'B'
    criteria_source = "Adapted from CaptainCook4D's step-level error annotations"
    pipeline = PIPELINE
    notes = (
        '96 recordings, 500 steps (250 with an error, 250 without); the recipe step is quoted. Every item shows the '
        "complete recording with the step's annotated interval given in the text (recordings median 793 s, steps median "
        "34 s), so the surrounding steps are visible for order and omission errors. Gold is CaptainCook4D's error "
        "annotation; some error kinds (such as quantities) are hard to see."
    )
