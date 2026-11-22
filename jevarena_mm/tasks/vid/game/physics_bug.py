"""vid.game/physics_bug: Which event triggered the game glitch."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class PhysGameGlitch(FrozenTask):
    name = 'vid.game/physics_bug/physgame'
    leaf = 'vid.game/physics_bug'
    status = 'released'
    source = 'PhysGame (Cao et al., 2024)'
    url = 'https://huggingface.co/datasets/PhysGame/GameBench'
    license = 'Apache-2.0 (dataset card); gameplay clips from Reddit and YouTube, rights not established'
    tier = 'A'
    criteria_source = 'Original PhysGame questions and options'
    pipeline = PIPELINE
    notes = (
        '500 gameplay videos, one question each; 4 options.'
    )
