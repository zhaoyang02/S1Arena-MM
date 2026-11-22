"""vid.embodied/robot_success: Did the robot complete the task?"""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class RoboFACSuccess(FrozenTask):
    name = 'vid.embodied/robot_success/robofac'
    leaf = 'vid.embodied/robot_success'
    status = 'released'
    source = 'RoboFAC robotic failure analysis dataset (2025)'
    url = 'https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset'
    license = 'MIT'
    tier = 'B'
    criteria_source = "Adapted from RoboFAC's trajectory success labels; the task text is RoboFAC's"
    pipeline = PIPELINE
    notes = (
        "500 trajectories across RoboFAC's environments (250 successes, 250 failures); the window is the whole "
        'trajectory, so the outcome is visible at its end.'
    )
