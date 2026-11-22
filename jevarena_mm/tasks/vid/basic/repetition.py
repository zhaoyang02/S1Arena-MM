"""vid.basic/repetition: Range of the repetition count."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class RepCountRepetitions(FrozenTask):
    name = 'vid.basic/repetition/repcount_a'
    leaf = 'vid.basic/repetition'
    status = 'released'
    source = 'RepCount-A, test and validation splits (Hu et al., 2022)'
    url = 'https://svip-lab.github.io/dataset/RepCount_dataset.html'
    license = 'No license stated; YouTube videos, rights not established'
    tier = 'C'
    criteria_source = 'Written; levels are fixed ranges of the annotated count'
    pipeline = PIPELINE
    notes = (
        '250 clips. Ten ordered levels: 0, 1, 2, 3, 4-5, 6-10, 11-15, 16-25, 26-50, 51 or more complete '
        'repetitions; gold is the level of the source count. Known limit: counting fast cycles needs dense frames.'
    )


@register_task
class CountixRepetitions(FrozenTask):
    name = 'vid.basic/repetition/countix'
    leaf = 'vid.basic/repetition'
    status = 'released'
    source = 'Countix, test split (Dwibedi et al., 2020)'
    url = 'https://sites.google.com/view/repnet'
    license = 'No license stated; Kinetics YouTube clips, video rights not established'
    tier = 'C'
    criteria_source = 'Written; levels are fixed ranges of the annotated count'
    pipeline = PIPELINE
    notes = (
        "250 clips cut to Countix's annotated repetition interval. Same ten levels as the RepCount-A task. Known limit: "
        'counting fast cycles needs dense frames; at 1 fps the reference models score below the majority level, while '
        "OneJev-27B reached 36-40 % at 4-8 fps."
    )
