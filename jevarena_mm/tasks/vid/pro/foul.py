"""vid.pro/foul: Foul class, offence and severity."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask

PIPELINE = "s1-arena release v0.1"


@register_task
class SoccerNetMVFoul(FrozenTask):
    name = 'vid.pro/foul/soccernet_mvfoul'
    leaf = 'vid.pro/foul'
    status = 'released'
    source = 'SoccerNet-MVFoul, test split (Held et al., 2023)'
    url = 'https://huggingface.co/datasets/SoccerNet/SN-MVFouls-2024'
    license = 'SoccerNet NDA for its data and annotations (SoccerNet/sn-mvfoul README); GPL-3.0 on the Hugging Face card'
    tier = 'B'
    criteria_source = 'Adapted from the SoccerNet-MVFoul labels: action class, offence and the original five-level severity'
    redistributable = False
    pipeline = PIPELINE
    notes = (
        '268 incidents, each shown from 2-4 unsynchronized camera views. 200 action-class choices (8 '
        'options), 200 offence yes/no questions and 100 severity ratings on the original five levels. The '
        'answers come from SoccerNet\'s annotations, which SoccerNet shares under its NDA, so they ship as a recipe; '
        '`jevarena-mm build` fills them in from annotations.json of SoccerNet\'s release (fetched with the clips).'
    )
