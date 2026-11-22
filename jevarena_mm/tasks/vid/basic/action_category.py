"""vid.basic/action_category: Which action category, out of a complete category list."""

from jevarena_mm.registry import register_task
from jevarena_mm.tasks.frozen import FrozenTask


@register_task
class UCF101ActionCategory(FrozenTask):
    name = "vid.basic/action_category/ucf101"
    leaf = "vid.basic/action_category"
    status = 'released'
    source = "UCF101, test split 1 (Soomro et al., 2012)"
    url = "https://www.crcv.ucf.edu/data/UCF101.php"
    license = "No license stated (cite Soomro et al., 2012); YouTube videos"
    tier = "B"
    criteria_source = "Adapted: UCF101's 101 category names are the options"
    pipeline = "s1-arena standalone benchmark stores"
    notes = ("500 test videos; every question lists all 101 UCF101 categories (sports, body motion, playing instruments and "
             "interactions), so the model classifies over the dataset's complete label set.")
