"""Explicit shared-voice controller track, separate from native model speech."""
from functools import lru_cache
import json
from urllib.request import urlopen
from .jev_controller import Episode as JevEpisode, CONFIG

MODEL = 'audio-controller'


@lru_cache(maxsize=1)
def identity():
    with urlopen('http://127.0.0.1:8767/health', timeout=5) as response:
        value = json.load(response)
    if value.get('protocol') != 'audio-shared-voice-controller-v1' or not value.get('ready'):
        raise ValueError('Unexpected shared-voice controller identity')
    if not value.get('model_id') or not value.get('server_sha256'):
        raise ValueError('Controller identity missing model or implementation')
    return value


class Episode(JevEpisode):
    config = CONFIG
    decision_event = 'controller_decision'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = identity()['model_id'] + ' + external Qwen voice'
