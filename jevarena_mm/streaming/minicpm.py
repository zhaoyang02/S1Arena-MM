"""MiniCPM-o 4.5 native duplex preset; same virtual sink and episode deadlines."""
import base64
import os
from pathlib import Path

from .qwen import Episode as QwenEpisode
from .gemini import SYSTEM

MODEL = 'openbmb/MiniCPM-o-4_5'
CONFIG = {
    'model': MODEL, 'modalities': ['audio', 'text'], 'instructions': SYSTEM,
    'input_audio_format': 'pcm16', 'output_audio_format': 'pcm16', 'sample_rate_hz': 16000,
    'audio': {'input': {'sample_rate_hz': 16000}, 'output': {'sample_rate_hz': 24000}},
    'turn_detection': None, 'overlap_policy': 'listen_only', 'playback_commit_policy': 'ack_only',
    'extra_body': {'auto_response': True, 'force_listen_count': 0},
}


# Explicit bundled voice prompt: older installed runtimes do not resolve the default.
if os.environ.get('MINICPM_REF_AUDIO'):
    CONFIG['ref_audio'] = 'data:audio/wav;base64,' + base64.b64encode(
        Path(os.environ['MINICPM_REF_AUDIO']).read_bytes()
    ).decode()


class Episode(QwenEpisode):
    config = CONFIG
    model_name = MODEL
    implementation_level = 'model_native_duplex'

    def __init__(self, *args, **kwargs):
        if 'ref_audio' not in self.config:
            raise ValueError('Set MINICPM_REF_AUDIO to the pinned bundled reference WAV')
        super().__init__(*args, **kwargs)
