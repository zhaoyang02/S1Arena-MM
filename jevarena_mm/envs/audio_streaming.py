"""Discover streaming cards without importing optional model runtimes."""

from ..registry import register_env
from ..streaming.catalog import SOURCE, TASKS
from .base import Environment


class AudioStreamingEnvironment(Environment):
    modalities = frozenset({"audio"})
    source = "Full-Duplex-Bench (adapted audio streaming diagnostic)"
    url = SOURCE
    tier = "B"
    mode = "realtime"
    episodes = 20
    max_decisions = 0
    score_name = "timing_diagnostics"
    recorded_only = True
    streaming_only = True
    criteria_source = "Held-out source event windows and measured virtual playback"
    notes = (
        "Persistent audio I/O via stream-run and stream-score. Reports timing, coverage and "
        "censoring separately; no scalar success or official FDB score. See docs/audio_streaming.md."
    )

    @classmethod
    def play(cls, model, seeds, workers=1):
        print(
            f"{cls.name}: requires persistent audio I/O; use stream-run --env {cls.name} (docs/audio_streaming.md)"
        )
        return 0

    def reset(self, seed):
        raise NotImplementedError(
            "Use stream-run; a DecisionRequest cannot represent concurrent audio I/O"
        )

    def card(self):
        return {
            **super().card(),
            "status": "released",
            "streaming_only": True,
            "recorded_only": True,
        }


for _category, _name in TASKS.items():
    register_env(
        type(
            "AudioStream" + "".join(p.title() for p in _category.split("_")),
            (AudioStreamingEnvironment,),
            {
                "name": _name,
                "leaf": _name.rsplit("/", 1)[0],
                "license": "CANDOR Corpus License v2.0 (BetterUp); audio from betterup/candor-full-duplex-bench, "
                "not redistributed"
                if _category.startswith("candor_")
                else "MIT synthetic FDB v1.5 data",
            },
        )
    )
