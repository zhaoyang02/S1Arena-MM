"""Silero VAD consumes only arrived PCM, with no dataset boundaries."""

import numpy as np


class OnlineVAD:
    def __init__(self):
        from silero_vad import load_silero_vad, VADIterator

        self.vad = VADIterator(
            load_silero_vad(onnx=True),
            threshold=0.5,
            sampling_rate=16000,
            min_silence_duration_ms=500,
            speech_pad_ms=300,
        )
        self.pending = bytearray()
        self.samples = 0

    def feed(self, pcm):
        import torch

        self.pending.extend(pcm)
        events = []
        while len(self.pending) >= 1024:
            frame = bytes(self.pending[:1024])
            del self.pending[:1024]
            self.samples += 512
            event = self.vad(
                torch.from_numpy(
                    np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768
                ),
                return_seconds=False,
            )
            if event:
                events.append({**event, "observed_samples": self.samples})
        return events
