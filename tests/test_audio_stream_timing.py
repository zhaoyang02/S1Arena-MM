"""Temporal correctness checks for native streaming diagnostics (no API calls)."""

from pathlib import Path
import struct
import sys
import unittest

from jevarena_mm.streaming import core


class PlaybackTests(unittest.TestCase):
    def test_receive_does_not_mean_play(self):
        p = core.Playback()
        p.add(struct.pack("<h", 4000) * 24000)
        self.assertEqual(p.intervals, [])
        p.tick(0.020)
        self.assertEqual(p.played_frames, 480)
        self.assertAlmostEqual(p.buffered / 48000, 0.98)

    def test_interrupt_discards_unplayed_audio(self):
        p = core.Playback()
        p.add(struct.pack("<h", 4000) * 24000)
        p.tick(0.020)
        self.assertAlmostEqual(p.clear(), 0.98)
        self.assertEqual(p.tick(0.040), b"")
        self.assertEqual(p.played_frames, 480)

    def test_silence_is_not_speech(self):
        p = core.Playback()
        p.add(b"\0" * 960)
        p.tick(0.02)
        self.assertEqual(p.intervals, [])
        self.assertEqual(p.audible_frames, 0)

    def test_packet_boundaries_preserve_pcm(self):
        p = core.Playback()
        pcm = struct.pack("<h", 4000) * 960
        p.add(pcm[:600])
        p.add(pcm[600:])
        self.assertEqual(p.tick(0.02) + p.tick(0.04), pcm)
        self.assertAlmostEqual(core.overlap(p.intervals, 0, 0.04), 0.04)

    def test_idle_gap_preserved(self):
        p = core.Playback()
        p.add(struct.pack("<h", 4000) * 480)
        p.tick(0.02)
        p.add(struct.pack("<h", 4000) * 480)
        p.tick(0.5)
        self.assertEqual(len(p.intervals), 2)

    def test_unexposed_gold_is_unscorable(self):
        result = core.summarize(
            {
                "mode": "stream",
                "gold_events": [{"type": "pause", "interval_s": [2, 3]}],
            },
            [],
            [],
        )
        self.assertFalse(result["events"][0]["scorable"])

    def test_no_response_not_success(self):
        result = core.summarize({"mode": "reactive"}, [], [])
        self.assertFalse(result["output_observed"])
        self.assertFalse(result["triggered"])

    def test_event_mapping_uses_actual_send_time(self):
        trace = [
            {
                "event": "input_chunk",
                "t": 2.2,
                "media_end_s": 2.0,
                "source": "input",
                "lateness_s": 0.2,
            },
            {
                "event": "input_chunk",
                "t": 3.2,
                "media_end_s": 3.0,
                "source": "input",
                "lateness_s": 0.2,
            },
        ]
        result = core.summarize(
            {
                "mode": "stream",
                "gold_events": [{"type": "pause", "interval_s": [2, 3]}],
            },
            trace,
            [[2.1, 2.3]],
        )
        self.assertAlmostEqual(result["events"][0]["playback_during_event_s"], 0.1)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from jevarena_mm.streaming import gemini

        cls.runmod = gemini

    async def test_chunk_not_sent_before_final_sample_time(self):
        from unittest.mock import patch
        from types import ModuleType, SimpleNamespace
        import tempfile, time

        sent = []

        class Session:
            async def send_realtime_input(self, **kwargs):
                sent.append((time.monotonic(), kwargs["audio"].data))

        google = ModuleType("google")
        genai = ModuleType("google.genai")
        genai.types = SimpleNamespace(Blob=lambda **kw: SimpleNamespace(**kw))
        google.genai = genai
        with tempfile.TemporaryDirectory() as tmp:
            episode = self.runmod.Episode({}, Path(tmp), Path(tmp), 20)
            episode.start = time.monotonic()
            begin = episode.start
            with patch.dict(sys.modules, {"google": google, "google.genai": genai}):
                await episode.send_pcm(Session(), b"\1\0" * 800, "input")
            self.assertEqual([len(pcm) // 2 for _, pcm in sent], [320, 320, 160])
            for (when, _), due in zip(sent, [0.02, 0.04, 0.05]):
                self.assertGreaterEqual(when - begin, due)
            self.assertEqual(b"".join(pcm for _, pcm in sent), b"\1\0" * 800)

    async def test_reactive_trigger_requires_playback_feedback(self):
        from unittest.mock import patch
        import tempfile, time

        with tempfile.TemporaryDirectory() as tmp:
            case = {
                "mode": "reactive",
                "audio": {},
                "user_event": {},
                "trigger_after_audible_s": 1,
            }
            episode = self.runmod.Episode(case, Path(tmp), Path(tmp), 100)
            episode.start = time.monotonic()
            sources = []
            checks = []

            async def send(session, pcm, source):
                sources.append(source)

            async def silence(session, seconds):
                checks.append(seconds)
                # Simulated playback feedback becomes available only after waiting.
                episode.playback.audible_frames = 24000
                episode.playback.last_audible = time.monotonic() - episode.start
                episode.playback.buffered = 48000

            episode.send_pcm = send
            episode.silence = silence
            with patch.object(self.runmod, "pcm_file", return_value=b"\1\0"):
                await episode.drive(None)
            self.assertEqual(sources, ["input", "user_event"])
            self.assertEqual(checks, [0.1, 8])
            self.assertEqual(
                sum(e["event"] == "reactive_user_event" for e in episode.trace), 1
            )

    async def test_gold_not_in_model_config(self):
        self.assertNotIn("gold_events", str(self.runmod.CONFIG))
        self.assertNotIn("source_id", str(self.runmod.CONFIG))


if __name__ == "__main__":
    unittest.main()
