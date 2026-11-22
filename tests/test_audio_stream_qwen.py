import json, tempfile, unittest
from pathlib import Path
from jevarena_mm.streaming.qwen import Episode


class WS:
    def __init__(self, events):
        self.events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return json.dumps(next(self.events))
        except StopIteration:
            raise StopAsyncIteration


class ProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_clears_playback(self):
        with tempfile.TemporaryDirectory() as d:
            e = Episode({"id": "x"}, Path(d), Path(d))
            e.playback.add(bytes(9600))
            e.active.add("r")
            await e.receive(
                WS(
                    [
                        {
                            "type": "response.done",
                            "response": {"id": "r", "status": "cancelled"},
                        }
                    ]
                )
            )
            self.assertEqual(e.playback.buffered, 0)
            self.assertFalse(e.active)
            self.assertEqual(e.trace[-1]["event"], "interrupted")

    async def test_normal_finish_preserves_unplayed_audio(self):
        with tempfile.TemporaryDirectory() as d:
            e = Episode({"id": "x"}, Path(d), Path(d))
            e.playback.add(bytes(9600))
            e.active.add("r")
            await e.receive(
                WS(
                    [
                        {
                            "type": "response.done",
                            "response": {"id": "r", "status": "completed"},
                        }
                    ]
                )
            )
            self.assertEqual(e.playback.buffered, 9600)
            self.assertFalse(e.active)


if __name__ == "__main__":
    unittest.main()
