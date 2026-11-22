import argparse
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.playback_ledger import apply_playback_ack

p = argparse.ArgumentParser()
p.add_argument("--expect-bug", action="store_true")
a = p.parse_args()
s = DuplexEngineSession(
    session_id="audio-only-regression", config=DuplexSessionConfig()
)
r = s.begin_response()
s.mark_audio_sent(3416, audio_complete=True)
apply_playback_ack(s, {"response_id": r, "played_ms": 976})
s.end_response()
admitted = s.has_assistant_response_item(r, "item_" + r)
if a.expect_bug:
    assert not admitted, "Original bug was not reproduced"
    print("Original runtime reproduces missing audio-only response after end_response")
else:
    assert admitted, "Audio-only response dropped before playback drained"
    for t in [996, 2000, 3416]:
        events = apply_playback_ack(s, {"response_id": r, "played_ms": t})
        assert all(type(e).__name__ == "PlaybackAcknowledged" for e in events), events
        assert s.playback_for_response(r).played_ms == t
    assert not s.has_assistant_response_item("unknown", "item_unknown")
    assert not any(m.get("content") for m in s._conversation.messages), (
        "Fabricated transcript"
    )
    unknown = apply_playback_ack(s, {"response_id": "unknown", "played_ms": 20})
    assert not any(type(e).__name__ == "PlaybackAcknowledged" for e in unknown)
    next_id = s.begin_response()
    s.mark_audio_sent(100)
    events = apply_playback_ack(s, {"response_id": r, "played_ms": 3416})
    assert all(type(e).__name__ == "PlaybackAcknowledged" for e in events)
    assert s.playback_for_response(next_id).played_ms == 0
    empty = DuplexEngineSession(
        session_id="empty-regression", config=DuplexSessionConfig()
    )
    empty_id = empty.begin_response()
    empty.end_response()
    assert not empty.has_assistant_response_item(empty_id, "item_" + empty_id)
    print(
        "Audio-only drain and next-response isolation pass; unknown and empty responses rejected; no transcript fabricated"
    )
