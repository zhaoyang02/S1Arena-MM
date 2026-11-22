"""Dependency-free playback accounting and diagnostic scoring (no model labels)."""

from collections import deque
import math
import struct

RATE = 24000


def rms(pcm):
    if not pcm:
        return 0.0
    samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
    return math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768


class Playback:
    """Clocked virtual PCM sink. Never equate received bytes with played audio."""

    def __init__(self):
        self.queue = deque()
        self.buffered = 0
        self.played_frames = 0
        self.audible_frames = 0
        self.intervals = []
        self.last_audible = None

    def add(self, pcm):
        if len(pcm) % 2:
            raise ValueError("Invalid PCM16 byte length")
        self.queue.append(pcm)
        self.buffered += len(pcm)

    def clear(self):
        discarded = self.buffered / (2 * RATE)
        self.queue.clear()
        self.buffered = 0
        return discarded

    def tick(self, now, frames=480):
        needed = frames * 2
        parts = []
        while needed and self.queue:
            data = self.queue.popleft()
            take = min(needed, len(data))
            parts.append(data[:take])
            if take < len(data):
                self.queue.appendleft(data[take:])
            self.buffered -= take
            needed -= take
        pcm = b"".join(parts)
        duration = len(pcm) / (2 * RATE)
        self.played_frames += len(pcm) // 2
        if rms(pcm) > 0.005:
            # Block is consumed at its END, so timestamps refer to playback time.
            start = max(0.0, now - duration)
            if self.intervals and start - self.intervals[-1][1] <= 0.025:
                self.intervals[-1][1] = now
            else:
                self.intervals.append([start, now])
            self.audible_frames += len(pcm) // 2
            self.last_audible = now
        return pcm


def overlap(intervals, a, b):
    return sum(max(0.0, min(y, b) - max(x, a)) for x, y in intervals)


def summarize(case, trace, intervals):
    """Transparent local diagnostics; NOT the official FDB scoring implementation."""
    # A transport may receive cleanup audio while closing a session. It cannot
    # create output coverage or cancellation events inside the observation window.
    deadline = next((e for e in trace if e["event"] == "observation_end"), None)
    if deadline:
        cutoff = deadline["t"]
        trace = [e for e in trace if e["t"] <= cutoff]
        intervals = [[a, min(b, cutoff)] for a, b in intervals if a < cutoff]
    out = {
        "audible_playback_s": sum(b - a for a, b in intervals),
        "output_observed": bool(intervals),
        "official_fdb_score": False,
    }
    ends = [e for e in trace if e["event"] == "observation_end"]
    out["observation_end_s"] = ends[-1]["t"] if ends else None
    out["unplayed_buffer_at_end_s"] = ends[-1]["buffered_s"] if ends else None
    out["observation_censored"] = bool(ends and ends[-1]["buffered_s"] > 0)
    out["active_generation_at_deadline"] = bool(
        deadline and deadline.get("active_responses", 0)
    )
    out["observation_censored"] |= out["active_generation_at_deadline"]
    out["controller_pending_at_deadline"] = bool(
        ends
        and (ends[-1].get("controller_pending") or ends[-1].get("controller_queued"))
    )
    out["observation_censored"] |= out["controller_pending_at_deadline"]
    packets = [e for e in trace if e["event"] == "input_chunk"]
    out["max_send_lateness_ms"] = max(
        (e["lateness_s"] * 1000 for e in packets), default=None
    )
    if case["mode"] == "stream":
        windows = case["gold_events"]

        # Map media time to the actual completion of the packet exposing it.
        def wall(media_s):
            for e in packets:
                if e["source"] == "input" and e["media_end_s"] >= media_s:
                    return e["t"] - (e["media_end_s"] - media_s)
            return None

        out["events"] = []
        for event in windows:
            a, b = [wall(t) for t in event["interval_s"]]
            if a is None or b is None:
                out["events"].append({"type": event["type"], "scorable": False})
                continue
            m = {
                "type": event["type"],
                "scorable": True,
                "playback_during_event_s": overlap(intervals, a, b),
            }
            if event["type"] == "turn":
                starts = [max(x, a) for x, y in intervals if y > a]
                m["first_audio_after_boundary_ms"] = (
                    (min(starts) - a) * 1000 if starts else None
                )
                m["playback_before_boundary_s"] = overlap(intervals, 0, a)
            out["events"].append(m)
    else:
        triggers = [e for e in trace if e["event"] == "reactive_user_event"]
        out["triggered"] = bool(triggers)
        if triggers:
            trigger = triggers[0]["t"]
            out["assistant_audible_at_trigger"] = triggers[0]["assistant_audible"]
            signals = [
                e["t"]
                for e in trace
                if e["event"] == "interrupted" and e["t"] >= trigger
            ]
            out["provider_interrupt_after_trigger_ms"] = (
                (signals[0] - trigger) * 1000 if signals else None
            )
            if any(e["event"] == "decision_start" for e in trace):
                stops = [
                    e["t"]
                    for e in trace
                    if e["event"] == "controller_stop" and e["t"] >= trigger
                ]
                out["controller_stop_after_trigger_ms"] = (
                    (stops[0] - trigger) * 1000 if stops else None
                )
            out["playback_overlap_first_1s"] = overlap(intervals, trigger, trigger + 1)
            end = next(
                (e["t"] for e in trace if e["event"] == "reactive_user_event_end"), None
            )
            out["response_after_user_event"] = bool(
                end is not None and any(y > end for x, y in intervals)
            )
            # Do not call continued old audio a semantically correct new response.
            out["semantic_followup_correctness"] = "not_scored"
    return out
