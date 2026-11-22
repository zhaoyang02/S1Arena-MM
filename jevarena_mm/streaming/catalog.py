"""Stable MM names for the four existing audio-live-v1 categories."""

TASKS = {
    "candor_turn_taking": "st.audio/turn_taking/candor",
    "candor_pause_handling": "st.audio/pause_handling/candor",
    "user_interruption": "st.audio/user_interruption/full_duplex_bench_v15",
    "user_backchannel": "st.audio/backchannel/full_duplex_bench_v15",
}
PROTOCOL = "mm-audio-stream-v1"
SOURCE = "https://github.com/DanielLin94144/Full-Duplex-Bench"
QWEN_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


def select_cases(manifest, prefix="st.audio", limit=0):
    from ..registry import matches

    cases = []
    for category, name in TASKS.items():
        if matches(name, prefix):
            group = [c for c in manifest["cases"] if c["category"] == category]
            cases.extend(group[:limit] if limit else group)
    if not cases:
        raise ValueError(f"No streaming cases selected by {prefix!r}")
    return cases
