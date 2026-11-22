"""Publish counts and conditional timing statistics without transcripts or audio."""

import statistics

CATEGORIES = (
    "candor_turn_taking",
    "candor_pause_handling",
    "user_interruption",
    "user_backchannel",
)


def stats(values):
    values = [x for x in values if x is not None]
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def aggregate(manifest, progress):
    expected = {c["id"]: c for c in manifest["cases"]}
    if len(expected) != len(manifest["cases"]):
        raise ValueError("Duplicate manifest case ID")
    by_id = {}
    for r in progress["results"]:
        if r["id"] not in expected or r["id"] in by_id:
            raise ValueError("Unknown or repeated result ID")
        if r["category"] != expected[r["id"]]["category"]:
            raise ValueError("Result category mismatch")
        if r["status"] not in ("completed", "failed"):
            raise ValueError("Unknown result status")
        by_id[r["id"]] = r
    rows = []
    for category in CATEGORIES:
        ids = [i for i, c in expected.items() if c["category"] == category]
        observed = [by_id[i] for i in ids if i in by_id]
        good = [r for r in observed if r["status"] == "completed"]
        m = [r["metrics"] for r in good]
        row = {
            "category": category,
            "planned": len(ids),
            "completed": len(good),
            "failed": sum(r["status"] == "failed" for r in observed),
            "pending": len(ids) - len(observed),
            "output_observed": sum(x["output_observed"] for x in m),
            "right_censored": sum(x["observation_censored"] for x in m),
            "max_send_lateness_ms": stats([x["max_send_lateness_ms"] for x in m]),
            "episodes_send_lateness_over_200ms": sum(
                (x["max_send_lateness_ms"] or 0) > 200 for x in m
            ),
        }
        if category.startswith("candor"):
            all_events = [e for x in m for e in x["events"] if e["scorable"]]
            row["annotated_events_planned"] = sum(
                len(expected[i]["gold_events"]) for i in ids
            )
            row["scorable_events"] = len(all_events)
            if category == "candor_turn_taking":
                row["first_audio_after_boundary_ms"] = stats(
                    [e["first_audio_after_boundary_ms"] for e in all_events]
                )
                row["events_with_early_playback"] = sum(
                    e["playback_before_boundary_s"] > 0 for e in all_events
                )
            else:
                row["pause_events_with_overlap"] = sum(
                    e["playback_during_event_s"] > 0 for e in all_events
                )
                row["pause_overlap_s"] = sum(
                    e["playback_during_event_s"] for e in all_events
                )
                row["episodes_no_pause_overlap_and_output"] = sum(
                    x["output_observed"]
                    and bool(x["events"])
                    and all(
                        e["scorable"] and e["playback_during_event_s"] == 0
                        for e in x["events"]
                    )
                    for x in m
                )
        else:
            triggered = [x for x in m if x["triggered"]]
            row["triggered"] = len(triggered)
            row["provider_interruption_signals"] = sum(
                x["provider_interrupt_after_trigger_ms"] is not None for x in triggered
            )
            row["provider_interrupt_after_trigger_ms"] = stats(
                [x["provider_interrupt_after_trigger_ms"] for x in triggered]
            )
            row["playback_after_user_event"] = sum(
                x["response_after_user_event"] for x in triggered
            )
            if any("controller_stop_after_trigger_ms" in x for x in triggered):
                row["controller_stop_after_trigger_ms"] = stats(
                    [x.get("controller_stop_after_trigger_ms") for x in triggered]
                )
            row["semantic_correctness"] = "not_scored"
        rows.append(row)
    return {
        "planned": len(expected),
        "completed": sum(r["completed"] for r in rows),
        "failed": sum(r["failed"] for r in rows),
        "pending": sum(r["pending"] for r in rows),
        "categories": rows,
    }
