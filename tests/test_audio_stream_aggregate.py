import unittest

from jevarena_mm.streaming import aggregate as report


def case(i="one", category="user_interruption"):
    return {"id": i, "category": category, "gold_events": []}


def result(i="one", status="completed", category="user_interruption", **metrics):
    return {
        "id": i,
        "category": category,
        "status": status,
        "metrics": {
            "output_observed": False,
            "observation_censored": False,
            "max_send_lateness_ms": 0,
            "triggered": False,
            **metrics,
        },
    }


class SuiteReportTests(unittest.TestCase):
    def test_missing_stays_in_denominator(self):
        a = report.aggregate(
            {"cases": [case("one"), case("two")]}, {"results": [result()]}
        )
        self.assertEqual((a["planned"], a["completed"], a["pending"]), (2, 1, 1))

    def test_failure_not_counted_as_output_success(self):
        a = report.aggregate(
            {"cases": [case()]},
            {"results": [result(status="failed", output_observed=True)]},
        )
        self.assertEqual(a["failed"], 1)
        self.assertEqual(a["categories"][2]["output_observed"], 0)

    def test_no_trigger_not_in_latency_stat(self):
        a = report.aggregate({"cases": [case()]}, {"results": [result()]})
        r = a["categories"][2]
        self.assertEqual(r["triggered"], 0)
        self.assertEqual(r["provider_interrupt_after_trigger_ms"]["n"], 0)

    def test_unknown_or_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            report.aggregate({"cases": [case()]}, {"results": [result("other")]})
        with self.assertRaises(ValueError):
            report.aggregate({"cases": [case()]}, {"results": [result(), result()]})

    def test_silent_model_does_not_pass_pause_test(self):
        c = case(category="candor_pause_handling")
        c["gold_events"] = [{"type": "pause", "interval_s": [1, 2]}]
        r = result(
            category="candor_pause_handling",
            events=[{"scorable": True, "playback_during_event_s": 0}],
        )
        a = report.aggregate({"cases": [c]}, {"results": [r]})
        self.assertEqual(a["categories"][1]["episodes_no_pause_overlap_and_output"], 0)
        self.assertEqual(a["categories"][1]["scorable_events"], 1)

    def test_censoring_and_scheduling_exposed(self):
        a = report.aggregate(
            {"cases": [case()]},
            {"results": [result(observation_censored=True, max_send_lateness_ms=250)]},
        )
        self.assertEqual(a["categories"][2]["right_censored"], 1)
        self.assertEqual(a["categories"][2]["episodes_send_lateness_over_200ms"], 1)


if __name__ == "__main__":
    unittest.main()
