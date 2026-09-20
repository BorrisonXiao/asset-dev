"""Report-only scoring and snapshot checks; no model imports or GPU work."""

import importlib.util
import json
from pathlib import Path
import re
import statistics
from types import SimpleNamespace
import unittest
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "recipes/LibriSpeech/ASR/transformer/adaptive_frequency_report.py"
SPEC = importlib.util.spec_from_file_location("adaptive_frequency_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class TestReportScoring(unittest.TestCase):
    def row(self, hypothesis="A B", errors=0):
        return dict(id="example", hypothesis=hypothesis, reference="A B",
                    word_errors=errors, reference_words=2)

    def test_matching_scoring_conventions(self):
        result = REPORT.audit_transcripts([self.row()], 0, 2)
        self.assertEqual(result["whitespace_differences"], [])
        self.assertEqual(result["normalized_WER"], 0)

    def test_leading_space_is_explicit_not_silently_fixed(self):
        result = REPORT.audit_transcripts([self.row(" A B")], 1, 2)
        self.assertEqual(result["main_errors"], 1)
        self.assertEqual(result["normalized_errors"], 0)
        self.assertEqual(len(result["whitespace_differences"]), 1)

    def test_trailing_and_repeated_whitespace(self):
        for hypothesis in ("A B ", "A  B"):
            result = REPORT.audit_transcripts([self.row(hypothesis)], 1, 2)
            self.assertEqual(result["normalized_errors"], 0)

    def test_empty_hypothesis(self):
        result = REPORT.audit_transcripts([self.row("", 2)], 2, 2)
        self.assertEqual(result["normalized_WER"], 100)

    def test_unknown_main_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Main WER report"):
            REPORT.audit_transcripts([self.row()], 1, 2)

    def test_bad_diagnostic_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "Diagnostic error"):
            REPORT.audit_transcripts([self.row(errors=1)], 0, 2)

    def test_complete_snapshot(self):
        data = json.loads(REPORT.SNAPSHOT.read_text())
        self.assertEqual({(r["seed"], r["arm"]) for r in data["runs"]},
                         {(seed, arm) for seed in (3407, 3408, 3409) for arm in REPORT.ARMS})
        self.assertEqual(data["config"], REPORT.CONFIG)
        self.assertEqual(sum(len(t["whitespace_differences"]) for r in data["runs"]
                             for t in r["tests"].values()), 8)
        for run in data["runs"]:
            self.assertEqual(run["events"][-1]["step"], 8000)
            self.assertTrue(run["events"][-1]["training_done"])
            for split, count in REPORT.SPLITS.items():
                self.assertEqual(run["tests"][split]["utterances"], count)

    def test_reported_three_seed_means(self):
        data = json.loads(REPORT.SNAPSHOT.read_text())
        mean, sd = REPORT.summary(data, "adaptive", "test-other", "WER")
        self.assertAlmostEqual(mean, 5.151914614497959)
        self.assertGreater(sd, 0)
        data["runs"] = data["runs"][:-1]
        with self.assertRaisesRegex(ValueError, "three seeds"):
            REPORT.summary(data, "adaptive", "test-other", "WER")


class TestReportCurves(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(REPORT.SNAPSHOT.read_text())
        self.runs = REPORT.training_runs(self.data)
        self.hk = SimpleNamespace(C={"blue": "var(--series-sed)",
                                     "orange": "var(--pair-sed-asr)",
                                     "green": "var(--series-asr)",
                                     "red": "var(--diverge-neg)"})

    def charts(self):
        markup = REPORT.training_figures(self.hk, self.data)
        return {svg.attrib["id"]: svg for svg in
                map(ET.fromstring, re.findall(r"<svg\b.*?</svg>", markup, re.S))}

    def test_means_and_sample_sd_at_every_check(self):
        for arm in REPORT.ARMS:
            for key in ("other_hz", "other_wer"):
                curve = REPORT.curve_summary(self.runs, arm, key)
                self.assertEqual([p[0] for p in curve], list(range(0, 8001, 500)))
                for i, (_, mean, sd) in enumerate(curve):
                    values = [self.runs[(s, arm)]["events"][i][key] for s in (3407, 3408, 3409)]
                    self.assertEqual(mean, statistics.mean(values))
                    self.assertEqual(sd, statistics.stdev(values))

    def test_incomplete_misaligned_or_duplicate_histories_rejected(self):
        self.data["runs"][0]["events"].pop(5)
        with self.assertRaisesRegex(ValueError, "validation steps"):
            REPORT.training_runs(self.data)
        self.data = json.loads(REPORT.SNAPSHOT.read_text())
        self.data["runs"][0] = self.data["runs"][1]
        with self.assertRaisesRegex(ValueError, "nine unique"):
            REPORT.training_runs(self.data)

    def test_bad_ema_or_nonfinite_values_rejected(self):
        self.data["runs"][0]["events"][2]["ema_other_wer"] += 0.1
        with self.assertRaisesRegex(ValueError, "EMA does not reproduce"):
            REPORT.training_runs(self.data)
        self.data["runs"][0]["events"][0]["other_hz"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            REPORT.training_runs(self.data)

    def test_target_is_post_step_not_interpolated(self):
        self.assertEqual(REPORT.step_points([(0, 11), (500, 11), (1000, 10.5)]),
                         [(0, 11), (500, 11), (500, 11), (1000, 11), (1000, 10.5)])
        with self.assertRaisesRegex(ValueError, "must increase"):
            REPORT.step_points([(500, 11), (0, 10.5)])
        for seed in (3407, 3408, 3409):
            events = self.runs[(seed, "adaptive")]["events"]
            points = REPORT.step_points([(e["step"], e["target_hz"]) for e in events])
            self.assertEqual(len(points), 33)
            self.assertTrue(all(x0 == x1 or y0 == y1 for (x0, y0), (x1, y1) in zip(points, points[1:])))
            for previous, event in zip(events, events[1:]):
                self.assertEqual(event["target_before_hz"], previous["target_hz"])

    def test_rendered_values_and_events_match_logs(self):
        charts = self.charts()
        self.assertEqual(len(charts), 8)  # Two panels plus three rate/WER pairs.
        for chart in charts.values():
            self.assertEqual(chart.attrib["role"], "img")
            for identifier in chart.attrib["aria-labelledby"].split():
                self.assertIsNotNone(chart.find(f".//*[@id='{identifier}']"))
        for seed in (3407, 3408, 3409):
            events = self.runs[(seed, "adaptive")]["events"]
            rate = charts[f"adaptive-seed-{seed}-rate"]
            squares = rate.findall(".//*[@data-event='lower_target']")
            self.assertEqual([int(p.attrib["data-step"]) for p in squares],
                             [e["step"] for e in events if e["action"] == "lower_target"])
            wer = charts[f"adaptive-seed-{seed}-wer"]
            rings = wer.findall(".//*[@data-event='quality_failed']")
            self.assertEqual([int(p.attrib["data-step"]) for p in rings], [1000] if seed == 3409 else [])
            for chart, key, ylim, series in ((rate, "other_hz", (8.5, 12), "measured-rate"),
                                            (wer, "other_wer", (4.6, 6), "raw-wer"),
                                            (wer, "ema_other_wer", (4.6, 6), "ema-wer")):
                polyline = chart.find(f".//polyline[@data-series='{series}']")
                points = [tuple(map(float, point.split(','))) for point in polyline.attrib["points"].split()]
                self.assertEqual(len(points), 17)
                for (x, y), event in zip(points, events):
                    self.assertAlmostEqual(x, 52 + event["step"] / 8000 * 350, delta=0.00051)
                    self.assertAlmostEqual(y, 230 - (event[key] - ylim[0]) / (ylim[1] - ylim[0]) * 215, delta=0.00051)
            guard = events[0]["other_wer"] + 0.1
            guard_points = wer.find(".//polyline[@data-series='wer-limit']").attrib["points"].split()
            self.assertEqual(len(guard_points), 2)
            self.assertEqual(guard_points[0].split(',')[1], guard_points[1].split(',')[1])
            self.assertAlmostEqual(float(guard_points[0].split(',')[1]), 230 - (guard-4.6)/1.4*215, delta=0.00051)
            self.assertTrue(all(e["clean_wer"] <= events[0]["clean_wer"] + 0.1 for e in events))

    def test_comparison_bands_and_explanations(self):
        charts = self.charts()
        for metric in ("other_hz", "other_wer"):
            chart = charts[f"adaptive-comparison-{metric}"]
            self.assertEqual(len(chart.findall("polygon")), 3)
            self.assertEqual(len(chart.findall("polyline")), 3)
            self.assertTrue(all(len(p.attrib["points"].split()) == 34 for p in chart.findall("polygon")))
        markup = REPORT.training_figures(self.hk, self.data)
        self.assertEqual(markup.count('<figure '), 2)
        self.assertEqual(markup.count('<figcaption '), 2)
        for explanation in ("sample SD, not a confidence interval", "without smoothing", "after</b> each check",
                            "8,000-update budget", "separate dev-clean guard", "rate penalty stays on"):
            self.assertIn(explanation, markup)

    def test_axes_refuse_to_clip_values_silently(self):
        with self.assertRaisesRegex(ValueError, "outside axes"):
            REPORT.curve_svg("test", "Test", "Test", [dict(key="outside", color="blue", points=[(0, 15)])],
                             (8.5, 12), (9, 10, 11, 12), "Rate (Hz)")

    def test_removed_section_and_its_cross_references_stay_removed(self):
        self.hk.card = lambda inner, title=None: f'<div><h3>{title}</h3>{inner}</div>'
        self.hk.equation = lambda tex: tex
        self.hk.section = lambda title, lead, body: f'<section><h2>{title}</h2><p>{lead}</p>{body}</section>'
        markup = REPORT.build_section(self.hk, self.data)
        for removed in ("Scoring and provenance", "Scoring note:", "Checkpoint choice:",
                        "Exact initialization checkpoints and reproducible evidence",
                        "Download the audited JSON snapshot", "scoring note below", "downloadable evidence below"):
            self.assertNotIn(removed, markup)
        self.assertIn('id="adaptive-results"', markup)
        self.assertIn('id="adaptive-decisions"', markup)
        self.assertEqual(markup.count('<svg '), 8)


if __name__ == "__main__":
    unittest.main()
