"""Report-only scoring and snapshot checks; no model imports or GPU work."""

import importlib.util
import json
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
