"""Scoring and evidence-integrity checks for the Table 3 audit."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "recipes/LibriSpeech/ASR/transformer/analyze_aishell_table3_errors.py"
SPEC = importlib.util.spec_from_file_location("aishell_error_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class TestAishellErrorAudit(unittest.TestCase):
    def test_reconstruct_and_score_chinese_word_alignments(self):
        # Artificial parser fixture, never presented as a model prediction.
        fixture = (
            "%WER 66.67 [ 2 / 3, 1 ins, 1 del, 0 sub ]\n"
            + "=" * 80 + "\n"
            "fixture, %WER 66.67 [ 2 / 3, 1 ins, 1 del, 0 sub ]\n"
            "北京 ; <eps> ; 欢迎 ; 你\n"
            " =   ;   I   ;  =   ; D \n"
            "北京 ; 很   ; 欢迎 ; <eps>\n"
            + "=" * 80 + "\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wer_test.txt"
            p.write_text(fixture)
            rows = audit.parse_predictions(p)
        self.assertEqual(rows["fixture"]["ref_words"], ["北京", "欢迎", "你"])
        self.assertEqual(rows["fixture"]["hyp_words"], ["北京", "很", "欢迎"])
        stats, _ = audit.score_run(rows, "")
        self.assertEqual(stats["ref_characters"], 5)
        self.assertEqual((stats["errors"], stats["dele"], stats["ins"]), (2, 1, 1))
        self.assertEqual(stats["cer"], 40)
        boundary_stats, _ = audit.score_run(rows, "_")
        self.assertEqual(boundary_stats["ref_characters"], 7)

    def test_substitution_deletion_insertion_and_empty_hypothesis(self):
        rows = {"a": {"ref_words": ["今天好"], "hyp_words": ["今天号"]},
                "b": {"ref_words": ["你好"], "hyp_words": []},
                "c": {"ref_words": ["你"], "hyp_words": ["你你"]}}
        stats, details = audit.score_run(rows, "")
        self.assertEqual(stats["ref_characters"], 6)
        self.assertEqual((stats["sub"], stats["dele"], stats["ins"]), (1, 2, 1))
        self.assertEqual(stats["empty_hypotheses"], 1)
        self.assertEqual(details["b"]["errors"], 2)

    def test_incomplete_evidence_cannot_write_results(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with self.assertRaisesRegex(FileNotFoundError, "All 12"):
                audit.collect(p, p / "output")
            self.assertFalse((p / "output/analysis.json").exists())

    def test_bad_or_duplicate_alignment_blocks_are_rejected(self):
        block = "x, %WER 0.00 [ 0 / 1, 0 ins, 0 del, 0 sub ]\n你\n=\n你\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wer.txt"
            p.write_text(block + "=" * 80 + "\n" + block)
            with self.assertRaisesRegex(ValueError, "Duplicate ID"):
                audit.parse_predictions(p)


if __name__ == "__main__":
    unittest.main()
