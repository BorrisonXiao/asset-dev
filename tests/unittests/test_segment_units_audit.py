"""Timing/overlap checks independent of the model and CUDA."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'recipes/LibriSpeech/ASR/transformer/audit_segment_units.py'
SPEC = importlib.util.spec_from_file_location('segment_units_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class TimingTests(unittest.TestCase):
    def test_uniform_control_preserves_count_and_frame_lattice(self):
        cuts = audit.count_matched_grid(101, 21)
        self.assertEqual(len(cuts), 21)
        self.assertTrue(all(abs(t/.02 - round(t/.02)) < 1e-8 for t in cuts))
        self.assertEqual(audit.count_matched_grid(50, 4), [.2, .4, .6, .8])

    def test_one_to_one_and_inclusive_tolerance(self):
        self.assertEqual(audit.match_count([.99, 1.01], [1.], .02), 1)
        self.assertEqual(audit.match_count([1., 1.03], [1.02, 1.05], .02), 2)
        self.assertEqual(audit.match_count([], [1.], .02), 0)

    def test_phone_edges_exclude_utterance_endpoints(self):
        intervals = [[0, .2, 'AH0'], [.2, .4, 'T'], [.4, .6, ''], [.6, .8, 'S']]
        self.assertEqual(audit.reference_edges(intervals, .8), [.2, .4, .6])

    def test_only_internal_gaps_are_selected(self):
        words = [[0, .3, ''], [.3, .5, 'a'], [.5, .75, ''], [.75, 1., 'word'], [1., 1.5, '']]
        self.assertEqual(audit.pauses(words), [[.5, .75, 'a', 'word']])

    def test_silent_tokens_require_interval_overlap(self):
        aligned = audit.gap_stats([.2, .5], 1., [.2, .5, 'a', 'word'])
        self.assertEqual(aligned['pure_tokens'], 1)
        self.assertEqual(aligned['internal_cuts'], 0)
        self.assertEqual(aligned['onset_error_ms'], 0)
        self.assertEqual(aligned['offset_error_ms'], 0)
        split = audit.gap_stats([.3, .4], 1., [.2, .5, 'a', 'word'])
        self.assertEqual(split['pure_tokens'], 1)
        self.assertEqual(split['internal_cuts'], 2)
        merged = audit.gap_stats([], 1., [.2, .5, 'a', 'word'])
        self.assertEqual(merged['mostly_silent_tokens'], 0)


if __name__ == '__main__':
    unittest.main()
