"""Opening diagnostic units, physical settlement and paired reproducibility."""
import unittest

from hybrid.policy_source import UniformLegalPolicy
from training.preflop_diagnostics import opening_matrix, paired_opening_values


class PreflopDiagnosticsTests(unittest.TestCase):
    def test_matrix_preserves_exact_combo_multiplicity_and_uniform_policy(self):
        report = opening_matrix(UniformLegalPolicy(), 2, 25)
        self.assertEqual(sum(row['combos'] for row in report['classes'].values()), 1326)
        self.assertEqual(report['classes']['AA']['combos'], 6)
        self.assertEqual(report['classes']['AKs']['combos'], 4)
        self.assertEqual(report['classes']['AKo']['combos'], 12)
        self.assertEqual(report['card_reversal_max_error'], 0)
        for row in report['classes'].values():
            for probability in row['probabilities'].values():
                self.assertAlmostEqual(probability, 1 / 7)

    def test_fold_utility_includes_blinds_and_paired_results_repeat(self):
        for count, expected in ((2, -0.5), (3, 0.0)):
            report = paired_opening_values(UniformLegalPolicy(), count, 5, (48, 49), 8)
            self.assertEqual(report['actions']['FOLD']['ev']['mean'], expected)
            self.assertEqual(report['actions']['FOLD']['ev']['se'], 0)
            self.assertEqual(report, paired_opening_values(UniformLegalPolicy(), count, 5, (48, 49), 8))
            self.assertAlmostEqual(sum(row['probability'] for row in report['actions'].values()), 1)

    def test_invalid_holding_rejected(self):
        with self.assertRaises(ValueError):
            paired_opening_values(UniformLegalPolicy(), 2, 25, (48, 48), 2)
