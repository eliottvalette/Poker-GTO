"""Exploratory calibration selection reports boundary minima without convergence claims."""
import unittest
from itertools import combinations
from scripts.hu_coverage_experiment import choose_epoch, opening_probe_gates


class CoverageExperimentTests(unittest.TestCase):
    def test_earliest_epoch_within_relative_minimum(self):
        rows = [{"epoch": epoch, "heldout_loss": loss} for epoch, loss in ((2, 2), (5, 1.04), (10, 1), (20, 1.1), (50, 1.2))]
        chosen = choose_epoch(rows)
        self.assertEqual(chosen["epochs"], 5)
        self.assertEqual(chosen["minimum_epoch"], 10)
        self.assertFalse(chosen["convergence_established"])
        self.assertFalse(chosen["minimum_at_last_measurement"])

    def test_boundary_minimum_is_not_convergence(self):
        chosen = choose_epoch([{"epoch": 2, "heldout_loss": 2}, {"epoch": 50, "heldout_loss": 1}])
        self.assertEqual(chosen["epochs"], 50)
        self.assertTrue(chosen["minimum_at_last_measurement"])
        self.assertFalse(chosen["convergence_established"])

    def test_invalid_measurement_fails_explicitly(self):
        for rows in ([], [{"epoch": 2, "heldout_loss": float("nan")}], [{"epoch": 2, "heldout_loss": -1}]):
            with self.assertRaises(ValueError):
                choose_epoch(rows)


class OpeningProbeGateTests(unittest.TestCase):
    def test_total_variation_is_half_l1_over_class_mean_probabilities(self):
        combos = list(combinations(range(52), 2))
        probabilities = [[.1, .9] if c[0] // 4 == c[1] // 4 else [.4, .6] for c in combos]
        result = opening_probe_gates({"combos": combos, "probabilities": probabilities,
                                      "actions": ["FOLD", "CALL"], "max_card_order_gap": 0})
        self.assertAlmostEqual(result["tv_AA_72o"], .3)
        self.assertAlmostEqual(result["tv_KK_32o"], .3)
        self.assertTrue(result["tv_separation_pass"])
        self.assertTrue(result["strong_fold_below_weak"])
        self.assertTrue(result["card_order_exact"])
