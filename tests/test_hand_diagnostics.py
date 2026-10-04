"""Hand diagnostics retain censored attempts and legal-target mask contracts."""
import json
import unittest
from scripts.hand_diagnostics import run_hand_probe


class HandDiagnosticsTests(unittest.TestCase):
    def test_failed_passes_do_not_publish_partial_targets_or_conditional_means(self):
        report = run_hand_probe(trials=2, max_nodes=1)
        self.assertEqual(len(report["rows"]), 32)
        for row in report["rows"]:
            self.assertEqual(row["status"], "censored")
            self.assertEqual(row["regret_targets"], [])
            self.assertEqual(row["regret_masks"], [])
            self.assertEqual(row["average_weights"], [])
            self.assertIsNone(row["value"])
        for comparison in report["comparisons"]:
            self.assertFalse(comparison["uncensored"])
            self.assertIsNone(comparison["within_six_standard_errors"])
        json.dumps(report, allow_nan=False)

    def test_same_hand_comparison_retains_masks_and_fixed_exploration(self):
        report = run_hand_probe(trials=2)
        self.assertEqual(report["objective"], "hand_chip_delta")
        self.assertEqual(report["epsilon"], 0.6)
        self.assertEqual(report["traversal_scope"], "current_hand")
        for row in report["rows"]:
            self.assertEqual(row["status"], "completed")
            self.assertEqual(len(row["regret_masks"]), len(row["regret_targets"]))
            for vector, mask in zip(row["regret_targets"], row["regret_masks"]):
                self.assertTrue(all(value == 0 for value, legal in zip(vector, mask) if not legal))
            if row["mode"] == "outcome_sampling":
                self.assertEqual(row["diagnostics"]["hand_number"], 1)
        json.dumps(report, allow_nan=False)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({"trials": 0}, {"max_nodes": 0}, {"max_depth": -1}, {"epsilon": 0}, {"deal_seed": "3"}):
            with self.assertRaises(ValueError):
                run_hand_probe(**kwargs)
