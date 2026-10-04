import json
import math
import unittest

from infoset import observe
from scripts.outcome_diagnostics import (full_tournament_path, json_safe, probe_strategy,
                                         quantiles, run_probe, _raw_weights, _summarize)
from tournament import TournamentState
import random


class OutcomeDiagnosticTests(unittest.TestCase):
    def test_quantiles_preserve_zero_and_structural_zero_counts(self):
        result = quantiles([-math.inf, 0.0, 2.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["finite_count"], 3)
        self.assertEqual(result["zero_count"], 1)
        self.assertEqual(result["negative_infinity_count"], 1)
        self.assertEqual(result["p50"], 2.0)
        self.assertEqual(quantiles([])["p50"], None)
        self.assertEqual(quantiles([-math.inf])["p50"], None)
        with self.assertRaises(ValueError):
            quantiles([math.nan])

    def test_json_zero_logs_remain_explicit_without_nonstandard_numbers(self):
        self.assertEqual(json_safe({"logs": [-math.inf, 0.0]}), {"logs": [None, 0.0]})
        with self.assertRaises(ValueError):
            json_safe(math.inf)

    def test_probe_keeps_every_censored_attempt_and_emits_no_partial_targets(self):
        report = run_probe((2,), max_nodes=1, max_depth=1)
        self.assertEqual(len(report["rows"]), 8)
        self.assertEqual(report["summary"]["counts"]["full_tournament"]["censored"], 2)
        self.assertEqual(report["summary"]["counts"]["outcome_traverser"]["censored"], 6)
        for row in report["rows"]:
            self.assertEqual(row["status"], "censored")
            self.assertTrue(row["failure"])
            if row["kind"] == "outcome_traverser":
                self.assertEqual(row["regret_samples"], 0)
                self.assertEqual(row["average_samples"], 0)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["summary"]["actual_terminal_hand_lengths"]["count"], 0)

    def test_full_paths_repeat_under_same_seed_and_q(self):
        a = full_tournament_path(3, probe_strategy("mild_skew"), 8, 8, 0.6)
        b = full_tournament_path(3, probe_strategy("mild_skew"), 8, 8, 0.6)
        a.pop("seconds")
        b.pop("seconds")
        self.assertEqual(a, b)
        self.assertEqual(a["status"], "censored")
        self.assertGreater(a["log_inverse_q"], 0)

    def test_diagnostic_policies_are_valid_distinct_full_support_distributions(self):
        state = TournamentState(rng=random.Random(1))
        state.start_hand()
        obs = observe(state)
        uniform = probe_strategy("uniform")(obs)
        skewed = probe_strategy("mild_skew")(obs)
        self.assertNotEqual(uniform, skewed)
        for distribution in (uniform, skewed):
            self.assertAlmostEqual(sum(distribution), 1)
            for legal, probability in zip(obs.legal_mask, distribution):
                self.assertGreater(probability, 0) if legal else self.assertEqual(probability, 0)
        with self.assertRaises(ValueError):
            probe_strategy("unknown")

    def test_raw_weights_distinguish_structural_zero_underflow_and_overflow(self):
        result = _raw_weights([-math.inf, -1000.0, 0.0, 1000.0])
        self.assertEqual(result["values"], [0.0, None, 1.0, None])
        self.assertEqual(result["structural_zero_count"], 1)
        self.assertEqual(result["underflow_count"], 1)
        self.assertEqual(result["overflow_count"], 1)
        self.assertEqual(result["finite_positive_count"], 1)
        json.dumps(result, allow_nan=False)

    def test_raw_quantiles_are_computed_from_weights_not_exponentiated_log_quantiles(self):
        row = {"kind": "outcome_traverser", "status": "settled_payoff", "nodes": 2,
               "decisions": 1, "hand_number": 1, "seconds": 0.0,
               "diagnostics": {"log_average_weights": [0.0, math.log(9.0)], "max_abs_regret": 0.0},
               "raw_importance_weights": {"log_average_weights": _raw_weights([0.0, math.log(9.0)])}}
        result = _summarize([row])["raw_importance_weights"]["log_average_weights"]
        self.assertAlmostEqual(result["finite_represented_quantiles"]["p50"], 5.0)
        self.assertEqual(result["overflow_excluded_count"], 0)
        self.assertEqual(result["underflow_excluded_count"], 0)

    def test_invalid_probe_configuration_is_not_silently_repaired(self):
        for kwargs in ({"seeds": ()}, {"seeds": (1, 1)}, {"max_nodes": 0},
                       {"max_depth": 0}, {"epsilon": 0.0}, {"epsilon": math.nan}):
            with self.assertRaises(ValueError):
                run_probe(**kwargs)
