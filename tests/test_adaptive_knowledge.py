"""Paired uncertainty, bounded stopping, and family split ownership."""
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from actions import legal_actions
from hybrid.adaptive_labels import LabelBudget, generate, paired_summary, response_target, save
from hybrid.knowledge_benchmark import break_even
from hybrid.knowledge_curriculum import FAMILIES, board_texture, curriculum, fingerprint, manifest
from hybrid.policy_source import UniformLegalPolicy
from hybrid.state import instantiate
from test_hybrid_river import fixture


class AdaptiveKnowledgeTests(unittest.TestCase):
    def test_common_noise_cancels_in_paired_differences(self):
        rows = [{"a": 100 * (-1)**i + .5, "b": 100 * (-1)**i} for i in range(2048)]
        result = paired_summary(rows, utility_span_bb=1, alpha=.05, looks=4, tolerance_bb=.01)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["selected_action"], "a")
        self.assertEqual(result["pairs"][0]["paired_se_bb"], 0)
        self.assertLess(result["pairs"][0]["interval_bb"][0], .5)
        self.assertGreater(result["pairs"][0]["interval_bb"][1], .5)

    def test_small_gap_can_be_epsilon_optimal_without_ranking_claim(self):
        result = paired_summary([{"a": .001, "b": 0.}] * 4096, utility_span_bb=1,
                                alpha=.05, looks=4, tolerance_bb=.05)
        self.assertEqual(result["status"], "epsilon_optimal")
        self.assertGreater(result["pairs"][0]["interval_bb"][1], 0)
        self.assertLess(result["pairs"][0]["interval_bb"][0], 0)

    def test_repeated_look_correction_and_invalid_payoff_bound(self):
        rows = [{"a": 0., "b": .1}] * 32
        a = paired_summary(rows, utility_span_bb=1, alpha=.05, looks=1, tolerance_bb=0)
        b = paired_summary(rows, utility_span_bb=1, alpha=.05, looks=10, tolerance_bb=0)
        self.assertGreaterEqual(b["selection_loss_upper_bb"], a["selection_loss_upper_bb"])
        with self.assertRaisesRegex(ValueError, "conserved-chip"):
            paired_summary(rows, utility_span_bb=.01, alpha=.05, looks=1, tolerance_bb=0)

    def test_exact_river_reference_and_deterministic_worlds(self):
        root, ranges = fixture(2)
        next(a for a in legal_actions(root) if a.action_id == "ALL_IN").apply(root)
        hero = root.current_player
        holding = next(iter(ranges.ranges[hero].weights))
        policies = {p: UniformLegalPolicy() for p in root.players}
        budget = LabelBudget(looks=(32, 128, 512), seed=71)
        first = generate(root, ranges, holding, policies, budget)
        second = generate(root, ranges, holding, policies, budget)
        self.assertEqual(first["paired_worlds"], second["paired_worlds"])
        exact = {}
        for action in legal_actions(root):
            value = 0.
            for deal in ranges.conditioned(hero, holding).enumerate():
                state = instantiate(root, deal.hands)
                action.apply(state)
                self.assertTrue(state.terminal)
                value += deal.probability * state.utility(hero) / root.blinds.big
            exact[action.action_id] = value
        for pair in first["estimate"]["pairs"]:
            a, b = pair["actions"]
            self.assertLessEqual(pair["interval_bb"][0], exact[a] - exact[b])
            self.assertGreaterEqual(pair["interval_bb"][1], exact[a] - exact[b])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "label.json"
            save(path, first)
            self.assertEqual(json.loads(path.read_text())["paired_worlds"], first["paired_worlds"])

    def test_budget_cutoff_keeps_only_scheduled_confidence(self):
        root, ranges = fixture(2)
        policies = {p: UniformLegalPolicy() for p in root.players}
        holding = next(iter(ranges.ranges[root.current_player].weights))
        result = generate(root, ranges, holding, policies, LabelBudget(looks=(16, 64), max_nodes=1))
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["work"]["completed_worlds"], 0)
        self.assertEqual(result["stop_reason"], "node_budget")
        with self.assertRaisesRegex(ValueError, "unresolved"):
            response_target(result)
        cancelled = generate(root, ranges, holding, policies, LabelBudget(), cancelled=lambda: True)
        self.assertEqual(cancelled["stop_reason"], "cancelled")

    def test_epsilon_stop_avoids_unnecessary_worlds(self):
        root, ranges = fixture(2)
        policies = {p: UniformLegalPolicy() for p in root.players}
        holding = next(iter(ranges.ranges[root.current_player].weights))
        result = generate(root, ranges, holding, policies, LabelBudget(looks=(16, 64), tolerance_bb=root.total_chips))
        self.assertEqual(result["work"]["completed_worlds"], 16)
        self.assertEqual(result["stop_reason"], "precision_reached")
        self.assertEqual(response_target(result)["semantics"], "fixed-policy root action values")

    def test_family_manifest_and_generated_roots(self):
        with tempfile.TemporaryDirectory() as d:
            plan = manifest(Path(d) / "manifest.json", 100)
            self.assertEqual(plan["count"], 4000)
            owners = {}
            for row in plan["roots"]:
                owners.setdefault(row["family"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in owners.values()))
        roots = list(curriculum(1))
        self.assertEqual(len(roots), 40)
        self.assertEqual(len({fingerprint(r) for r in roots}), 40)
        for item in roots:
            self.assertFalse(item.root.terminal)
            self.assertTrue(legal_actions(item.root))
            family = next(f for f in FAMILIES if f.name == item.family)
            if item.root.street != "PREFLOP":
                self.assertEqual(board_texture(item.root.board), family.texture)
            deal = item.ranges.sample(random.Random(1))
            instantiate(item.root, deal).assert_invariants()

    def test_dataset_resume_and_contract_rejection(self):
        from hybrid.label_dataset import build
        with tempfile.TemporaryDirectory() as d:
            plan, output = Path(d) / "plan.json", Path(d) / "data"
            manifest(plan, 1)
            budget = LabelBudget(looks=(4,), max_nodes=10000)
            first = build(plan, output, budget, max_new_labels=1)
            second = build(plan, output, budget, max_new_labels=1)
            self.assertEqual(first["new_labels"], 1)
            self.assertEqual(second["existing_labels"], 1)
            self.assertEqual(second["new_labels"], 1)
            with self.assertRaisesRegex(ValueError, "contract changed"):
                build(plan, output, replace(budget, seed=1))
            with self.assertRaisesRegex(ValueError, "storage limit"):
                build(plan, output, budget, max_storage_bytes=1)

    def test_amortization_requires_quality_and_savings(self):
        self.assertEqual(break_even(10, 1, 100, quality_pass=True)["queries_to_amortize"], 12)
        self.assertIsNone(break_even(10, 1, 100, quality_pass=False)["queries_to_amortize"])
        self.assertIsNone(break_even(1, 2, 1, quality_pass=True)["queries_to_amortize"])
