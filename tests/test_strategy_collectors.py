"""Enumerate actual traversal RNG branches against independent reach formulas."""
import unittest
import random
from dataclasses import asdict
from unittest.mock import patch

from actions import ACTION_IDS
from cfr_solver import Traversal, TraversalBudgetExceeded
from ml.deep_cfr import DeepCFRSolver, FrozenStrategy, TraversalTask, generate_samples
from features.neural import observe_neural
from scripts.strategy_collector_audit import exact_reference, small_root, varying_policy, enumerated_expectation


class StrategyCollectorTests(unittest.TestCase):
    def test_actual_collectors_match_exact_inclusion_expectations(self):
        for count in (2, 3):
            root = small_root(count)
            exact = exact_reference(root)
            for collector in ("uniform_importance", "opponent_nodes", "partial_enumeration"):
                actual = enumerated_expectation(root, collector)
                expected = exact["expected"][collector]
                self.assertEqual(actual.keys(), expected.keys())
                for key, row in expected.items():
                    for a, e in zip(actual[key]["action_mass"], row["action_mass"]):
                        self.assertAlmostEqual(a, e, places=11)
                bias = max(abs(a / expected[key]["mass"] - e / reference["mass"])
                           for key, reference in exact["reference"].items()
                           for a, e in zip(expected[key]["action_mass"], reference["action_mass"]))
                if count == 3 and collector == "opponent_nodes":
                    self.assertGreater(bias, .1)
                else:
                    self.assertLess(bias, 1e-12)

    def test_partial_collector_both_orientations_with_zero_probability_actions(self):
        def sparse_policy(iteration):
            def policy(obs):
                legal = [i for i, legal in enumerate(obs.legal_mask) if legal]
                chosen = legal[(iteration + obs.hero + len(obs.history)) % len(legal)]
                return tuple(float(i == chosen) for i in range(len(ACTION_IDS)))
            return policy

        root = small_root(3)
        for factory in (varying_policy, sparse_policy):
            reference = exact_reference(root, factory)["reference"]
            for orientation in (0, 1):
                actual = enumerated_expectation(root, "partial_enumeration", factory, orientation)
                for key, expected in reference.items():
                    observed = actual.get(key, {"action_mass": [0.0] * len(ACTION_IDS)})
                    for a, e in zip(observed["action_mass"], expected["action_mass"]):
                        self.assertAlmostEqual(a, e, places=11)

    def test_three_player_generation_matches_two_halfweighted_orientations(self):
        solver = DeepCFRSolver((0, 1, 2), seed=7)
        snapshot = solver.snapshot()
        strategy = FrozenStrategy(snapshot)
        for player in solver.players:
            task = TraversalTask(0, player, small_root(3), 3, 1000, 64)
            result = generate_samples(snapshot, task)
            rng = random.Random(task.seed)
            regret_walk = Traversal(strategy, rng, task.max_nodes, task.max_depth, observer=observe_neural)
            regret_walk.regrets(task.root, player, lambda *_: None)
            expected = []
            nodes = regret_walk.nodes
            for opponent in solver.players:
                if opponent == player:
                    continue
                walk = Traversal(strategy, rng, task.max_nodes, task.max_depth, observer=observe_neural)
                walk.average_partial(task.root, player, opponent,
                                     lambda obs, target, weight: expected.append((obs, target, weight / 2)))
                nodes += walk.nodes
            self.assertTrue(expected)
            self.assertEqual([(sample.state, sample.target, sample.weight) for sample in result.strategies], expected)
            self.assertEqual(result.nodes, nodes)
            self.assertTrue(all(sample.player == player for sample in result.strategies))

    def test_second_orientation_failure_returns_no_partial_generation(self):
        solver = DeepCFRSolver((0, 1, 2), seed=7)
        snapshot = solver.snapshot()
        task = TraversalTask(0, 0, small_root(3), 3, 1000, 64)
        original_root = asdict(task.root)
        original = Traversal.average_partial
        orientations = []

        def fail_second(walk, state, player, opponent, sink, sample_reach=1.0, depth=0):
            if depth == 0:
                orientations.append(opponent)
                if len(orientations) == 2:
                    raise TraversalBudgetExceeded("Injected second-orientation failure")
            return original(walk, state, player, opponent, sink, sample_reach, depth)

        with patch.object(Traversal, "average_partial", fail_second):
            with self.assertRaisesRegex(TraversalBudgetExceeded, "stage=strategy_enumerate_2.*no partial samples returned"):
                generate_samples(snapshot, task)
        self.assertEqual(orientations, [1, 2])
        self.assertEqual(asdict(task.root), original_root)
        self.assertEqual(solver.version, 0)
        self.assertEqual(solver.strategy_memory.seen, 0)
        self.assertTrue(generate_samples(snapshot, task).strategies)

    def test_hu_generation_records_only_opponents_with_unit_reach_weights(self):
        solver = DeepCFRSolver((0, 1), seed=7)
        for player in solver.players:
            result = generate_samples(solver.snapshot(), TraversalTask(0, player, small_root(2), 3, 1000, 64))
            self.assertTrue(result.strategies)
            self.assertTrue(all(s.player != player and s.weight == 1 for s in result.strategies))
            self.assertTrue(all(s.player == player for s in result.advantages))
            self.assertTrue(all(len(s.target) == len(ACTION_IDS) for s in result.strategies))


if __name__ == "__main__":
    unittest.main()
