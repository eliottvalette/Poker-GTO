"""Enumerate actual traversal RNG branches against independent reach formulas."""
import unittest

from actions import ACTION_IDS
from cfr_solver import Traversal
from ml.deep_cfr import DeepCFRSolver, TraversalTask, generate_samples
from scripts.strategy_collector_audit import add_mass, exact_reference, small_root, varying_policy


class BranchRequired(Exception):
    def __init__(self, probabilities):
        self.probabilities = probabilities


class BranchReplay:
    def __init__(self, prefix):
        self.prefix = iter(prefix)
        self.probabilities = ()

    def pick(self, probabilities):
        index = next(self.prefix, None)
        if index is None:
            raise BranchRequired(probabilities)
        return index

    def random(self):
        index = self.pick(self.probabilities)
        return sum(self.probabilities[:index]) + self.probabilities[index] / 2

    def choice(self, actions):
        return actions[self.pick(tuple(1 / len(actions) for _ in actions))]


def enumerated_expectation(root, collector):
    rows = {}
    for iteration in (1, 2, 3):
        base_policy = varying_policy(iteration)
        for player in root.players:
            pending = [((), 1.0)]
            paths = 0
            while pending:
                paths += 1
                if paths > 10000:
                    raise RuntimeError("Exact RNG enumeration exceeded 10000 paths")
                prefix, probability = pending.pop()
                rng = BranchReplay(prefix)

                def policy(obs):
                    rng.probabilities = base_policy(obs)
                    return rng.probabilities

                samples = []
                sink = lambda obs, target, weight: samples.append((obs.key(), target, weight))
                walk = Traversal(policy, rng)
                try:
                    if collector == "uniform_importance":
                        walk.average(root, player, sink)
                    else:
                        walk.regrets(root, player, lambda *_: None, strategy_sink=sink)
                except BranchRequired as branch:
                    pending.extend((prefix + (i,), probability * p)
                                   for i, p in enumerate(branch.probabilities) if p > 0)
                    continue
                for key, target, weight in samples:
                    add_mass(rows, key, target, probability * iteration * weight)
    return rows


class StrategyCollectorTests(unittest.TestCase):
    def test_actual_collectors_match_exact_inclusion_expectations(self):
        for count in (2, 3):
            root = small_root(count)
            exact = exact_reference(root)
            for collector in ("uniform_importance", "opponent_nodes"):
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
