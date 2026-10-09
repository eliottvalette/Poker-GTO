"""Value labels are independently checkable and rejected models cannot drive search."""
from pathlib import Path
import tempfile
import unittest
from evaluation import subgame_best_response
from hybrid.continuation import exact_river_labels, LearnedContinuation
from hybrid.experiments import small_game
from hybrid.learning import Learner
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget, instantiate


class ContinuationTests(unittest.TestCase):
    def test_private_indexed_labels_match_independent_policy_evaluation(self):
        root, ranges = small_game(2)
        budget = ComputeBudget(iterations=30)
        solution = RiverSolver().solve(root, ranges, budget)
        labels = exact_river_labels(root, ranges, budget, "reference")
        for holding, label in zip(solution.strategy_by_hand, labels):
            private = ranges.conditioned(root.current_player, holding)
            roots = [(d.probability, instantiate(root, d.hands)) for d in private.enumerate()]
            reference = subgame_best_response(roots, solution.policy.probabilities, root.current_player)
            self.assertAlmostEqual(label.target[0], reference["policy_value_bb"])
            self.assertIn("finite-iteration-error", label.provenance)
            self.assertEqual(label.compute_budget["street"], "RIVER")

    def test_rejected_value_model_has_durable_explicit_gate(self):
        labels = []
        for seed in range(3):
            root, ranges = small_game(2, seed=seed)
            labels.extend(exact_river_labels(root, ranges, ComputeBudget(iterations=5), str(seed)))
        learner = Learner(labels, "continuation", seed=11)
        learner.fit_to(3)
        learner.validate(0., 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.pt"
            learner.save(path)
            restored = Learner.load(path)
            self.assertFalse(restored.accepted)
            with self.assertRaisesRegex(ValueError, "acceptance"):
                LearnedContinuation(restored)
