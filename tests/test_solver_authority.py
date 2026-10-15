"""Computed game state and optional student resume must remain independent."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hybrid.distillation import SolverStudents, conditional_weights
from hybrid.knowledge import calculate, supervision, cached_policy
from infoset import observe
from hybrid.distillation import ResidualAdvantage
from ml.health import numerical_health
from hybrid.policy_source import UniformLegalPolicy
from hybrid.ranges import HandRange, JointRanges
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget
from training.solver_runner import SolverRunner, SolverTask
from test_hybrid_river import fixture


from hybrid.solver_experiments import bounded_roots as curriculum


class SolverAuthorityTests(unittest.TestCase):
    def test_observation_key_keeps_exact_transport_after_copy_optimization(self):
        for count in (2, 3):
            for root, _ in curriculum(count, "test"):
                obs = observe(root)
                self.assertEqual(obs.key(), json.dumps(asdict(obs), sort_keys=True, separators=(",", ":"), allow_nan=False))

    def test_runner_resumes_absolute_target_and_bounds_storage(self):
        root, ranges = fixture(2)
        task = SolverTask("hu-river", root, ranges, "train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner = SolverRunner(path, [task], ComputeBudget(iterations=3))
            runner.refine_to(7)
            saved = (path / "hu-river.json").read_bytes()
            repeated = runner.refine_to(7)
            self.assertEqual(repeated["work"], [])
            self.assertTrue(repeated["complete"])
            self.assertEqual((path / "hu-river.json").read_bytes(), saved)
            limited = SolverRunner(path, [task], ComputeBudget(iterations=3), max_storage_bytes=1)
            with self.assertRaisesRegex(ValueError, "storage budget"):
                limited.refine_to(8)
            self.assertEqual((path / "hu-river.json").read_bytes(), saved)
            with self.assertRaisesRegex(ValueError, "contract changed"):
                SolverRunner(path, [task], ComputeBudget(iterations=3, seed=42))

    def test_explicit_regret_resume_matches_uninterrupted(self):
        for count in (2, 3):
            root, ranges = fixture(count)
            first = RiverSolver().solve(root, ranges, ComputeBudget(iterations=7))
            resumed = RiverSolver(resume=json.loads(json.dumps(first.checkpoint))).solve(root, ranges, ComputeBudget(iterations=13))
            direct = RiverSolver().solve(root, ranges, ComputeBudget(iterations=20))
            self.assertEqual(resumed.checkpoint, direct.checkpoint)
            self.assertEqual(resumed.action_ev_by_hand, direct.action_ev_by_hand)
            changed = JointRanges({p: HandRange({h: 1 for h in r.weights}) for p, r in ranges.ranges.items()}, tuple(root.board))
            with self.assertRaisesRegex(ValueError, "identical public-range game"):
                RiverSolver(resume=first.checkpoint).solve(root, changed, ComputeBudget(iterations=1))

    def test_chance_resume_and_artifact_contract(self):
        for count in (2, 3):
            with tempfile.TemporaryDirectory() as directory:
                root, ranges = next(curriculum(count, "train"))
                budget = ComputeBudget(iterations=3, samples=16, max_depth=1, seed=92)
                path = Path(directory) / "game.json"
                first = calculate(path, root, ranges, budget, split="train")
                resumed = calculate(path, root, ranges, budget, split="train")
                direct = RiverSolver().solve_chance(root, ranges, replace(budget, iterations=6), continuation=UniformLegalPolicy())
                self.assertEqual(resumed.checkpoint, direct.checkpoint)
                self.assertEqual(first.work.iterations, 3)
                _, samples = supervision(path, "strategy")
                self.assertTrue(samples)
                self.assertTrue(all(s.traversal_mode == "finite_game_cfr" for s in samples))
                policy = cached_policy(path, root, ranges)
                self.assertEqual(policy.probabilities(observe(root)), resumed.policy.probabilities(observe(root)))
                with self.assertRaisesRegex(ValueError, "contract changed"):
                    calculate(path, root, ranges, replace(budget, seed=93), split="train")

    def test_student_checkpoint_resumes_all_eight_models_exactly(self):
        torch.set_num_threads(1)
        for count in (2, 3):
            with tempfile.TemporaryDirectory() as directory:
                paths = []
                for split in ("train", "validation", "test"):
                    for root, ranges in curriculum(count, split):
                        path = Path(directory) / f"{split}-{root.street}.json"
                        calculate(path, root, ranges, ComputeBudget(iterations=2, samples=16, max_depth=1), split=split)
                        paths.append(path)
                student = SolverStudents(count)
                with self.assertRaisesRegex(MemoryError, "encoding requires"):
                    student.fit(paths, updates=2, batch_size=8, max_encoding_bytes=1)
                leaked = json.loads(paths[0].read_text())
                leaked["contract"]["split"] = "test"
                leak_path = Path(directory) / "leaked-root.json"
                leak_path.write_text(json.dumps(leaked))
                with self.assertRaisesRegex(ValueError, "root leakage"):
                    student.fit([*paths, leak_path], updates=2, batch_size=8)
                self.assertTrue(all(count == 0 for count in student.updates.values()))
                student.fit(paths, updates=2, batch_size=8)
                path = Path(directory) / "student.pt"
                student.save(path)
                resumed = SolverStudents.load(path)
                a = student.fit(paths, updates=2, batch_size=8)
                b = resumed.fit(paths, updates=2, batch_size=8)
                self.assertEqual(a, b)
                for kind in student.models:
                    for key, value in student.models[kind].state_dict().items():
                        self.assertTrue(torch.equal(value, resumed.models[kind].state_dict()[key]))
                _, records = supervision(paths[0], "strategy")
                weighted = conditional_weights(records)
                self.assertEqual([s.target for s in weighted], [s.target for s in records])

    def test_residual_direct_path_receives_gradients_even_with_zero_hidden_path(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root, ranges = next(curriculum(3, "train"))
            path = Path(directory) / "game.json"
            calculate(path, root, ranges, ComputeBudget(iterations=2, samples=16, max_depth=1), split="train")
            _, rows = supervision(path, "advantage")
            model = ResidualAdvantage()
            with torch.no_grad():
                model.head[-1].weight.zero_()
                model.head[-1].bias.zero_()
            health = numerical_health(model, rows)
            self.assertGreater(health["encoder_gradient_norm"], 0.)
            self.assertGreater(health["gradient_norms"]["direct.weight"], 0.)


if __name__ == "__main__":
    unittest.main()
