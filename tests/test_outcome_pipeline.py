"""Estimator selection, replay ownership, worker reproducibility and atomic failure."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from actions import ACTION_IDS
from cfr_solver import TraversalBudgetExceeded, regret_matching
from infoset import observe
from ml.deep_cfr import DeepCFRSolver, ModelSnapshot, TraversalTask, generate_samples
from ml.memory import ReservoirMemory, TrainingSample
from ml.model import AdvantageNetwork
from scripts.parallel_cfr import collect_samples
from test_solver import river


class OutcomePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_outcome_generation_metadata_and_worker_determinism(self):
        snapshot = ModelSnapshot(0, "hand_chip_delta", {}, True)
        tasks = [TraversalTask(i, i % 2, river(), i + 17, 100, 100,
                               traversal_mode="outcome_sampling", epsilon=0.6) for i in range(4)]
        sequential = collect_samples(snapshot, tasks, 1)
        self.assertEqual(sequential, collect_samples(snapshot, tasks, 2))
        self.assertTrue(all(result.traversal_mode == "outcome_sampling" for result in sequential))
        self.assertTrue(all(sample.traversal_mode == "outcome_sampling"
                            for result in sequential for sample in result.advantages + result.strategies))
        self.assertTrue(all(sample.weight == 1 for result in sequential for sample in result.advantages))
        json.dumps([result.diagnostics for result in sequential], allow_nan=False)

    def test_replay_rejects_mixed_estimators_atomically_and_roundtrips_mode(self):
        observation = observe(river())
        target = regret_matching([0.0] * len(ACTION_IDS), observation.legal_mask)
        sample = TrainingSample(1, observation.hero, observation, target, 1, "strategy", 0, "outcome_sampling")
        memory = ReservoirMemory(4, 17, "strategy", "hand_chip_delta")
        memory.add(sample)
        before = (memory.samples[:], memory.seen, memory.rng.getstate(), memory.used_bytes)
        with self.assertRaisesRegex(ValueError, "do not mix estimators"):
            memory.add(replace(sample, traversal_mode="external_sampling"))
        self.assertEqual(before, (memory.samples, memory.seen, memory.rng.getstate(), memory.used_bytes))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.gz"
            memory.save(path)
            loaded = ReservoirMemory.load(path)
            self.assertEqual(loaded.traversal_mode, "outcome_sampling")
            self.assertEqual(loaded.samples, memory.samples)

    def test_structural_zero_diagnostics_remain_explicit_and_json_safe(self):
        model = AdvantageNetwork()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head[-1].bias.fill_(-1)
            model.head[-1].bias[ACTION_IDS.index("ALL_IN")] = 1
        weights = {player: model.state_dict() for player in (0, 1)}
        snapshot = ModelSnapshot(1, "hand_chip_delta", weights, False)
        task = TraversalTask(0, 0, river(), 18, 100, 100, traversal_mode="outcome_sampling")
        result = generate_samples(snapshot, task)
        self.assertGreater(sum(result.diagnostics["structural_zero_log_counts"].values()), 0)
        self.assertIn(None, result.diagnostics["log_regret_prefix_weights"])
        json.dumps(result.diagnostics, allow_nan=False)

    def test_budget_failure_preserves_solver_mode_rng_and_memories(self):
        solver = DeepCFRSolver((0, 1), "hand_chip_delta", seed=17)
        before = solver.rng.getstate()
        with self.assertRaises(TraversalBudgetExceeded):
            solver.run_iteration(lambda _: river(), traversal_mode="outcome_sampling", max_nodes=1)
        self.assertIsNone(solver.traversal_mode)
        self.assertEqual(solver.version, 0)
        self.assertEqual(solver.rng.getstate(), before)
        self.assertEqual(solver.strategy_memory.seen, 0)
        self.assertTrue(all(memory.traversal_mode is None for memory in solver.advantage_memory.values()))

    def test_outcome_tiny_training_records_estimator_and_rejects_mode_switch(self):
        solver = DeepCFRSolver((0, 1), "hand_chip_delta", seed=17)
        metric = solver.run_iteration(lambda _: river(), traversal_mode="outcome_sampling",
                                      traversals_per_player=3, max_nodes=100)
        self.assertEqual(metric["traversal_mode"], "outcome_sampling")
        self.assertEqual(metric["epsilon"], 0.6)
        self.assertEqual(solver.traversal_mode, "outcome_sampling")
        self.assertEqual(solver.strategy_memory.traversal_mode, "outcome_sampling")
        before = solver.rng.getstate()
        with self.assertRaisesRegex(ValueError, "mixed traversal mode"):
            solver.run_iteration(lambda _: river(), traversal_mode="external_sampling", max_nodes=100)
        self.assertEqual(solver.rng.getstate(), before)
        self.assertEqual(solver.version, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "average.pt"
            solver.export_average(path)
            metadata = torch.load(path, weights_only=True)["training_metadata"]
            self.assertEqual(metadata["traversal_mode"], "outcome_sampling")
            self.assertEqual(metadata["epsilon_by_iteration"], [0.6])

    def test_invalid_sampling_configuration_fails_before_generation(self):
        snapshot = ModelSnapshot(0, "hand_chip_delta", {}, True)
        task = TraversalTask(0, 0, river(), 17, 100, 100, traversal_mode="invalid")
        with self.assertRaisesRegex(ValueError, "traversal mode"):
            generate_samples(snapshot, task)
        task = replace(task, traversal_mode="outcome_sampling", epsilon=0)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            generate_samples(snapshot, task)


if __name__ == "__main__":
    unittest.main()
