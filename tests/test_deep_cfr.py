import random
import gzip
import json
from dataclasses import replace
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
import torch
from actions import ACTION_IDS
from cfr_solver import regret_matching
from infoset import observe
from ml.deep_cfr import DeepCFRSolver, ModelSnapshot, TraversalTask, NeuralAveragePolicy, generate_samples
from ml.memory import ReservoirMemory, TrainingSample, sample_bytes
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch
from ml.train import evaluate_loss, loss_for
from scripts.parallel_cfr import collect_samples
from test_solver import river


class DeepCFRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shapes_masking_and_invalid_inputs(self):
        obs = observe(river())
        batch = encode_batch([obs, obs])
        self.assertEqual(tuple(AdvantageNetwork()(batch).shape), (2, len(ACTION_IDS)))
        p = AveragePolicyNetwork()(batch)
        self.assertTrue(torch.allclose(p.sum(1), torch.ones(2)))
        self.assertEqual(float(p[~batch["mask"]].sum().detach()), 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            encode_batch([])

    def test_reservoir_bounds_roundtrip_and_reproducibility(self):
        o = observe(river())
        a = ReservoirMemory(4, 1, "strategy", "hand_chip_delta")
        b = ReservoirMemory(4, 1, "strategy", "hand_chip_delta")
        target = regret_matching([0.0] * len(ACTION_IDS), o.legal_mask)
        for i in range(1, 31):
            s = TrainingSample(i, o.hero, o, target, 1, "strategy", i - 1)
            a.add(s)
            b.add(s)
        self.assertEqual(a.samples, b.samples)
        self.assertEqual(len(a.samples), 4)
        self.assertEqual(a.seen, 30)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.gz"
            a.save(path)
            c = ReservoirMemory.load(path)
            self.assertEqual(a.samples, c.samples)
            self.assertEqual(a.rng.getstate(), c.rng.getstate())
            self.assertEqual(a.byte_budget, c.byte_budget)
            with gzip.open(path, "rt") as file:
                raw = json.load(file)
            raw["seen"] = 2
            with gzip.open(path, "wt") as file:
                json.dump(raw, file)
            with self.assertRaisesRegex(ValueError, "reservoir counts"):
                ReservoirMemory.load(path)
            raw["version"] = 2
            with gzip.open(path, "wt") as file:
                json.dump(raw, file)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                ReservoirMemory.load(path)

    def test_parallel_frozen_generation_matches_single_worker(self):
        snapshot = ModelSnapshot(0, "hand_chip_delta", {}, True)
        tasks = [TraversalTask(i, i % 2, river(), i + 17, 1000, 100) for i in range(4)]
        sequential = collect_samples(snapshot, tasks, 1)
        parallel = collect_samples(snapshot, tasks, 2)
        self.assertEqual(sequential, parallel)
        self.assertTrue(all(s.model_version == 0 for r in parallel for s in r.advantages + r.strategies))

    def test_reservoir_budget_failure_is_atomic(self):
        obs = observe(river())
        target = regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)
        sample = TrainingSample(1, obs.hero, obs, target, 1, "strategy", 0)
        budget = sample_bytes(sample) + 1000
        memory = ReservoirMemory(1, 1, "strategy", "hand_chip_delta", byte_budget=budget)
        memory.add(sample)
        before = (list(memory.samples), memory.seen, memory.used_bytes, memory.rng.getstate())
        large = replace(sample, state=replace(obs, recall=obs.recall[:-1] + ',"' + "x" * budget + '"]'))
        with self.assertRaisesRegex(MemoryError, "budget"):
            memory.add(large)
        self.assertEqual(before, (memory.samples, memory.seen, memory.used_bytes, memory.rng.getstate()))
        append_memory = ReservoirMemory(2, 1, "strategy", "hand_chip_delta", byte_budget=budget)
        append_memory.add(sample)
        with self.assertRaisesRegex(MemoryError, "budget"):
            append_memory.add(sample)
        self.assertEqual(append_memory.seen, 1)
        self.assertEqual(append_memory.samples, [sample])
        replacement_memory = ReservoirMemory(2, 0, "strategy", "hand_chip_delta",
                                             byte_budget=sample_bytes(sample) * 2 + 1000)
        replacement_memory.add(sample)
        replacement_memory.add(sample)
        larger = replace(sample, state=replace(obs, recall=obs.recall[:-1] + ',"' + "x" * 2000 + '"]'))
        self.assertLess(sample_bytes(larger), replacement_memory.byte_budget)
        before_rng = replacement_memory.rng.getstate()
        with self.assertRaisesRegex(MemoryError, "Reservoir update"):
            replacement_memory.add(larger)
        self.assertEqual(replacement_memory.rng.getstate(), before_rng)
        self.assertEqual(replacement_memory.seen, 2)
        self.assertEqual(replacement_memory.samples, [sample, sample])

    def test_generation_budget_and_iteration_failure_are_atomic(self):
        solver = DeepCFRSolver((0, 1), "hand_chip_delta", seed=2)
        before_rng = solver.rng.getstate()
        task = TraversalTask(0, 0, river(), 17, 1000, 100, 1)
        with self.assertRaisesRegex(MemoryError, "no partial"):
            generate_samples(solver.snapshot(), task)
        with self.assertRaisesRegex(MemoryError, "no partial"):
            solver.run_iteration(lambda _: river(), max_nodes=1000, sample_byte_budget=1)
        self.assertEqual(solver.version, 0)
        self.assertEqual(solver.rng.getstate(), before_rng)
        self.assertEqual(solver.strategy_memory.seen, 0)
        self.assertTrue(all(m.seen == 0 for m in solver.advantage_memory.values()))
        self.assertEqual(solver.metrics, [])
        with self.assertRaisesRegex(MemoryError, "budget"):
            solver.run_iteration(lambda _: river(), max_nodes=1000, generation_byte_budget=1)
        self.assertEqual(solver.version, 0)
        self.assertEqual(solver.rng.getstate(), before_rng)
        self.assertEqual(solver.strategy_memory.seen, 0)
        with patch("ml.deep_cfr.fit", side_effect=ValueError("Injected training failure")):
            with self.assertRaisesRegex(ValueError, "Injected training failure"):
                solver.run_iteration(lambda _: river(), traversals_per_player=2, max_nodes=1000)
        self.assertEqual(solver.rng.getstate(), before_rng)
        self.assertEqual(solver.version, 0)
        self.assertEqual(solver.strategy_memory.seen, 0)

    def test_global_weight_normalization_and_bounded_metrics(self):
        obs = observe(river())
        mask = obs.legal_mask
        first = next(i for i, legal in enumerate(mask) if legal)
        sample_a = TrainingSample(1, obs.hero, obs, tuple(1.0 if i == first else 0.0 for i in range(len(ACTION_IDS))),
                                  1.0, "strategy", 0)
        sample_b = replace(sample_a, iteration=3, model_version=2,
                           target=regret_matching([0.0] * len(ACTION_IDS), mask), weight=2.0)
        samples = [sample_a, sample_b]
        model = AveragePolicyNetwork().eval()
        full = float(loss_for(model, samples).detach())
        expected = (float(loss_for(model, [sample_a], 3.5).detach())
                    + float(loss_for(model, [sample_b], 3.5).detach())) / 2
        self.assertAlmostEqual(full, expected, places=6)
        self.assertAlmostEqual(full, evaluate_loss(model, samples, 1), places=6)

    def test_trained_snapshot_generation_matches_parallel_workers(self):
        solver = DeepCFRSolver((0, 1), "hand_chip_delta", seed=2, advantage_capacity=100, strategy_capacity=100)
        solver.run_iteration(lambda _: river(), traversals_per_player=2, max_nodes=1000)
        tasks = [TraversalTask(i, i % 2, river(), i + 17, 1000, 100) for i in range(4)]
        self.assertEqual(collect_samples(solver.snapshot(), tasks, 1), collect_samples(solver.snapshot(), tasks, 2))

    def test_two_outer_iterations_and_average_export(self):
        solver = DeepCFRSolver((0, 1), "hand_chip_delta", seed=2, advantage_capacity=100, strategy_capacity=100)
        for _ in range(2):
            metrics = solver.run_iteration(lambda _: river(), traversals_per_player=2, max_nodes=1000)
            self.assertGreater(metrics["advantage_samples"], 0)
            self.assertTrue(all(torch.isfinite(torch.tensor(v["heldout_loss"])) for v in metrics.values() if isinstance(v, dict)))
        self.assertEqual(solver.version, 2)
        frozen = solver.snapshot()
        for model in solver.advantage_models.values():
            with torch.no_grad():
                next(model.parameters()).add_(1)
        self.assertFalse(torch.equal(next(iter(frozen.advantage_weights[0].values())), next(solver.advantage_models[0].parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "average.pt"
            solver.export_average(path)
            policy = NeuralAveragePolicy(path)
            self.assertAlmostEqual(sum(policy.query(observe(river()))), 1, places=6)
            from poker_game_expresso import HandState
            three = HandState.start({0: 25, 1: 25, 2: 25}, 0, random.Random(2))
            with self.assertRaisesRegex(KeyError, "no training coverage"):
                policy.query(observe(three))
            torch.save({"version": 1}, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                NeuralAveragePolicy(path)


if __name__ == "__main__":
    unittest.main()
