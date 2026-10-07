"""Shared regret approximation must be independent of simulator seat labels."""
from collections import Counter
from dataclasses import replace
from itertools import permutations
import random
import unittest
from unittest.mock import patch
import torch
from cfr_solver import TraversalBudgetExceeded
from features.neural import neural_observation
from infoset import observe
from ml.deep_cfr import DeepCFRSolver, TraversalTask, generate_samples
from ml.model import AdvantageNetwork, AveragePolicyNetwork, encode_batch
from poker_game_expresso import HandState


def renamed_ring(hand, mapping, rotation):
    changed = hand.clone()
    seats = list(hand.players)
    seats = seats[rotation:] + seats[:rotation]
    changed.players = {mapping[i]: replace(hand.players[i], player_id=mapping[i]) for i in seats}
    changed.initial_stacks = {mapping[i]: hand.initial_stacks[i] for i in seats}
    changed.button = mapping[hand.button]
    changed.current_player = mapping[hand.current_player]
    changed.pending = {mapping[i] for i in hand.pending}
    changed.awards = {mapping[i]: v for i, v in hand.awards.items()}
    changed.history = [replace(event, player_id=mapping[event.player_id]) for event in hand.history]
    changed.assert_invariants()
    return changed


class SharedAdvantageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_capacities_preserve_total_replay(self):
        for count, total in ((2, 20000), (3, 30000)):
            solver = DeepCFRSolver(tuple(range(count)))
            self.assertEqual(solver.advantage_memory.capacity, total)
            self.assertIsNone(solver.advantage_model)
            self.assertFalse(hasattr(solver, 'advantage_models'))
            snapshot = solver.snapshot()
            self.assertEqual(snapshot.player_count, count)
            self.assertIsNone(snapshot.advantage_weights)

    def test_every_consumed_tensor_and_prediction_is_seat_invariant(self):
        for count in (2, 3):
            hand = HandState.start({i: 75 / count + (i - (count - 1) / 2) * 4 for i in range(count)}, 1, random.Random(7))
            cases = [hand.clone()]
            if count == 3:
                folded = hand.clone()
                folded.act('FOLD')
                cases.append(folded)
            for street in ('FLOP', 'TURN', 'RIVER'):
                while hand.street != street:
                    hand.act('CALL' if hand.to_call() else 'CHECK')
                cases.append(hand.clone())
            for original in cases:
                obs = observe(original)
                variants = [obs]
                for permutation in permutations(range(count)):
                    mapping = dict(enumerate(permutation))
                    for rotation in range(count):
                        variants.append(observe(renamed_ring(original, mapping, rotation)))
                batch = encode_batch(variants)
                for name, tensor in batch.items():
                    for row in tensor:
                        self.assertTrue(torch.equal(row, tensor[0]), f'{count}/{original.street}/{name}')
                with torch.no_grad():
                    for model in (AdvantageNetwork(), AveragePolicyNetwork()):
                        outputs = model(batch)
                        torch.testing.assert_close(outputs, outputs[:1].expand_as(outputs), rtol=1e-6, atol=1e-6)

    def test_fits_once_on_pooled_samples_from_every_traverser(self):
        solver = DeepCFRSolver((0, 1), seed=2, advantage_capacity=200, strategy_capacity=200)
        def river(_):
            hand = HandState.start({0: 2, 1: 2}, 0, random.Random(3))
            while hand.street != 'RIVER':
                hand.act('CALL' if hand.to_call() else 'CHECK')
            return hand
        from ml.deep_cfr import fit
        with patch('ml.deep_cfr.fit', wraps=fit) as fits:
            metric = solver.run_iteration(river, traversals_per_player=3, max_nodes=1000)
        self.assertEqual(fits.call_count, 2)
        advantage_call = fits.call_args_list[0]
        self.assertIsInstance(advantage_call.args[0], AdvantageNetwork)
        self.assertEqual(set(s.player for s in advantage_call.args[1]), {0, 1})
        self.assertEqual(solver.advantage_memory.seen, metric['advantage_samples'])
        self.assertEqual(Counter(s.player for s in solver.advantage_memory.samples),
                         Counter({int(p): n for p, n in metric['advantage_retained_by_player'].items()}))
        frozen = solver.snapshot()
        with torch.no_grad():
            next(solver.advantage_model.parameters()).add_(1)
        self.assertFalse(torch.equal(next(iter(frozen.advantage_weights.values())), next(solver.advantage_model.parameters())))
        self.assertIsNotNone(solver.average_model)

    def test_shared_worker_has_identical_regrets_under_seat_renaming(self):
        hand = HandState.start({0: 2, 1: 2}, 0, random.Random(3))
        while hand.street != 'RIVER':
            hand.act('CALL' if hand.to_call() else 'CHECK')
        solver = DeepCFRSolver((0, 1))
        solver.version = 1
        solver.advantage_model = AdvantageNetwork().eval()
        changed = renamed_ring(hand, {0: 9, 1: 7}, 1)
        first = generate_samples(solver.snapshot(), TraversalTask(0, hand.current_player, hand, 17, 1000, 64))
        second = generate_samples(solver.snapshot(), TraversalTask(0, changed.current_player, changed, 17, 1000, 64))
        self.assertEqual(first.value, second.value)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(len(first.advantages), len(second.advantages))
        self.assertEqual(len(first.strategies), len(second.strategies))
        for left, right in zip(first.advantages + first.strategies, second.advantages + second.strategies):
            self.assertEqual(left.target, right.target)
            self.assertEqual(left.weight, right.weight)
            self.assertEqual(left.state.numeric_data, right.state.numeric_data)
            self.assertEqual(left.state.history_data, right.state.history_data)

    def test_foreign_track_snapshot_is_rejected(self):
        solver = DeepCFRSolver((0, 1, 2))
        hand = HandState.start({0: 25, 1: 50}, 0, random.Random(2))
        with self.assertRaisesRegex(ValueError, 'track mismatch'):
            generate_samples(solver.snapshot(), TraversalTask(0, 0, hand, 17, 1000, 64))

    def test_checkpoint_rejects_retired_seat_layout_and_restores_pooled_rng(self):
        import tempfile
        from pathlib import Path
        from training.config import load_config
        from training.runner import TrainingRunner
        from training.checkpoint import read_checkpoint, write_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            config = load_config('configs/deep_cfr_pilot.json')
            config.update(output_dir=directory, workers=1, advantage_epochs=1, average_epochs=1,
                          checkpoint_every=1, evaluation_every=100)
            for name in ('3max', 'hu'):
                config[name]['traversals_per_player'] = 2
                config[name]['advantage_capacity'] = 1000
            runner = TrainingRunner(config)
            runner.run_iteration()
            path = Path(directory) / 'checkpoints/iteration_000001.pt'
            restored = TrainingRunner.load_checkpoint(path)
            for name in runner.solvers:
                left, right = runner.solvers[name], restored.solvers[name]
                self.assertEqual(left.advantage_memory.samples, right.advantage_memory.samples)
                self.assertEqual(left.advantage_memory.rng.getstate(), right.advantage_memory.rng.getstate())
                self.assertEqual(left.advantage_memory.used_bytes, right.advantage_memory.used_bytes)
                for key, weight in left.advantage_model.state_dict().items():
                    self.assertTrue(torch.equal(weight, right.advantage_model.state_dict()[key]))
            raw = read_checkpoint(path)
            raw['contract']['checkpoint'] = 1
            raw['contract']['advantage_layout'] = 'per_seat'
            write_checkpoint(path, raw)
            with self.assertRaisesRegex(ValueError, 'Incompatible checkpoint schema'):
                TrainingRunner.load_checkpoint(path)

    def test_checkpoint_does_not_ignore_models_at_iteration_zero_or_wrong_replay_kind(self):
        import tempfile
        from pathlib import Path
        from training.config import load_config
        from training.runner import TrainingRunner
        from training.checkpoint import read_checkpoint, write_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            config = load_config('configs/deep_cfr_pilot.json')
            config['output_dir'] = directory
            runner = TrainingRunner(config)
            path = Path(directory) / 'initial.pt'
            runner.save_checkpoint(path)
            raw = read_checkpoint(path)
            raw['tracks']['3max']['advantage_weights'] = AdvantageNetwork().state_dict()
            write_checkpoint(path, raw)
            with self.assertRaisesRegex(ValueError, 'Uninitialized checkpoint contains trained state'):
                TrainingRunner.load_checkpoint(path)
            runner.save_checkpoint(path)
            raw = read_checkpoint(path)
            memory = raw['tracks']['3max']['advantage_memory']
            memory.update(kind='strategy', capacity=config['strategy_capacity'], byte_budget=config['memory_byte_budget'])
            write_checkpoint(path, raw)
            with self.assertRaisesRegex(ValueError, 'replay slot kind mismatch'):
                TrainingRunner.load_checkpoint(path)
