"""Fresh self-play ownership and continuous-run restart contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.selfplay import configuration, initialize, migrate_v4, publish_status
from training.vps import run_continuous
from training.runner import TrainingRunner
from test_publication import MemoryStorage


class SelfPlayTests(unittest.TestCase):
    def test_v4_migration_preserves_complete_learning_state_for_both_tracks(self):
        from copy import deepcopy
        import hashlib
        import torch
        from training.checkpoint import read_checkpoint
        from deploy.vps.verify_v4 import verify

        def assert_tree_equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, dict):
                self.assertEqual(set(left), set(right))
                for key in left:
                    assert_tree_equal(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(type(left), type(right))
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    assert_tree_equal(a, b)
            else:
                self.assertEqual(left, right)

        for track in ('hu', '3max'):
            with self.subTest(track=track), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / 'selfplay-v3'
                destination = Path(temporary) / 'selfplay-v4'
                before = configuration(track, source)
                before[track]['traversals_per_player'] = 64
                before['fit_schedule'].update(advantage_max_updates=2048, average_max_updates=2048, average_every=1)
                with patch('training.selfplay.configuration', return_value=before):
                    initialize(source, track)
                runner = TrainingRunner.load_checkpoint(source / 'checkpoint.pt')
                runner.run_iteration()
                runner.save_checkpoint(source / 'checkpoint.pt')
                original = read_checkpoint(source / 'checkpoint.pt')
                checksum = hashlib.sha256((source / 'checkpoint.pt').read_bytes()).hexdigest()
                contract = migrate_v4(destination, track, source_directory=source)
                migrated = read_checkpoint(destination / 'checkpoint.pt')
                self.assertEqual(checksum, hashlib.sha256((source / 'checkpoint.pt').read_bytes()).hexdigest())
                for key in set(original) - {'config', 'config_hash', 'metadata'}:
                    assert_tree_equal(original[key], migrated[key])
                self.assertNotEqual(original['metadata']['run_id'], contract['run_id'])
                self.assertEqual(contract['migration']['fixed_hyperparameters_until_iteration'], original['iteration'] + 500)
                self.assertEqual(migrate_v4(destination, track, source_directory=source), contract)
                restored = TrainingRunner.load_checkpoint(destination / 'checkpoint.pt')
                self.assertEqual(restored.iteration, original['iteration'])
                self.assertEqual(restored.config[track]['traversals_per_player'], 256)
                self.assertEqual(restored.config['fit_schedule']['average_every'], 3)
                verification = verify(Path(temporary), track)
                self.assertTrue(verification['learning_state_equal'])
                self.assertTrue(verification['full_runner_reload'])
                invalid = deepcopy(configuration(track, destination))
                invalid['learning_rate'] *= 2
                with patch('training.selfplay.configuration', return_value=invalid):
                    with self.assertRaisesRegex(ValueError, 'configuration changed'):
                        migrate_v4(destination, track, source_directory=source)

    def test_migration_fingerprint_covers_unordered_state(self):
        from deploy.vps.verify_v4 import digest
        self.assertEqual(digest({2, 1}), digest({1, 2}))
        self.assertNotEqual(digest({1, 2}), digest({1, 3}))
        self.assertNotEqual(digest({1, 2}), digest([1, 2]))

    def test_average_three_iteration_cadence_survives_resume(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'run'
            config = configuration('hu', directory)
            config['hu']['traversals_per_player'] = 4
            config['fit_schedule'].update(advantage_max_updates=2, average_max_updates=2)
            config.update(checkpoint_every=100, evaluation_every=100)
            with patch('training.selfplay.configuration', return_value=config):
                initialize(directory, 'hu')
            runner = TrainingRunner.load_checkpoint(directory / 'checkpoint.pt')
            for iteration in range(1, 5):
                previous = runner.solvers['hu'].average_model
                row = runner.run_iteration()['tracks']['hu']
                self.assertEqual(row['advantage']['updates_completed'], 2)
                if iteration in (2, 4):
                    self.assertTrue(row['average_policy']['skipped'])
                    for key, value in previous.state_dict().items():
                        self.assertTrue(torch.equal(value, runner.solvers['hu'].average_model.state_dict()[key]))
                    self.assertEqual(row['average_model_iteration'], 1 if iteration == 2 else 3)
                else:
                    self.assertEqual(row['average_policy']['updates_completed'], 2)
                if iteration == 2:
                    runner.save_checkpoint(directory / 'checkpoint.pt')
                    runner = TrainingRunner.load_checkpoint(directory / 'checkpoint.pt')

    def test_initialization_is_idempotent_and_rejects_foreign_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            first = initialize(root, 'hu')
            self.assertEqual(initialize(root, 'hu'), first)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                initialize(root, '3max')
            foreign = Path(temporary) / 'foreign'
            foreign.mkdir()
            (foreign / 'checkpoint.pt').write_text('legacy')
            with self.assertRaisesRegex(ValueError, 'Nonempty'):
                initialize(foreign, 'hu')

    def test_continuous_resume_exceeds_declared_batch_count_and_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            config = configuration('hu', root)
            config.update(outer_iterations=1, evaluation_every=100, checkpoint_every=1)
            config['hu']['traversals_per_player'] = 4
            with patch('training.selfplay.configuration', return_value=config):
                initialize(root, 'hu')
            original_iteration = TrainingRunner.run_iteration
            for expected in (1, 2):
                def complete_then_interrupt(runner, **kwargs):
                    if runner.iteration == expected:
                        raise KeyboardInterrupt
                    return original_iteration(runner, **kwargs)

                with patch('training.vps.export_live', return_value=root / 'export'), \
                     patch.object(TrainingRunner, 'run_iteration', autospec=True,
                                  side_effect=complete_then_interrupt):
                    run_continuous('hu', root)
                self.assertEqual(TrainingRunner.load_checkpoint(root / 'checkpoint.pt').iteration, expected)
            storage = MemoryStorage()
            publish_status(storage, root, 'hu')
            status = json.loads(storage.objects['selfplay/hu/status.json'])
            self.assertEqual(status['iteration'], 2)
            self.assertIsNone(status['iteration_limit'])
            self.assertEqual(status['engine'], 'continuous_selfplay_v1')
            self.assertNotIn('solver/hu/status.json', storage.objects)

    def test_fitting_budget_migration_retains_replay_and_changes_run_identity(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'selfplay-v2'
            target = Path(temporary) / 'selfplay-v3'
            before = configuration('hu', source)
            before['hu']['traversals_per_player'] = 4
            before.update(advantage_epochs=1, average_epochs=1)
            before['fit_schedule'].update(advantage_max_updates=8, average_max_updates=8)
            with patch('training.selfplay.configuration', return_value=before):
                contract = initialize(source, 'hu')
            runner = TrainingRunner.load_checkpoint(source / 'checkpoint.pt')
            runner.run_iteration()
            runner.save_checkpoint(source / 'checkpoint.pt')
            after = deepcopy(before)
            after.update(output_dir=str(target.resolve()), advantage_epochs=2, average_epochs=2)
            after['fit_schedule'].update(advantage_max_updates=16, average_max_updates=16)
            with patch('training.selfplay.configuration', return_value=after):
                migrated = initialize(target, 'hu', source_directory=source)
            restored = TrainingRunner.load_checkpoint(target / 'checkpoint.pt')
            self.assertEqual(restored.iteration, runner.iteration)
            self.assertNotEqual(migrated['run_id'], contract['run_id'])
            for kind in ('advantage_memory', 'strategy_memory'):
                self.assertEqual(getattr(restored.solvers['hu'], kind).samples,
                                 getattr(runner.solvers['hu'], kind).samples)
            self.assertEqual(restored.metadata['fitting_budget_migration']['source_run_id'], contract['run_id'])
            invalid = deepcopy(after)
            invalid['learning_rate'] *= 2
            invalid['output_dir'] = str(Path(temporary) / 'invalid')
            with patch('training.selfplay.configuration', return_value=invalid):
                with self.assertRaisesRegex(ValueError, 'cannot change'):
                    initialize(Path(temporary) / 'invalid', 'hu', source_directory=source)
