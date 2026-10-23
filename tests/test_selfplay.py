"""Fresh self-play ownership and continuous-run restart contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.selfplay import configuration, initialize, publish_status
from training.vps import run_continuous
from training.runner import TrainingRunner
from test_publication import MemoryStorage


class SelfPlayTests(unittest.TestCase):
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
