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
            for expected in (1, 2):
                with patch('training.vps.export_live', return_value=root / 'export'):
                    run_continuous('hu', root, total_seconds=.001)
                self.assertEqual(TrainingRunner.load_checkpoint(root / 'checkpoint.pt').iteration, expected)
            storage = MemoryStorage()
            publish_status(storage, root, 'hu')
            status = json.loads(storage.objects['selfplay/hu/status.json'])
            self.assertEqual(status['iteration'], 2)
            self.assertIsNone(status['iteration_limit'])
            self.assertEqual(status['engine'], 'continuous_selfplay_v1')
            self.assertNotIn('solver/hu/status.json', storage.objects)
