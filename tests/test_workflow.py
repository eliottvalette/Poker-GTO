"""Timed checkpoint continuation and reviewed policy/ONNX publication."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from training.config import load_config
from training.runner import TrainingRunner
from training import workflow


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.catalog = self.root / 'ui/public/policy/index.json'
        self.patcher = patch.multiple(workflow, REPOSITORY=self.root, POLICY_ROOT=self.root / 'policy',
                                      UI_POLICY_ROOT=self.catalog.parent, CATALOG=self.catalog)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self.directory.cleanup)
        (self.root / 'configs').mkdir()
        for track in ('3max', 'hu'):
            config = load_config(f'configs/train_{track}.json')
            config.update(output_dir=str(self.root / 'runs' / track), workers=1, advantage_epochs=1, average_epochs=1,
                          checkpoint_every=1, evaluation_every=100)
            config[track]['traversals_per_player'] = 2
            (self.root / 'configs' / f'train_{track}.json').write_text(json.dumps(config))

    def train(self, track):
        return workflow.train_track(track, .001)

    def test_timed_training_resumes_without_resetting_iteration(self):
        first = self.train('3max')
        self.assertEqual(first['final_iteration'], 1)
        second = self.train('3max')
        self.assertEqual(second['initial_iteration'], 1)
        self.assertEqual(second['final_iteration'], 2)
        self.assertGreater(first['elapsed_seconds'], first['requested_seconds'])
        runner = TrainingRunner.load_checkpoint(workflow.candidate_checkpoint('3max'))
        self.assertEqual(set(runner.solvers), {'3max'})

    def test_activation_requires_publication_and_active_checkpoint_can_resume(self):
        self.train('hu')
        prepared = workflow.prepare_migration('activate', ('hu',))
        self.addCleanup(prepared.close)
        self.assertFalse(self.catalog.exists())
        prepared.publish()
        active = workflow.active_checkpoint('hu')
        self.assertIsNotNone(active)
        self.assertEqual(workflow.read_catalog()['active']['hu']['iteration'], 1)
        self.assertEqual(workflow.read_catalog()['exports'], {})
        workflow.candidate_checkpoint('hu').unlink()
        continued = self.train('hu')
        self.assertEqual(continued['initial_iteration'], 1)
        self.assertEqual(continued['final_iteration'], 2)

    def test_export_both_publishes_complete_ui_bundle_without_activation(self):
        self.train('3max')
        self.train('hu')
        prepared = workflow.prepare_migration('export', ('3max', 'hu'))
        self.addCleanup(prepared.close)
        self.assertFalse(self.catalog.exists())
        prepared.publish()
        catalog = workflow.read_catalog()
        self.assertEqual(catalog['active'], {})
        self.assertEqual(set(catalog['exports']), {'3max', 'hu'})
        bundle = workflow.UI_POLICY_ROOT / 'releases' / catalog['exports']['hu']['bundle_id']
        manifest = workflow.validate_bundle(bundle)
        self.assertEqual(len(manifest['files']), 4)
        for track in ('3max', 'hu'):
            exported = json.loads((bundle / f'average_{track}.json').read_text())
            self.assertLess(exported['validation_max_absolute_error'], 1e-6)
        combined = workflow.prepare_migration('both', ('3max', 'hu'))
        self.addCleanup(combined.close)
        combined.publish()
        current = workflow.read_catalog()
        self.assertEqual(set(current['active']), {'3max', 'hu'})
        self.assertEqual(set(current['exports']), {'3max', 'hu'})
        for track in ('3max', 'hu'):
            self.assertEqual(current['active'][track]['release_id'], current['exports'][track]['checkpoint_sha256'])

    def test_failed_export_does_not_publish_and_catalog_conflict_is_explicit(self):
        self.train('3max')
        self.train('hu')
        with patch('training.workflow.export_average_policy', side_effect=RuntimeError('Injected export failure')):
            with self.assertRaisesRegex(RuntimeError, 'Injected export failure'):
                workflow.prepare_migration('both', ('3max', 'hu'))
        self.assertFalse(self.catalog.exists())
        self.assertFalse(list(self.root.rglob('.prepared-*')))
        prepared = workflow.prepare_migration('activate', ('3max',))
        self.addCleanup(prepared.close)
        self.catalog.parent.mkdir(parents=True, exist_ok=True)
        self.catalog.write_text(json.dumps({'version': 1, 'active': {}, 'exports': {}}))
        with self.assertRaisesRegex(RuntimeError, 'changed since preparation'):
            prepared.publish()
        self.assertEqual(workflow.read_catalog()['active'], {})

    def test_corrupt_candidate_does_not_fall_back_to_active_policy(self):
        self.train('3max')
        prepared = workflow.prepare_migration('activate', ('3max',))
        prepared.publish()
        workflow.candidate_checkpoint('3max').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Invalid training checkpoint'):
            self.train('3max')

    def test_interrupt_saves_only_complete_state(self):
        config = workflow.config_for('hu')
        runner = TrainingRunner(config)
        runner.run_iteration()
        path = workflow.candidate_checkpoint('hu')
        with patch.object(runner, 'run_iteration', side_effect=KeyboardInterrupt):
            result = runner.run_for(1, path)
        self.assertEqual(result['stop_reason'], 'interrupted')
        self.assertEqual(TrainingRunner.load_checkpoint(path).iteration, 1)

    def test_independent_ui_exports_require_only_selected_track_and_preserve_other(self):
        self.train('3max')
        prepared = workflow.prepare_migration('export', ('3max',))
        prepared.publish()
        first = workflow.read_catalog()
        self.assertEqual(set(first['exports']), {'3max'})
        self.assertEqual(first['active'], {})
        bundle = workflow.UI_POLICY_ROOT / 'releases' / first['exports']['3max']['bundle_id']
        self.assertEqual(len(workflow.validate_bundle(bundle)['files']), 2)
        self.train('hu')
        workflow.prepare_migration('export', ('hu',)).publish()
        second = workflow.read_catalog()
        self.assertEqual(second['exports']['3max'], first['exports']['3max'])
        self.assertEqual(set(second['exports']), {'3max', 'hu'})
        self.assertEqual(second['active'], {})
        self.train('3max')
        workflow.prepare_migration('export', ('3max',)).publish()
        final = workflow.read_catalog()
        self.assertEqual(final['exports']['hu'], second['exports']['hu'])
        self.assertEqual(final['exports']['3max']['iteration'], 2)
