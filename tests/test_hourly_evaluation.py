"""Paired outcomes, bounded retention, real export inference and publication retry."""
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from actions import ACTION_IDS
from features import FEATURE_SCHEMA_VERSION
from features.neural import numeric_names
from hybrid.policy_source import UniformLegalPolicy
from infoset import STATE_VERSION, observe
from ml.deep_cfr import NeuralAveragePolicy
from ml.export_onnx import export_average_policy
from ml.model import AveragePolicyNetwork, model_architecture
from ml.onnx_policy import OnnxPolicy
from poker_game_expresso import HandState
from test_publication import MemoryStorage
from training.evaluation_history import evaluate_published, retain_history, retain_traces
from training.poker_evaluation import EvaluationBudget, evaluate_policy, play, statistics
from training.publication import StorageError, encoded


class HourlyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.artifacts = {}
        for count in (2, 3):
            with torch.random.fork_rng():
                torch.manual_seed(820 + count)
                weights = AveragePolicyNetwork().state_dict()
            checkpoint = cls.root / f'{count}.pt'
            torch.save({'version': 5, 'feature_schema_version': FEATURE_SCHEMA_VERSION,
                        'training_metadata': {'traversal_mode': 'external_sampling'}, 'state_version': STATE_VERSION,
                        'architecture': model_architecture(FEATURE_SCHEMA_VERSION), 'actions': list(ACTION_IDS),
                        'numeric_names': list(numeric_names()), 'objective': 'hand_chip_delta', 'iteration': 1,
                        'supported_player_counts': [count], 'weights': weights}, checkpoint)
            state = HandState.start({p: 12. for p in range(count)}, 0, random.Random(32))
            path = cls.root / f'{count}.onnx'
            manifest = export_average_policy(checkpoint, path, path.with_suffix('.json'), observe(state))
            cls.artifacts[count] = (path.read_bytes(), manifest, checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_onnx_matches_torch_across_real_decisions_and_rejects_corruption(self):
        for count, (model, manifest, checkpoint) in self.artifacts.items():
            policy = OnnxPolicy(model, manifest, count)
            reference = NeuralAveragePolicy(checkpoint)
            state = HandState.start({p: 25. for p in range(count)}, 0, random.Random(832))
            while not state.terminal:
                observation = observe(state)
                np.testing.assert_allclose(policy.probabilities(observation), reference.query(observation), atol=1e-6)
                state.act('CALL' if state.to_call() else 'CHECK')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                OnnxPolicy(model + b'x', manifest, count)
            with self.assertRaises(ValueError):
                OnnxPolicy(model, manifest, 5 - count)

    def test_paired_identical_policies_have_exactly_zero_difference(self):
        for count in (2, 3):
            policy = UniformLegalPolicy()
            report, rows = evaluate_policy(policy, policy, count, EvaluationBudget(groups=2))
            self.assertEqual(report['status'], 'complete', report['warnings'])
            self.assertEqual(report['difference']['mean'], 0)
            self.assertEqual(report['difference']['interval'], [0, 0])
            self.assertEqual(report['completed_groups'], 2)
            self.assertEqual(report['hands'], 2 * 12 * count * 2)
            repeat, repeated_rows = evaluate_policy(policy, policy, count, EvaluationBudget(groups=2))
            self.assertEqual(rows, repeated_rows)
            self.assertEqual(report['river'], repeat['river'])
            self.assertGreaterEqual(report['river']['best_response_gain_bb'], -1e-10)

    def test_exact_fold_payoff_and_observable_policy_boundary(self):
        seen = []
        class FoldPolicy:
            def probabilities(self, obs):
                seen.append(obs)
                action = 'FOLD' if obs.legal_mask[ACTION_IDS.index('FOLD')] else 'CHECK'
                return tuple(float(a == action) for a in ACTION_IDS)
        policy = FoldPolicy()
        root = HandState.start({0: 25., 1: 25.}, 0, random.Random(3))
        hero = root.current_player
        expected = root.clone()
        expected.act('FOLD')
        actual, _ = play(root, hero, policy, UniformLegalPolicy(), 99, EvaluationBudget(), lambda: None)
        self.assertEqual(actual, expected.utility(hero) / root.blinds.big)
        self.assertFalse(root.terminal)
        self.assertEqual(len(seen), 1)
        self.assertFalse(hasattr(seen[0], 'deck'))

    def test_budget_and_failures_are_not_successful_results(self):
        policy = UniformLegalPolicy()
        report, rows = evaluate_policy(policy, policy, 2, EvaluationBudget(cpu_seconds=1e-12))
        self.assertEqual(report['status'], 'budget_limited')
        self.assertEqual(rows, [])
        self.assertIsNone(report['difference']['mean'])
        class BrokenPolicy:
            def probabilities(self, obs):
                raise ValueError('unsupported profile fixture')
        report, _ = evaluate_policy(BrokenPolicy(), policy, 2, EvaluationBudget(groups=1))
        self.assertEqual(report['status'], 'failed')
        self.assertIn('unsupported profile', report['warnings'][-1])

    def test_statistics_and_retention_preserve_recent_values(self):
        result = statistics([1., 2., 3.])
        self.assertEqual(result['mean'], 2.)
        self.assertEqual(result['m2'], 2.)
        self.assertAlmostEqual(result['se'], (1 / 3) ** .5)
        now = datetime.now(timezone.utc)
        old = {'evaluated_at': (now - timedelta(days=31)).isoformat(), 'iteration': 1}
        recent = {'evaluated_at': now.isoformat(), 'iteration': 2}
        compact = retain_history({'runs': [old, recent]}, now)
        self.assertEqual(compact['runs'], [recent])
        self.assertEqual(len(retain_history({'runs': [recent] * 800}, now)['runs']), 720)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / 'old.json.gz', root / 'new.json.gz'
            a.write_bytes(b'old'); b.write_bytes(b'new')
            os.utime(a, (now.timestamp() - 49 * 3600,) * 2)
            retain_traces(root, now)
            self.assertFalse(a.exists())
            self.assertTrue(b.exists())

    def test_export_to_history_lifecycle_retry_and_reference_pinning(self):
        model, manifest, _ = self.artifacts[2]
        store = MemoryStorage()
        def publish(release, iteration):
            store.objects['hu/current.json'] = encoded({'version': 1, 'track': 'hu', 'release_id': release,
                'iteration': iteration, 'history': [], 'published_at': datetime.now(timezone.utc).isoformat()})
            store.objects[f'hu/releases/{release}/average_hu.onnx'] = model
            store.objects[f'hu/releases/{release}/average_hu.json'] = encoded({**manifest, 'iteration': iteration})
        publish('a' * 64, 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = evaluate_published(store, 'hu', root, EvaluationBudget(groups=2))
            self.assertEqual(report['status'], 'complete')
            self.assertTrue(report['is_reference'])
            writes = len([e for e in store.events if e[0] == 'put'])
            self.assertIsNone(evaluate_published(store, 'hu', root))
            self.assertEqual(writes, len([e for e in store.events if e[0] == 'put']))
            publish('b' * 64, 2)
            prior = store.objects['evaluation/hu/history.json.gz']
            store.fail = 'history.json'
            with self.assertRaises(StorageError):
                evaluate_published(store, 'hu', root, EvaluationBudget(groups=1))
            self.assertEqual(prior, store.objects['evaluation/hu/history.json.gz'])
            store.fail = None
            report = evaluate_published(store, 'hu', root, EvaluationBudget(groups=1))
            self.assertEqual(report['reference']['iteration'], 1)
            self.assertEqual(len(json.loads(gzip.decompress(store.objects['evaluation/hu/history.json.gz']))['runs']), 2)
            self.assertEqual(json.loads(gzip.decompress((root / ('b' * 64 + '.json.gz')).read_bytes()))['report'], report)
            publish('c' * 64, 3)
            store.fail = 'index.json'
            with self.assertRaises(StorageError):
                evaluate_published(store, 'hu', root, EvaluationBudget(groups=1))
            store.fail = None
            with patch('training.evaluation_history.evaluate_policy', side_effect=AssertionError('Must not rerun completed evaluation')):
                self.assertIsNone(evaluate_published(store, 'hu', root))

    def test_evaluator_import_does_not_load_torch_or_training_replay(self):
        result = subprocess.run([sys.executable, '-c',
            'import training.evaluation_history, sys; assert "torch" not in sys.modules; assert "training.checkpoint" not in sys.modules'],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
