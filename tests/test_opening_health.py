"""Opening diagnostics block publication only when explicitly required."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from cfr_solver import regret_matching
from training.opening_health import assess_opening
from training.publication import encoded, publish_pending
from training.workflow import seal_bundle
from test_publication import MemoryStorage, bundle
from actions import ACTION_IDS


class ConstantPolicy:
    def probabilities(self, obs):
        return regret_matching([0.] * len(ACTION_IDS), obs.legal_mask)


class OpeningHealthTests(unittest.TestCase):
    def test_constant_cold_policy_is_rejected(self):
        solver = SimpleNamespace(players=(0, 1, 2), average_model=ConstantPolicy(),
                                 advantage_memory=SimpleNamespace(samples=[]),
                                 strategy_memory=SimpleNamespace(samples=[]))
        result = assess_opening(solver)
        self.assertFalse(result['passed'])
        self.assertEqual(result['maximum_action_span'], 0.)
        self.assertEqual(result['coverage']['strategy']['classes'], 0)

    def test_rejected_export_cannot_replace_live_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exports = root / 'exports'
            exports.mkdir()
            candidate = bundle(root, 'hu', 1)
            destination = exports / 'iteration_000001'
            candidate.rename(destination)
            (destination / 'bundle.json').unlink()
            seal_bundle(destination, {'kind': 'live_training_export', 'tracks': ['hu'], 'iteration': 1,
                                     'opening_health': {'hu': {'passed': False, 'reasons': ['insufficient coverage']}}})
            (exports / 'current.json').write_text(json.dumps({'directory': destination.name}))
            storage = MemoryStorage()
            storage.objects['hu/current.json'] = encoded({'existing': 'unchanged'})
            self.assertIsNone(publish_pending(storage, 'hu', root, require_opening_health=True))
            self.assertEqual(json.loads(storage.objects['hu/current.json']), {'existing': 'unchanged'})
            self.assertFalse(any(event[0] == 'put' for event in storage.events))

            del storage.objects['hu/current.json']
            pointer = publish_pending(storage, 'hu', root, require_opening_health=False)
            self.assertEqual(pointer['iteration'], 1)
            self.assertEqual(json.loads(storage.objects['hu/current.json']), pointer)
