"""Fresh worlds, information boundaries, paired rollouts and capped-fit prefixes."""
import copy
import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from actions import ACTION_IDS, apply_action
from cfr_solver import regret_matching
from hybrid.policy_source import UniformLegalPolicy
from infoset import observe
from ml.memory import TrainingSample
from ml.model import AveragePolicyNetwork
from ml.train import fit
from poker_game_expresso import HandState
from training.independent_audit import deal_identity, hourly_exclusions, make_suite, rollout, unpack_hand
from training.hu_budget_audit import historical_projection, conditional_panel_statistics


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fresh_suite_preserves_infosets_and_excludes_hourly_river(self):
        suite = make_suite(2, 128)
        exclusions, boards = hourly_exclusions()
        identities = [g['deal_sha256'] for g in suite.games]
        for case, worlds in zip(suite.cases, suite.worlds):
            observation = observe(worlds[0])
            self.assertEqual(tuple(case['legal_mask']), observation.legal_mask)
            for world in worlds:
                world.assert_invariants()
                self.assertEqual(observe(world), observation)
                identities.append(deal_identity(world))
                if world.street == 'RIVER':
                    self.assertNotIn(tuple(sorted(world.board[:3]) + world.board[3:]), boards)
        self.assertEqual(len(set(identities)), len(identities))
        self.assertFalse(set(identities) & exclusions)
        self.assertEqual(len({c['branch'] for c in suite.cases}), 8)

    def test_terminal_payoff_and_self_comparison_are_exact(self):
        root = HandState.start({0: 25., 1: 25.}, 0, random.Random(41))
        folded = root.clone()
        apply_action(folded, 'FOLD')
        policy = UniformLegalPolicy()
        self.assertEqual(rollout(folded, 0, policy, policy, 7), -.5)
        self.assertEqual(rollout(folded, 1, policy, policy, 7), .5)
        self.assertEqual(rollout(root, 0, policy, policy, 7), rollout(root, 0, policy, policy, 7))

    def test_suite_serialization_preserves_complete_worlds(self):
        suite = make_suite(2, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'suite.json.gz'
            suite.save(path)
            with gzip.open(path, 'rt') as stream:
                raw = json.load(stream)
            for original, stored in zip(suite.worlds, raw['worlds']):
                for before, payload in zip(original, stored):
                    self.assertEqual(before, unpack_hand(payload))
            for before, payload in zip(suite.games, raw['games']):
                self.assertEqual(before['root'], unpack_hand(payload['root']))
            self.assertFalse(path.with_suffix('.gz.partial').exists())

    def test_historical_mean_separates_conflicting_teachers_from_fit_error(self):
        means, weights, scatter = historical_projection(
            np.asarray([[1., 0.], [0., 1.]]), np.ones(2), np.zeros(2, dtype=int))
        np.testing.assert_array_equal(means, [[.5, .5]])
        np.testing.assert_array_equal(weights, [2.])
        self.assertAlmostEqual(scatter, np.log(2))
        self.assertEqual(conditional_panel_statistics(np.zeros((3, 4)))['interval'], [0., 0.])

    def test_repeated_caps_match_the_same_optimizer_prefix(self):
        root = HandState.start({0: 25., 1: 25.}, 0, random.Random(9))
        obs = observe(root)
        samples = [TrainingSample(i, obs.hero, obs, regret_matching([0.] * len(ACTION_IDS), obs.legal_mask),
                                  1., 'strategy', i - 1, root_group=str(i % 3), trajectory_id=str(i))
                   for i in range(1, 21)]
        torch.manual_seed(91)
        base = AveragePolicyNetwork()
        model = copy.deepcopy(base)
        milestones = {}
        original = torch.optim.Adam.step
        count = 0
        def step(optimizer, *args, **kwargs):
            nonlocal count
            result = original(optimizer, *args, **kwargs)
            count += 1
            if count in (2, 4, 8):
                milestones[count] = copy.deepcopy(model.state_dict())
            return result
        with patch.object(torch.optim.Adam, 'step', step):
            fit(model, samples, 10, 4, 93, max_updates=8, cache_encoding=True,
                sampling='weighted_replacement', explicit_partitions=(samples[:16], samples[16:]))
        for updates in (2, 4, 8):
            capped = copy.deepcopy(base)
            fit(capped, samples, 10, 4, 93, max_updates=updates, cache_encoding=True,
                sampling='weighted_replacement', explicit_partitions=(samples[:16], samples[16:]))
            for key, weight in capped.state_dict().items():
                self.assertTrue(torch.equal(weight, milestones[updates][key]), key)
