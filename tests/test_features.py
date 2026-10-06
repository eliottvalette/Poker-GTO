"""Observable feature, equity, compression and suit invariance contracts."""
from dataclasses import replace
from itertools import permutations
import random
import unittest
from actions import ACTION_IDS
from features.cards import canonical_suits, card_features, made_category
from features.deterministic import DERIVED_NAMES, derived_features
from features.equity import equity, hand_equity
from features.neural import neural_observation
from features.range_features import normalize_range
from infoset import NUMERIC_NAMES, observe
from ml.memory import TrainingSample, sample_bytes
from ml.model import encode_batch
from poker_game_expresso import HandState


class FeatureTests(unittest.TestCase):
    def test_arithmetic_uses_call_payment_and_per_opponent_stacks(self):
        hand = HandState.start({0: 25, 1: 10, 2: 40}, 0, random.Random(1))
        obs = observe(hand)
        features = dict(zip(DERIVED_NAMES, derived_features(obs.cards, obs.numeric, NUMERIC_NAMES)))
        self.assertAlmostEqual(features['pot_odds'], 1 / 2.5)
        self.assertAlmostEqual(features['call_pot'], 1 / 1.5)
        self.assertAlmostEqual(features['spr_1'], 9.5 / 1.5)
        self.assertAlmostEqual(features['spr_2'], 25 / 1.5)
        hand.actor.stack = 0.1
        obs = observe(hand)
        features = dict(zip(DERIVED_NAMES, derived_features(obs.cards, obs.numeric, NUMERIC_NAMES)))
        self.assertAlmostEqual(features['pot_odds'], 0.1 / 1.6)

    def test_all_made_categories(self):
        examples = [(48, 41, 34, 23, 16), (48, 49, 40, 33, 22), (48, 49, 44, 45, 32),
                    (48, 49, 50, 40, 33), (0, 5, 10, 15, 16), (48, 40, 32, 20, 4),
                    (48, 49, 50, 44, 45), (48, 49, 50, 51, 44), (32, 36, 40, 44, 48)]
        for index, cards in enumerate(examples):
            self.assertEqual(made_category(cards), index)
        self.assertEqual(made_category((48, 49, 50, 44, 45, 46, 4)), 6)

    def test_exact_draw_structure(self):
        # 8s 9s on Ts Js 2h: nine flush outs and eight straight outs.
        values = card_features((24, 28), (32, 36, 1))
        self.assertEqual(values[1], 1)
        self.assertAlmostEqual(values[2], 8 / 52)
        self.assertAlmostEqual(values[3], 9 / 52)
        self.assertEqual(card_features((24, 28), (32, 36, 1, 2, 3))[2:4], (0, 0))

    def test_global_suit_permutations_leave_every_neural_input_identical(self):
        hand = HandState.start({0: 25, 1: 25, 2: 25}, 0, random.Random(7))
        while hand.street != 'TURN':
            hand.act('CALL' if hand.to_call() else 'CHECK')
        original = neural_observation(observe(hand))
        for permutation in permutations(range(4)):
            changed = hand.clone()
            convert = lambda c: c // 4 * 4 + permutation[c % 4]
            for player in changed.players.values():
                player.cards = tuple(map(convert, player.cards))
            changed.board = list(map(convert, changed.board))
            changed.deck = list(map(convert, changed.deck))
            changed.assert_invariants()
            self.assertEqual(original, neural_observation(observe(changed)))

    def test_opponent_deck_and_future_actions_cannot_enter_neural_input(self):
        hand = HandState.start({0: 25, 1: 25, 2: 25}, 0, random.Random(3))
        original = neural_observation(observe(hand))
        changed = hand.clone()
        opponent = changed.players[1]
        changed.deck[:2], opponent.cards = list(opponent.cards), tuple(changed.deck[:2])
        changed.deck.reverse()
        changed.assert_invariants()
        self.assertEqual(original, neural_observation(observe(changed)))
        future = hand.clone()
        future.act('CALL')
        self.assertEqual(original, neural_observation(observe(hand)))
        self.assertNotEqual(original.history_data, neural_observation(observe(future)).history_data)

    def test_compact_replay_excludes_recall_and_reduces_accounted_bytes(self):
        hand = HandState.start({0: 25, 1: 25, 2: 25}, 0, random.Random(1))
        while hand.street != 'RIVER':
            hand.act('CALL' if hand.to_call() else 'CHECK')
        obs = observe(hand)
        compact = neural_observation(obs)
        self.assertFalse(hasattr(compact, 'recall'))
        self.assertEqual(encode_batch([obs])['numeric'].shape[1], len(compact.numeric))
        # sample_bytes recursively accepts dataclasses, including diagnostics.
        self.assertLess(sample_bytes(compact), sample_bytes(obs))
        self.assertEqual(neural_observation(replace(obs, recall='["diagnostic"]')), compact)

    def test_range_normalization_and_impossible_ranges(self):
        self.assertEqual(normalize_range({(0, 1): 2, (2, 3): 6}), {(0, 1): .25, (2, 3): .75})
        self.assertEqual(normalize_range({(0, 1): 2, (2, 3): 6}, (0,)), {(2, 3): 1})
        for weights in ({(0, 1): -1}, {(0, 0): 1}, {(0, 1): 1, (1, 0): 1}, {(0, 1): 0}):
            with self.assertRaises(ValueError):
                normalize_range(weights)

    def test_exact_equity_tie_and_multiway(self):
        board = (32, 36, 40, 44, 48)
        heads_up = hand_equity((1, 2), (5, 6), board)
        self.assertTrue(heads_up.exact)
        self.assertEqual(heads_up.equity, .5)
        self.assertIsNone(heads_up.standard_error)
        result = equity((1, 2), ({(5, 6): 1}, {(9, 10): 1}), board)
        self.assertAlmostEqual(result.equity, 1 / 3)
        with self.assertRaisesRegex(ValueError, 'compatible'):
            equity((1, 2), ({(5, 6): 1}, {(5, 6): 1}), board)

    def test_range_weighted_exact_equity_and_seeded_mc_error(self):
        hero, opponent, board = (48, 49), (44, 45), (0, 5, 10, 15)
        exact = hand_equity(hero, opponent, board)
        sampled = hand_equity(hero, opponent, board, max_exact=1, samples=2000, seed=4)
        self.assertFalse(sampled.exact)
        self.assertEqual(sampled.sampling_count, 2000)
        self.assertAlmostEqual(sampled.equity, exact.equity, delta=4 * sampled.standard_error + .001)
        self.assertEqual(sampled, hand_equity(hero, opponent, board, max_exact=1, samples=2000, seed=4))
        other = (40, 41)
        ranged = equity(hero, ({opponent: 1, other: 3},), board)
        expected = .25 * exact.equity + .75 * hand_equity(hero, other, board).equity
        self.assertAlmostEqual(ranged.equity, expected)

    def test_neural_softmax_rounding_is_normalized_without_relaxing_solver_checks(self):
        from unittest.mock import patch
        import torch
        from ml.model import AveragePolicyNetwork
        from cfr_solver import validate_strategy
        obs = observe(HandState.start({0: 25, 1: 25, 2: 25}, 0, random.Random(1)))
        raw = torch.tensor([[0.13444598019123077, 0, 0.14317865669727325, 0.09930683672428131,
                             0.18911369144916534, 0.13596239686012268, 0.14604689180850983,
                             0, 0, 0, 0, 0, 0.15194565057754517]])
        with self.assertRaises(ValueError):
            validate_strategy(tuple(raw[0].tolist()), obs.legal_mask)
        model = AveragePolicyNetwork()
        with patch.object(model, 'forward', return_value=raw):
            probabilities = model.probabilities(obs)
        validate_strategy(probabilities, obs.legal_mask)
        self.assertAlmostEqual(sum(probabilities), 1, places=15)
        with patch.object(model, 'forward', return_value=raw * 2):
            with self.assertRaisesRegex(ValueError, 'mass outside'):
                model.probabilities(obs)
