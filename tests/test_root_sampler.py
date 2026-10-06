"""Root distribution and identity invariants."""
import random
import unittest
from dataclasses import asdict
from infoset import observe
from poker_game_expresso import HandState
from training.root_sampler import RootSampler, canonical_seats


def root_config(source=None):
    weights = {'on_policy': .5, 'synthetic': .25, 'stratified': .25} if source is None else {s: float(s == source) for s in ('on_policy', 'synthetic', 'stratified')}
    return {'mixture': weights, 'max_rollout_decisions': 128, 'max_rollout_hands': 128, 'tournament_start_players': 3}


class RootSamplerTests(unittest.TestCase):
    def test_reproducible_valid_conserved_roots_for_both_tracks(self):
        for count in (2, 3):
            a, b = (RootSampler(count, 42, root_config()) for _ in range(2))
            for _ in range(40):
                left, right = a.sample(), b.sample()
                self.assertEqual(asdict(left), asdict(right))
                self.assertEqual(tuple(left.players), tuple(range(count)))
                self.assertAlmostEqual(sum(left.initial_stacks.values()), 75)
                self.assertTrue(all(s > 0 for s in left.initial_stacks.values()))
                left.assert_invariants()
            self.assertEqual(a.coverage.as_dict(), b.coverage.as_dict())
            self.assertEqual(set(a.coverage.as_dict()['source']), {'on_policy', 'synthetic', 'stratified'})

    def test_stratified_cartesian_level_position_depth_and_asymmetry(self):
        for count in (2, 3):
            sampler = RootSampler(count, 2, root_config('stratified'))
            roots = [sampler.sample() for _ in range(12 * count)]
            self.assertEqual({r.blind_level_index for r in roots}, set(range(6)))
            self.assertEqual({r.button for r in roots}, set(range(count)))
            coverage = sampler.coverage.as_dict()
            self.assertEqual(set(coverage['effective_stack_bin']), {'deep', 'medium', 'shallow'})
            self.assertEqual(set(coverage['stack_ratio_class']), {'balanced', 'asymmetric'})

    def test_on_policy_persists_hands_and_calls_observable_policy(self):
        sampler = RootSampler(3, 42, root_config('on_policy'))
        observed = []
        def policy(obs):
            observed.append(obs)
            from cfr_solver import regret_matching
            from actions import ACTION_IDS
            return regret_matching([0] * len(ACTION_IDS), obs.legal_mask)
        sampler.policy = policy
        roots = [sampler.sample() for _ in range(10)]
        self.assertTrue(observed)
        self.assertTrue(any(root.hand_number > 1 for root in roots))
        self.assertTrue(all(len(sampler.tournament.completed) <= 1 for _ in roots))

    def test_survivor_ids_are_remapped_without_changing_public_semantics(self):
        hand = HandState.start({1: 5, 2: 70}, 2, random.Random(5))
        result = canonical_seats(hand)
        self.assertEqual(tuple(result.players), (0, 1))
        self.assertEqual(result.button, 1)
        self.assertEqual(result.players[0].cards, hand.players[1].cards)
        self.assertEqual(result.players[1].position, hand.players[2].position)
        self.assertEqual(result.initial_stacks, {0: 5, 1: 70})
        self.assertEqual(result.current_player, 1)
        self.assertEqual([e.player_id for e in result.history], [1, 0])
        self.assertEqual(observe(result).numeric, observe(hand).numeric)
