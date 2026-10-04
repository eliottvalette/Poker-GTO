"""Exact small-tree expectations for sampled average-strategy accumulation."""
import random
import unittest

from actions import ACTION_IDS, legal_actions
from cfr_solver import Traversal, regret_matching
from infoset import observe
from poker_game_expresso import HandState


class ChoiceRequired(Exception):
    def __init__(self, count):
        self.count = count


class ReplayChoices:
    def __init__(self, choices):
        self.choices = iter(choices)

    def choice(self, actions):
        try:
            index = next(self.choices)
        except StopIteration:
            raise ChoiceRequired(len(actions)) from None
        return actions[index]


def strategy(obs):
    return regret_matching([float(i + 1) for i in range(len(ACTION_IDS))], obs.legal_mask)


class AverageSamplingTests(unittest.TestCase):
    def test_exact_expectation_matches_full_tree_with_nonuniform_own_reach(self):
        root = HandState.start({0: 2.0, 1: 2.0}, 0, random.Random(3))
        while root.street != "RIVER":
            root.act("CALL" if root.to_call() else "CHECK")
        expected = {}

        def full(state, own_reach):
            if state.terminal:
                return
            obs = observe(state)
            probabilities = strategy(obs)
            if state.current_player == 0:
                expected[obs.key()] = tuple(own_reach * p for p in probabilities)
            for action in legal_actions(state):
                reach = own_reach
                if state.current_player == 0:
                    reach *= probabilities[ACTION_IDS.index(action.action_id)]
                full(Traversal.child(state, action), reach)

        full(root, 1.0)
        actual = {}
        pending = [((), 1.0)]
        leaves = 0
        while pending:
            prefix, probability = pending.pop()
            samples = []
            walk = Traversal(strategy, ReplayChoices(prefix))
            try:
                walk.average(root, 0, lambda o, target, weight: samples.append((o, target, weight)))
            except ChoiceRequired as choice:
                pending.extend((prefix + (i,), probability / choice.count) for i in range(choice.count))
                continue
            leaves += 1
            self.assertLessEqual(walk.nodes, len(prefix) + 1)
            for obs, target, weight in samples:
                row = actual.setdefault(obs.key(), [0.0] * len(ACTION_IDS))
                for i, value in enumerate(target):
                    row[i] += probability * weight * value
        self.assertGreater(leaves, 1)
        self.assertEqual(actual.keys(), expected.keys())
        for key, row in expected.items():
            for a, e in zip(actual[key], row):
                self.assertAlmostEqual(a, e, places=12)

    def test_folded_hand_payoff_requires_no_opponent_simulation(self):
        hand = HandState.start({0: 25.0, 1: 25.0, 2: 25.0}, 0, random.Random(7))
        hand.act("FOLD")
        def forbidden(_):
            self.fail("A folded player's fixed payoff needs no strategy query")
        walk = Traversal(forbidden, random.Random(0), max_nodes=1)
        self.assertEqual(walk.regrets(hand, 0, lambda *_: self.fail("Unexpected regret")), 0)
        self.assertEqual(walk.settled_prunes, 1)

    def test_unstarted_tournament_cannot_supply_a_hand_payoff(self):
        from tournament import TournamentState
        tournament = TournamentState({0: 0.0, 1: 40.0, 2: 35.0}, button=1)
        def forbidden(_):
            self.fail("An eliminated player needs no opponent strategy query")
        walk = Traversal(forbidden, random.Random(0), max_nodes=1)
        with self.assertRaisesRegex(ValueError, "requires an existing tournament hand"):
            walk.regrets(tournament, 0, lambda *_: self.fail("Unexpected regret"))
        self.assertEqual(walk.nodes, 0)
        walk = Traversal(forbidden, random.Random(0), max_nodes=1)
        with self.assertRaisesRegex(ValueError, "requires an existing tournament hand"):
            walk.average(tournament, 0, lambda *_: self.fail("Unexpected strategy"))
        self.assertEqual(walk.nodes, 0)
        self.assertIsNone(tournament.hand)
