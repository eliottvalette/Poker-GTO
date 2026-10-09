"""Action values are measured against settled chips and conditional chance."""
import random
import unittest

from features.equity import hand_equity
from hybrid.beliefs import BeliefState
from hybrid.decision_engine import HybridDecisionEngine
from hybrid.policy_source import category_policy
from hybrid.ranges import HandRange, JointRanges
from hybrid.search import search
from hybrid.state import BudgetExceeded, ComputeBudget, instantiate
from poker_game_expresso import HandState


def turn_fixture():
    root = HandState.start({0: 1.5, 1: 1.5}, 0, random.Random(7))
    while root.street != "TURN":
        root.act("CALL" if root.to_call() else "CHECK")
    root.board = [0, 5, 10, 31]
    hands = {0: (48, 49), 1: (44, 45)}
    root = instantiate(root, hands)
    ranges = JointRanges({p: HandRange({h: 1}) for p, h in hands.items()}, tuple(root.board))
    return root, ranges


class SearchTests(unittest.TestCase):
    def test_turn_mc_agrees_with_independent_exact_equity_when_all_called(self):
        root, ranges = turn_fixture()
        hero = root.current_player
        policies = {p: category_policy({"CALL": 1, "CHECK": 1}, "passive") for p in root.players}
        result = search(root, ranges, hero, policies, ComputeBudget(samples=2000, max_depth=1))
        other = next(p for p in root.players if p != hero)
        equity = hand_equity(root.players[hero].cards, root.players[other].cards, tuple(root.board)).equity
        expected = 3 * equity - 1.5
        self.assertLess(abs(result.action_ev["ALL_IN"] - expected), 4 * result.standard_errors["ALL_IN"])
        self.assertEqual(result.work.samples, 2000)

    def test_engine_without_model_all_streets_and_counts(self):
        for count in (2, 3):
            for street in ("PREFLOP", "FLOP", "TURN", "RIVER"):
                root = HandState.start({p: 1.5 for p in range(count)}, 0, random.Random(5))
                while root.street != street:
                    root.act("CALL" if root.to_call() else "CHECK")
                ranges = JointRanges({p: HandRange({tuple(sorted(h.cards)): 1}) for p, h in root.players.items()}, tuple(root.board))
                engine = HybridDecisionEngine()
                result = engine.analyze(root, BeliefState(ranges), ComputeBudget(samples=8, iterations=10, max_depth=1))
                self.assertAlmostEqual(sum(result.action_probabilities.values()), 1)
                self.assertEqual(set(result.action_probabilities), set(result.estimated_ev_by_action))
                self.assertLessEqual(result.search_nodes, 100000)

    def test_determinism_hidden_deck_and_cancellation(self):
        root, ranges = turn_fixture()
        engine = HybridDecisionEngine()
        budget = ComputeBudget(samples=20)
        a = engine.analyze(root, BeliefState(ranges), budget)
        root.deck.reverse()
        b = engine.analyze(root, BeliefState(ranges), budget)
        self.assertEqual(a.estimated_ev_by_action, b.estimated_ev_by_action)
        with self.assertRaises(BudgetExceeded):
            engine.analyze(root, BeliefState(ranges), budget, cancelled=lambda: True)
