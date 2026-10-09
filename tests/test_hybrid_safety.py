"""Semantic symmetries, side pots and off-tree rules are preserved by local solving."""
from dataclasses import replace
import random
import unittest

from actions import legal_actions
from hybrid.experiments import small_game
from hybrid.ranges import HandRange, JointRanges
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget
from poker_game_expresso import HandState


class SafetyTests(unittest.TestCase):
    def test_global_suit_permutation_and_reversed_private_cards(self):
        root, ranges = small_game(3)
        budget = ComputeBudget(iterations=40)
        original = RiverSolver().solve(root, ranges, budget)
        permutation = [2, 3, 1, 0]
        def card(c):
            return c // 4 * 4 + permutation[c % 4]
        changed = root.clone()
        changed.board = list(map(card, root.board))
        new_ranges = JointRanges({p: HandRange({tuple(map(card, h)): w for h, w in r.weights.items()})
                                  for p, r in ranges.ranges.items()}, tuple(changed.board))
        transformed = RiverSolver().solve(changed, new_ranges, budget)
        for hand, values in original.action_ev_by_hand.items():
            key = tuple(sorted(map(card, hand)))
            for action in values:
                self.assertAlmostEqual(values[action], transformed.action_ev_by_hand[key][action])
                self.assertAlmostEqual(original.strategy_by_hand[hand][action], transformed.strategy_by_hand[key][action])
        for player in root.players.values():
            player.cards = player.cards[::-1]
        self.assertEqual(original.action_ev_by_hand, RiverSolver().solve(root, ranges, budget).action_ev_by_hand)

    def test_seat_renaming_preserves_strategic_values(self):
        root, ranges = small_game(3)
        mapping = {0: 6, 1: 2, 2: 9}
        state = root.clone()
        state.players = {mapping[p]: replace(v, player_id=mapping[p]) for p, v in root.players.items()}
        state.button = mapping[root.button]
        state.current_player = mapping[root.current_player]
        state.pending = {mapping[p] for p in root.pending}
        state.initial_stacks = {mapping[p]: v for p, v in root.initial_stacks.items()}
        state.history = [replace(e, player_id=mapping[e.player_id]) for e in root.history]
        transformed = JointRanges({mapping[p]: r for p, r in ranges.ranges.items()}, ranges.board)
        budget = ComputeBudget(iterations=10)
        a = RiverSolver().solve(root, ranges, budget)
        b = RiverSolver().solve(state, transformed, budget)
        self.assertEqual(a.action_ev_by_hand, b.action_ev_by_hand)
        self.assertEqual(a.strategy_by_hand, b.strategy_by_hand)

    def test_unequal_stack_side_pot_tree_conserves_chips(self):
        root = HandState.start({0: 1.25, 1: 1.5, 2: 2.}, 0, random.Random(12))
        while root.street != "RIVER":
            root.act("CALL" if root.to_call() else "CHECK")
        ranges = JointRanges({p: HandRange({tuple(sorted(v.cards)): 1}) for p, v in root.players.items()}, tuple(root.board))
        result = RiverSolver().solve(root, ranges, ComputeBudget(iterations=30))
        self.assertAlmostEqual(sum(result.expected_utilities.values()), 0)
        def terminals(state):
            if state.terminal:
                self.assertAlmostEqual(sum(state.utility(p) for p in state.players), 0)
                return 1
            total = 0
            for action in legal_actions(state):
                child = state.clone()
                action.apply(child)
                total += terminals(child)
            return total
        self.assertGreater(terminals(root), 10)

    def test_folded_player_cards_remain_blockers(self):
        ranges = JointRanges({0: HandRange({(0, 1): 1}), 1: HandRange({(2, 3): 1}),
                              2: HandRange({(0, 4): 1, (5, 6): 1})})
        self.assertEqual(ranges.marginals()[2].probability((0, 4)), 0)
        self.assertEqual(ranges.marginals()[2].probability((5, 6)), 1)
