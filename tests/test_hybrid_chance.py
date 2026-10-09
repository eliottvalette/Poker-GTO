"""Public chance CFR uses shared information sets and real runout probabilities."""
import unittest
from hybrid.experiments import small_game
from hybrid.river_solver import RiverSolver
from hybrid.state import ComputeBudget, instantiate
from actions import legal_actions


def uniform_value(state, hero):
    if state.terminal:
        return state.utility(hero)
    values = []
    for action in legal_actions(state):
        child = state.clone()
        action.apply(child)
        values.append(uniform_value(child, hero))
    return sum(values) / len(values)


class ChanceTests(unittest.TestCase):
    def test_exhaustive_turn_matches_independent_full_tree(self):
        root, ranges = small_game(2, street="TURN")
        result = RiverSolver().solve_chance(root, ranges, ComputeBudget(iterations=1, samples=176, max_depth=64))
        expected = {p: 0. for p in root.players}
        for deal in ranges.enumerate():
            initial = instantiate(root, deal.hands)
            for card in initial.deck:
                state = initial.clone()
                state.deck.remove(card)
                state.deck.append(card)
                for p in expected:
                    expected[p] += deal.probability * uniform_value(state, p) / len(initial.deck)
        for p in expected:
            self.assertAlmostEqual(result.expected_utilities[p], expected[p], places=10)
        self.assertEqual(result.work.samples, 176)
        self.assertEqual(result.work.rollout_leaves, 0)

    def test_preflop_and_flop_have_bounded_actual_cfr_refinement(self):
        for street in ("PREFLOP", "FLOP"):
            root, ranges = small_game(2, street=street)
            budget = ComputeBudget(iterations=4, samples=8, max_depth=1)
            result = RiverSolver().solve_chance(root, ranges, budget)
            self.assertEqual(result.complete_iterations, 4)
            self.assertEqual(result.work.samples, 8)
            self.assertGreater(result.work.rollout_leaves, 0)
            self.assertAlmostEqual(sum(result.expected_utilities.values()), 0)
            self.assertTrue(any(any(v != 0 for v in row) for row in result.regrets.values()))
