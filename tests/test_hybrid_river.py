"""Finite poker references validate CFR values, reach averaging and convergence."""
import unittest
from actions import ACTION_IDS, legal_actions
from evaluation import subgame_best_response
from hybrid.ranges import HandRange, JointRanges
from hybrid.river_solver import RiverSolver
from hybrid.state import BudgetExceeded, ComputeBudget, instantiate
from infoset import observe
from scripts.strategy_collector_audit import small_root


def fixture(count):
    root = small_root(count)
    available = [c for c in range(52) if c not in root.board]
    factors = {p: HandRange({tuple(available[4*i:4*i+2]): 1,
                             tuple(available[4*i+2:4*i+4]): 2}) for i, p in enumerate(root.players)}
    return root, JointRanges(factors, tuple(root.board))


def uniform_value(state, player):
    if state.terminal:
        return state.utility(player)
    values = []
    for action in legal_actions(state):
        child = state.clone()
        action.apply(child)
        values.append(uniform_value(child, player))
    return sum(values) / len(values)


class RiverTests(unittest.TestCase):
    def test_first_iteration_matches_exhaustive_uniform_reference(self):
        for count in (2, 3):
            root, ranges = fixture(count)
            result = RiverSolver().solve(root, ranges, ComputeBudget(iterations=1))
            expected = {p: sum(d.probability * uniform_value(instantiate(root, d.hands), p)
                               for d in ranges.enumerate()) for p in root.players}
            for p in expected:
                self.assertAlmostEqual(result.expected_utilities[p], expected[p], places=12)
            for deal in ranges.enumerate():
                state = instantiate(root, deal.hands)
                self.assertAlmostEqual(sum(result.strategy_mass[observe(state).key()]), 1)
            self.assertAlmostEqual(sum(result.expected_utilities.values()), 0)

    def test_first_root_regret_matches_independent_calculation(self):
        root, ranges = fixture(2)
        result = RiverSolver().solve(root, ranges, ComputeBudget(iterations=1))
        expected = {}
        for deal in ranges.enumerate():
            state = instantiate(root, deal.hands)
            key = observe(state).key()
            row = expected.setdefault(key, [0.] * len(ACTION_IDS))
            baseline = uniform_value(state, state.current_player)
            for action in legal_actions(state):
                child = state.clone()
                action.apply(child)
                row[ACTION_IDS.index(action.action_id)] += deal.probability * (uniform_value(child, state.current_player) - baseline)
        for key in expected:
            for actual, target in zip(result.regrets[key], expected[key]):
                self.assertAlmostEqual(actual, target)

    def test_hu_independent_infoset_best_response_improves(self):
        root, ranges = fixture(2)
        roots = [(d.probability, instantiate(root, d.hands)) for d in ranges.enumerate()]
        gains = []
        for iterations in (1, 100):
            result = RiverSolver().solve(root, ranges, ComputeBudget(iterations=iterations))
            gains.append(sum(subgame_best_response(roots, result.policy.probabilities, p)["best_response_gain_bb"]
                             for p in root.players))
        self.assertLess(gains[1], gains[0] / 5)

    def test_budget_and_determinism(self):
        root, ranges = fixture(3)
        with self.assertRaises(BudgetExceeded):
            RiverSolver().solve(root, ranges, ComputeBudget(max_nodes=2))
        budget = ComputeBudget(iterations=5)
        a = RiverSolver().solve(root, ranges, budget)
        b = RiverSolver().solve(root, ranges, budget)
        self.assertEqual(a.action_ev_by_hand, b.action_ev_by_hand)
        self.assertEqual(a.regrets, b.regrets)
        self.assertEqual(a.work.nodes, b.work.nodes)
        self.assertEqual(len(a.strategy_by_hand), 2)

    def test_sampled_three_player_cfr_matches_exact_expectation_with_more_deals(self):
        root, ranges = fixture(3)
        exact = RiverSolver().solve(root, ranges, ComputeBudget(iterations=1))
        errors = []
        for samples in (8, 128):
            squared = []
            for seed in range(6):
                result = RiverSolver().solve_sampled(root, ranges, ComputeBudget(iterations=1, samples=samples, seed=seed))
                squared.append(sum((result.expected_utilities[p] - exact.expected_utilities[p])**2 for p in root.players))
                self.assertEqual(result.work.samples, samples)
            errors.append(sum(squared))
        self.assertLess(errors[1], errors[0])
