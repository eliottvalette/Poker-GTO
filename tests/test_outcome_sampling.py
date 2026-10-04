"""Exact and sampled estimator checks on finite Hold'em river trees."""
from __future__ import annotations

import math
import random
import unittest
from collections import defaultdict
from dataclasses import asdict
from typing import Callable

from actions import ACTION_IDS, legal_actions
from cfr_solver import Traversal, TraversalBudgetExceeded
from infoset import Observation, observe
from outcome_sampling import ImportanceNumericalError, OutcomeSamplingTraversal, _finite_exp, behavior_policy
from poker_game_expresso import HandState
from tournament import TournamentState

Strategy = Callable[[Observation], tuple[float, ...]]
EPSILON = 0.6
PLAYER = 1


def river() -> HandState:
    state = HandState.start({0: 2.0, 1: 2.0}, 0, random.Random(3))
    while state.street != "RIVER":
        state.act("CALL" if state.to_call() else "CHECK")
    return state


def threeway_river() -> HandState:
    state = HandState.start({0: 2.0, 1: 2.0, 2: 2.0}, 0, random.Random(3))
    while state.street != "RIVER":
        state.act("CALL" if state.to_call() else "CHECK")
    return state


def biased_strategy(obs: Observation) -> tuple[float, ...]:
    weights = tuple(float(1 + index % 3) if legal else 0.0 for index, legal in enumerate(obs.legal_mask))
    return tuple(weight / sum(weights) for weight in weights)


def zero_opponent_strategy(obs: Observation) -> tuple[float, ...]:
    weights = list(biased_strategy(obs))
    if obs.hero != PLAYER:
        legal = [i for i, allowed in enumerate(obs.legal_mask) if allowed]
        if len(legal) > 1:
            weights[legal[-1]] = 0.0
    return tuple(weight / sum(weights) for weight in weights)


def zero_own_strategy(obs: Observation) -> tuple[float, ...]:
    weights = list(biased_strategy(obs))
    if obs.hero == PLAYER:
        legal = [i for i, allowed in enumerate(obs.legal_mask) if allowed]
        if len(legal) > 1:
            weights[legal[0]] = 0.0
    return tuple(weight / sum(weights) for weight in weights)


class PrefixRNG(random.Random):
    """Choose a complete enumerated behavior path via probability midpoints."""
    def __init__(self, choices: tuple[float, ...]):
        super().__init__(0)
        self.choices = choices
        self.cursor = 0

    def random(self) -> float:
        if self.cursor >= len(self.choices):
            raise AssertionError("Outcome sampler requested an unexpected decision")
        value = self.choices[self.cursor]
        self.cursor += 1
        return value


def add_row(rows: dict[str, list[float]], key: str, target: tuple[float, ...], scale: float) -> None:
    row = rows.setdefault(key, [0.0] * len(ACTION_IDS))
    for index, value in enumerate(target):
        row[index] += scale * value


def full_reference(root: HandState, strategy: Strategy) -> tuple[float, dict, dict]:
    regrets, averages = {}, {}

    def recurse(state: HandState, own: float, opponent: float) -> float:
        if state.terminal:
            return state.utility(PLAYER)
        obs = observe(state)
        sigma = strategy(obs)
        updating = state.current_player == PLAYER
        if updating:
            add_row(averages, obs.key(), sigma, own)
        values = [0.0] * len(ACTION_IDS)
        for action in legal_actions(state):
            index = ACTION_IDS.index(action.action_id)
            values[index] = recurse(Traversal.child(state, action), own * sigma[index] if updating else own,
                                    opponent if updating else opponent * sigma[index])
        node_value = sum(p * value for p, value in zip(sigma, values))
        if updating:
            target = tuple(value - node_value if legal else 0.0 for value, legal in zip(values, obs.legal_mask))
            add_row(regrets, obs.key(), target, opponent)
        return node_value

    return recurse(root, 1.0, 1.0), regrets, averages


def behavior_paths(root: HandState, strategy: Strategy):
    def recurse(state: HandState, prefix: tuple[float, ...], probability: float):
        if state.terminal:
            yield prefix, probability
            return
        obs = observe(state)
        sigma = strategy(obs)
        q = [(1 - EPSILON) * p + EPSILON / sum(obs.legal_mask) if legal else 0.0
             for p, legal in zip(sigma, obs.legal_mask)]
        for action in legal_actions(state):
            index = ACTION_IDS.index(action.action_id)
            midpoint = sum(q[:index]) + q[index] / 2
            yield from recurse(Traversal.child(state, action), prefix + (midpoint,), probability * q[index])
    return list(recurse(root, (), 1.0))


class OutcomeSamplingTests(unittest.TestCase):
    def assert_rows_equal(self, actual: dict, expected: dict) -> None:
        # Missing infosets contribute exactly zero, rather than disappearing from expectations.
        for key in set(actual) | set(expected):
            left = actual.get(key, [0.0] * len(ACTION_IDS))
            right = expected.get(key, [0.0] * len(ACTION_IDS))
            for index, (a, b) in enumerate(zip(left, right)):
                self.assertAlmostEqual(a, b, places=10, msg=f"Action {ACTION_IDS[index]} estimator mismatch")

    def test_exact_path_expectations_match_full_tree(self):
        for factory in (river, threeway_river):
            for strategy in (biased_strategy, zero_opponent_strategy, zero_own_strategy):
                with self.subTest(state=factory.__name__, strategy=strategy.__name__):
                    root = factory()
                    baseline = asdict(root)
                    expected_value, expected_regrets, expected_average = full_reference(root, strategy)
                    paths = behavior_paths(root, strategy)
                    self.assertAlmostEqual(sum(probability for _, probability in paths), 1.0)
                    value, regrets, averages = 0.0, {}, {}
                    for prefix, probability in paths:
                        rng = PrefixRNG(prefix)
                        def regret_sink(obs, target, weight):
                            self.assertEqual(weight, 1.0)
                            self.assertTrue(all(v == 0 for v, legal in zip(target, obs.legal_mask) if not legal))
                            add_row(regrets, obs.key(), target, probability * weight)
                        def average_sink(obs, target, weight):
                            self.assertEqual(target, strategy(obs))
                            self.assertGreater(weight, 0)
                            add_row(averages, obs.key(), target, probability * weight)
                        walk = OutcomeSamplingTraversal(strategy, rng, epsilon=EPSILON)
                        value += probability * walk.run(root, PLAYER, regret_sink, average_sink)
                        self.assertLessEqual(rng.cursor, len(prefix))
                    self.assertAlmostEqual(value, expected_value, places=10)
                    self.assert_rows_equal(regrets, expected_regrets)
                    self.assert_rows_equal(averages, expected_average)
                    self.assertEqual(asdict(root), baseline)

    def test_deeper_hero_decisions_are_sampled_and_average_uses_prefix_reach(self):
        root = river()
        witnesses = 0
        for prefix, _ in behavior_paths(root, biased_strategy):
            samples, averages = [], []
            walk = OutcomeSamplingTraversal(biased_strategy, PrefixRNG(prefix), epsilon=EPSILON)
            walk.run(root, PLAYER, lambda o, target, weight: samples.append((o, target, weight)),
                     lambda o, target, weight: averages.append((o, target, weight)))
            if len(samples) > 1:
                witnesses += 1
                self.assertGreater(len({obs.key() for obs, _, _ in samples}), 1)
                # Reconstruct the own reach and behavior reach from the sampled prefix.
                state, own, behavior = root.clone(), 1.0, 1.0
                average_weights = {o.key(): weight for o, _, weight in averages}
                for draw in prefix:
                    obs = observe(state)
                    sigma = biased_strategy(obs)
                    if state.current_player == PLAYER:
                        self.assertAlmostEqual(average_weights[obs.key()], own / behavior)
                    q = [(1 - EPSILON) * p + EPSILON / sum(obs.legal_mask) if legal else 0.0
                         for p, legal in zip(sigma, obs.legal_mask)]
                    cumulative = 0.0
                    selected = None
                    for index, p in enumerate(q):
                        cumulative += p
                        if p > 0 and draw < cumulative:
                            selected = index
                            break
                    self.assertIsNotNone(selected)
                    action = next(a for a in legal_actions(state) if a.action_id == ACTION_IDS[selected])
                    if state.current_player == PLAYER:
                        own *= sigma[selected]
                    behavior *= q[selected]
                    state = Traversal.child(state, action)
        self.assertGreater(witnesses, 0)

    def test_empirical_means_include_zero_for_unvisited_infosets(self):
        trials = 1000
        for variant, factory in (("outcome", river), ("external", river), ("outcome_threeway", threeway_river)):
            with self.subTest(variant=variant):
                root = factory()
                expected_value, expected_regrets, expected_average = full_reference(root, biased_strategy)
                totals, squares = defaultdict(float), defaultdict(float)
                for seed in range(trials):
                    rows = {}
                    def regret_sink(obs, target, weight):
                        key = obs.key()
                        for index, value in enumerate(target):
                            rows[("regret", key, index)] = weight * value
                    def average_sink(obs, target, weight):
                        key = obs.key()
                        for index, value in enumerate(target):
                            rows[("average", key, index)] = weight * value
                    if variant.startswith("outcome"):
                        value = OutcomeSamplingTraversal(biased_strategy, random.Random(seed), epsilon=EPSILON).run(
                            root, PLAYER, regret_sink, average_sink)
                    else:
                        value = Traversal(biased_strategy, random.Random(seed)).regrets(root, PLAYER, regret_sink)
                        Traversal(biased_strategy, random.Random(seed + 10000)).average(root, PLAYER, average_sink)
                    rows[("value", "root", 0)] = value
                    for key, value in rows.items():
                        totals[key] += value
                        squares[key] += value * value
                expected = {("value", "root", 0): expected_value}
                for label, table in (("regret", expected_regrets), ("average", expected_average)):
                    for key, row in table.items():
                        for index, value in enumerate(row):
                            expected[(label, key, index)] = value
                for key in set(expected) | set(totals):
                    mean = totals[key] / trials  # Includes zero for every absent trial.
                    variance = max(0.0, squares[key] / trials - mean * mean)
                    # Six empirical standard errors over all 1,000 trials;
                    # exact path enumeration above supplies the nonstatistical oracle.
                    tolerance = 6 * math.sqrt(variance / trials) + 1e-10
                    self.assertLessEqual(abs(mean - expected.get(key, 0)), tolerance,
                                         f"{variant}: {key[0]} {key[2]} differs from exact expectation")

    def test_budget_failure_emits_no_partial_samples_and_preserves_root(self):
        root = river()
        baseline = asdict(root)
        samples = []
        for limits in ({"max_nodes": 1}, {"max_depth": 0}):
            with self.subTest(limits=limits):
                walk = OutcomeSamplingTraversal(biased_strategy, random.Random(4), epsilon=EPSILON, **limits)
                rng_state = walk.rng.getstate()
                with self.assertRaises(TraversalBudgetExceeded):
                    walk.run(root, PLAYER, lambda *sample: samples.append(sample), lambda *sample: samples.append(sample))
                self.assertEqual(samples, [])
                self.assertEqual(asdict(root), baseline)
                self.assertEqual(walk.rng.getstate(), rng_state)

    def test_invalid_or_unrepresentable_behavior_probabilities_fail_explicitly(self):
        obs = observe(river())
        for epsilon in (0, -0.1, 1.01, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "epsilon"):
                    OutcomeSamplingTraversal(biased_strategy, random.Random(0), epsilon=epsilon)
        sigma = tuple(float(index == ACTION_IDS.index("CHECK")) for index in range(len(ACTION_IDS)))
        with self.assertRaisesRegex(ImportanceNumericalError, "full support"):
            behavior_policy(sigma, obs.legal_mask, math.ulp(0.0))
        invalid = tuple(float("nan") if legal else 0.0 for legal in obs.legal_mask)
        with self.assertRaisesRegex(ValueError, "Invalid action distribution"):
            behavior_policy(invalid, obs.legal_mask, EPSILON)

    def test_log_conversion_preserves_large_values_and_rejects_float64_limits(self):
        self.assertEqual(_finite_exp(-math.inf, "structural zero"), 0)
        self.assertAlmostEqual(_finite_exp(math.log(1e200), "large target") / 1e200, 1.0, places=12)
        for value in (1000.0, -1000.0, math.inf, math.nan):
            with self.subTest(log_value=value):
                with self.assertRaises(ImportanceNumericalError):
                    _finite_exp(value, "unrepresentable target")

    def test_unstarted_tournament_is_not_a_learning_root(self):
        tournament = TournamentState({0: 0.0, 1: 37.5, 2: 37.5}, button=1)
        def unavailable_strategy(_):
            self.fail("An eliminated traverser must not query opponent strategies")
        samples = []
        walk = OutcomeSamplingTraversal(unavailable_strategy, random.Random(5), epsilon=EPSILON)
        with self.assertRaisesRegex(ValueError, "requires an existing tournament hand"):
            walk.run(tournament, 0, lambda *args: samples.append(args), lambda *args: samples.append(args))
        self.assertEqual(samples, [])
        self.assertIsNone(tournament.hand)
