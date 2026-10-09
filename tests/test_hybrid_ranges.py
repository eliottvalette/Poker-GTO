"""Independent probability calculations for joint beliefs and information boundaries."""
import random
import unittest

from hybrid.beliefs import BeliefState, IncompatibleObservation
from hybrid.ranges import HandRange, JointRanges
from hybrid.state import instantiate
from infoset import observe
from scripts.strategy_collector_audit import small_root


class RangeTests(unittest.TestCase):
    def test_full_combinations_and_blockers(self):
        self.assertEqual(len(HandRange.uniform().weights), 1326)
        self.assertEqual(len(HandRange.uniform((0, 1, 2)).weights), 1176)
        self.assertEqual(HandRange({(3, 2): 2, (4, 5): 0}).probability((2, 3)), 1)

    def test_joint_conditioning_is_not_independent_marginals(self):
        joint = JointRanges({0: HandRange({(0, 1): 1, (2, 3): 1}),
                             1: HandRange({(0, 4): 1, (5, 6): 3}),
                             2: HandRange({(7, 8): 1})})
        rows = joint.enumerate()
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(joint.marginals()[0].probability((0, 1)), 3 / 7)
        rng = random.Random(22)
        n = 5000
        observed = sum(joint.sample(rng)[0] == (0, 1) for _ in range(n)) / n
        self.assertLess(abs(observed - 3 / 7), .025)
        self.assertAlmostEqual(sum(d.probability for d in rows), 1)

    def test_successive_analytic_bayes_and_own_range(self):
        beliefs = BeliefState(JointRanges({0: HandRange({(0, 1): .25, (2, 3): .75}),
                                           1: HandRange({(4, 5): 1})}))
        posterior = beliefs.update(0, lambda h: .8 if h == (0, 1) else .2, model_version="known")
        self.assertAlmostEqual(posterior.public.marginals()[0].probability((0, 1)), 4 / 7)
        posterior = posterior.update(0, lambda h: .5 if h == (0, 1) else 1, model_version="known")
        self.assertAlmostEqual(posterior.public.marginals()[0].probability((0, 1)), .4)
        self.assertAlmostEqual(beliefs.public.ranges[0].probability((0, 1)), .25)
        with self.assertRaises(IncompatibleObservation):
            posterior.update(0, lambda _: 0, model_version="impossible")

    def test_public_policy_does_not_observe_hidden_cards_or_deck(self):
        state = small_root(2)
        hero = state.current_player
        other = next(p for p in state.players if p != hero)
        cards = [c for c in range(52) if c not in (*state.board, *state.actor.cards)]
        a = instantiate(state, {hero: state.actor.cards, other: tuple(cards[:2])}, random.Random(1))
        b = instantiate(state, {hero: state.actor.cards, other: tuple(cards[2:4])}, random.Random(2))
        self.assertEqual(observe(a), observe(b))
        reverse = instantiate(state, {hero: state.actor.cards[::-1], other: tuple(cards[:2])})
        self.assertEqual(observe(a), observe(reverse))

    def test_impossible_joint_is_explicit(self):
        joint = JointRanges({0: HandRange({(0, 1): 1}), 1: HandRange({(1, 2): 1})})
        with self.assertRaisesRegex(ValueError, "no compatible"):
            joint.enumerate()
        with self.assertRaisesRegex(RuntimeError, "product proposals"):
            joint.sample(random.Random(1), max_attempts=2)

    def test_private_factors_exclude_hero_blockers_without_changing_public_prior(self):
        joint = JointRanges({0: HandRange({(0, 1): 1, (2, 3): 1}),
                             1: HandRange({(0, 4): 1, (5, 6): 1})})
        private = joint.conditioned(0, (0, 1))
        self.assertEqual(private.ranges[1].probability((0, 4)), 0.)
        self.assertEqual(private.ranges[1].probability((5, 6)), 1.)
        self.assertEqual(joint.ranges[1].probability((0, 4)), .5)
