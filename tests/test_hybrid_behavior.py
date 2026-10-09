import random
import unittest

from actions import SolverAction
from hybrid.actions import local_actions
from hybrid.behavior import likelihood, observe_action
from hybrid.beliefs import BeliefState, IncompatibleObservation
from hybrid.policy_source import UniformLegalPolicy
from hybrid.ranges import HandRange, JointRanges
from poker_game_expresso import HandState
from hybrid.experiments import small_game
from hybrid.state import ComputeBudget, instantiate
from hybrid.decision_engine import HybridDecisionEngine


class BehaviorTests(unittest.TestCase):
    def test_actual_offtree_amount_and_explicit_likelihood_kernel(self):
        root = HandState.start({0: 10, 1: 10}, 0, random.Random(1))
        beliefs = BeliefState(JointRanges({p: HandRange.uniform() for p in root.players}))
        action = SolverAction("ACTUAL_RAISE", "RAISE", 2.25)
        with self.assertRaisesRegex(ValueError, "interpolate=True"):
            likelihood(UniformLegalPolicy(), root, root.actor.cards, action)
        child, posterior = observe_action(root, beliefs, action, UniformLegalPolicy(), interpolate=True)
        self.assertEqual(child.highest, 2.25)
        self.assertEqual(child.history[-1].amount_to, 2.25)
        self.assertAlmostEqual(posterior.public.ranges[0].probability((0, 1)), 1 / 1326)
        local = local_actions(root, (2, 2.25, 2.75, 10))
        self.assertTrue(any(a.amount_to == 2.25 for a in local))
        self.assertEqual(sum(a.amount_to == 10 for a in local), 1)
        with self.assertRaises(ValueError):
            local_actions(root, (1.25,))

    def test_joint_zero_likelihood_is_not_confident_posterior(self):
        belief = BeliefState(JointRanges({0: HandRange({(0, 1): 1, (2, 3): 1}),
                                          1: HandRange({(0, 4): 1})}))
        with self.assertRaises(IncompatibleObservation):
            belief.update(0, lambda h: float(h == (0, 1)), model_version="blocked-only")

    def test_analytical_posterior_changes_call_ev(self):
        root, _ = small_game(2)
        # A known synthetic action likelihood, independent of any learned label.
        # Hero holds a middle pair; the bettor can have a weaker or stronger pair.
        factors = {0: HandRange({(4, 5): 1}), 1: HandRange({(0, 1): 1, (48, 49): 1})}
        ranges = JointRanges(factors, tuple(root.board))
        root = instantiate(root, {0: (4, 5), 1: (0, 1)})
        beliefs = BeliefState(ranges)
        posterior = beliefs.update(1, lambda h: .9 if h == (48, 49) else .1, model_version="known-synthetic")
        self.assertAlmostEqual(posterior.public.marginals()[1].probability((48, 49)), .9)
        root.act("RAISE", .5)
        budget = ComputeBudget(iterations=20)
        engine = HybridDecisionEngine()
        prior_ev = engine.analyze(root, beliefs, budget).estimated_ev_by_action["CALL"]
        posterior_ev = engine.analyze(root, posterior, budget).estimated_ev_by_action["CALL"]
        self.assertAlmostEqual(prior_ev, 0.)
        self.assertAlmostEqual(posterior_ev, -1.2)

    def test_short_allin_reraise_and_unsupported_extrapolation(self):
        short = HandState.start({0: 1.2, 1: 10}, 0, random.Random(1))
        value = likelihood(UniformLegalPolicy(), short, short.actor.cards, SolverAction("actual", "RAISE", 1.2))
        self.assertGreater(value.probability, 0)
        root = HandState.start({0: 10, 1: 10}, 0, random.Random(1))
        root.act("RAISE", 2.25)
        self.assertEqual(root.min_raise_to, 3.5)
        self.assertTrue(any(a.amount_to == 3.5 for a in local_actions(root, (3.5, 6.5))))
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            likelihood(UniformLegalPolicy(), root, root.actor.cards,
                       SolverAction("actual", "RAISE", 3.5), interpolate=True)
        self.assertGreater(likelihood(UniformLegalPolicy(), root, root.actor.cards,
                                      SolverAction("actual", "RAISE", 6.5), interpolate=True).probability, 0)
        with self.assertRaises(ValueError):
            local_actions(root, (11,))
