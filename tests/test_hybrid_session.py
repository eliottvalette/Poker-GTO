import random
import unittest

from actions import ACTION_IDS, SolverAction, legal_actions
from hybrid.beliefs import BeliefState, IncompatibleObservation
from hybrid.policy_source import ScriptedPolicy
from hybrid.ranges import HandRange, JointRanges, combo
from hybrid.reference import ProfilePolicy, solve_reference
from hybrid.session import HybridSession, Hypothesis
from hybrid.state import ComputeBudget
from hybrid.experiments import small_game
from poker_game_expresso import HandState


class SessionTests(unittest.TestCase):
    def test_private_profile_conditioning_does_not_modify_public_beliefs(self):
        root, ranges = small_game(2)
        hero = root.current_player
        hands = tuple(ranges.ranges[hero].weights)
        policies = dict.fromkeys(root.players, ProfilePolicy("loose_passive"))
        hypotheses = []
        for name, mass in (("first", 0.9), ("second", 0.1)):
            factors = dict(ranges.ranges)
            factors[hero] = HandRange({hands[0]: mass, hands[1]: 1 - mass})
            hypotheses.append(
                Hypothesis(
                    name,
                    0.5,
                    BeliefState(JointRanges(factors, tuple(root.board))),
                    policies,
                )
            )
        session = HybridSession(root, tuple(hypotheses))
        result = session.analyze(
            ComputeBudget(samples=8, max_depth=1), mode="exploitative"
        )
        self.assertAlmostEqual(
            result.uncertainty_estimates["behavior_posterior"]["first"], 0.9
        )
        self.assertEqual([h.probability for h in session.hypotheses], [0.5, 0.5])
        self.assertAlmostEqual(sum(result.own_public_range.values()), 1.0)

    def test_tournament_transition_resets_private_beliefs_and_routes_two_seats(self):
        from tournament import TournamentState
        from hybrid.policy_source import UniformLegalPolicy

        for seed in range(30):
            game = TournamentState(
                stacks={0: 3.0, 1: 6.0, 2: 9.0}, rng=random.Random(seed)
            )
            root = game.start_hand()
            ranges = JointRanges(
                {p: HandRange({combo(v.cards): 1.0}) for p, v in root.players.items()}
            )
            session = HybridSession(
                root,
                (
                    Hypothesis(
                        "uniform",
                        1.0,
                        BeliefState(ranges),
                        dict.fromkeys(root.players, UniformLegalPolicy()),
                    ),
                ),
            )
            while not game.hand.terminal:
                action = next(
                    a
                    for a in legal_actions(game.hand)
                    if a.action_id == ("ALL_IN" if game.hand.can_raise() else "CALL")
                )
                session.observe_action(action, likelihood_source="known uniform")
                game.act(action.action_id)
            if len(game.active) != 2:
                continue
            next_root = game.start_hand()
            next_ranges = JointRanges(
                {p: HandRange.uniform() for p in next_root.players}
            )
            next_session = session.next_hand(
                next_root,
                (
                    Hypothesis(
                        "uniform",
                        1.0,
                        BeliefState(next_ranges),
                        dict.fromkeys(next_root.players, UniformLegalPolicy()),
                    ),
                ),
            )
            self.assertEqual(len(next_session.hypotheses[0].policies), 2)
            self.assertEqual(next_session.events, [])
            self.assertEqual(set(next_session.state.players), set(next_ranges.ranges))
            return
        self.fail(
            "Controlled transition fixture did not produce a two-seat survivor hand"
        )

    def session(self, count):
        root = HandState.start({p: 5.0 + p for p in range(count)}, 0, random.Random(42))
        factors = {
            p: HandRange({combo(player.cards): 1}) for p, player in root.players.items()
        }
        return HybridSession(
            root,
            (
                Hypothesis(
                    "known",
                    1.0,
                    BeliefState(JointRanges(factors)),
                    dict.fromkeys(root.players, ProfilePolicy("loose_passive")),
                ),
            ),
        )

    def test_complete_multistreet_hands_and_exact_observed_amount(self):
        for count in (2, 3):
            session = self.session(count)
            session.observe_action(
                SolverAction("human", "RAISE", 2.25),
                likelihood_source="human/explicit-profile",
                interpolate=True,
            )
            self.assertEqual(session.state.history[-1].amount_to, 2.25)
            streets = {"PREFLOP"}
            while not session.state.terminal:
                result = session.analyze(
                    ComputeBudget(samples=2, max_depth=1, max_nodes=10000),
                    mode="exploitative",
                )
                self.assertAlmostEqual(sum(result.action_probabilities.values()), 1.0)
                streets.add(session.state.street)
                action = next(
                    a
                    for a in legal_actions(session.state)
                    if a.category == ("CALL" if session.state.to_call() else "CHECK")
                )
                session.observe_action(action, likelihood_source="test/known-profile")
                session.observe_board(tuple(session.state.board))
            self.assertEqual(streets, {"PREFLOP", "FLOP", "TURN", "RIVER"})
            self.assertAlmostEqual(
                sum(session.state.utility(p) for p in session.state.players), 0.0
            )

    def test_analytic_profile_hand_correlation_and_atomic_failure(self):
        root, ranges = small_game(2)
        actor = root.current_player
        hands = tuple(ranges.ranges[actor].weights)

        def policy(probability):
            def query(obs):
                values = [0.0] * len(ACTION_IDS)
                checks = ACTION_IDS.index("CHECK")
                raises = next(
                    i for i, m in enumerate(obs.legal_mask) if m and i != checks
                )
                values[checks] = (
                    probability if combo(obs.cards[:2]) == hands[0] else 1 - probability
                )
                values[raises] = 1 - values[checks]
                return tuple(values)

            return ScriptedPolicy(query, f"known-{probability}")

        hypotheses = tuple(
            Hypothesis(
                str(p), 0.5, BeliefState(ranges), dict.fromkeys(root.players, policy(p))
            )
            for p in (0.9, 0.2)
        )
        session = HybridSession(root, hypotheses)
        check = next(a for a in legal_actions(root) if a.category == "CHECK")
        prior = ranges.ranges[actor].probability(hands[0])
        evidences = [prior * p + (1 - prior) * (1 - p) for p in (0.9, 0.2)]
        session.observe_action(check, likelihood_source="synthetic exact")
        for h, p, e in zip(session.hypotheses, (0.9, 0.2), evidences):
            self.assertAlmostEqual(h.probability, e / sum(evidences))
            self.assertAlmostEqual(
                h.beliefs.public.ranges[actor].probability(hands[0]), prior * p / e
            )
        before = len(session.state.history)
        impossible = ScriptedPolicy(
            lambda obs: tuple(float(a == "CHECK") for a in ACTION_IDS), "check-only"
        )
        with self.assertRaises((ValueError, IncompatibleObservation)):
            session.observe_action(
                next(a for a in legal_actions(session.state) if a.category == "RAISE"),
                likelihood_source="impossible",
                external_policy=impossible,
            )
        self.assertEqual(len(session.state.history), before)

    def test_computed_reference_and_profile_sensitivity(self):
        from infoset import observe

        for count in (2, 3):
            root, ranges = small_game(count)
            reference = solve_reference(root, ranges, iterations=500)
            self.assertAlmostEqual(sum(reference.probabilities(observe(root))), 1.0)
            self.assertEqual(reference.table_hits, 1)
            self.assertLess(reference.validation["response_gain_sum"], 0.05)
            left = ProfilePolicy("tight_passive").probabilities(observe(root))
            right = ProfilePolicy("loose_aggressive").probabilities(observe(root))
            self.assertGreater(sum(abs(a - b) for a, b in zip(left, right)), 0.1)

    def test_no_private_future_leak_and_unknown_profile(self):
        session = self.session(3)
        from infoset import observe

        policy = ProfilePolicy("tight_aggressive")
        first = policy.probabilities(observe(session.state))
        clone = session.state.clone()
        clone.deck.reverse()
        for p in clone.players:
            if p != clone.current_player:
                clone.players[p].cards = (0, 1)
        self.assertEqual(first, policy.probabilities(observe(clone)))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            ProfilePolicy("unknown")
