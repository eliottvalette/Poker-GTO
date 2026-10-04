"""Tournament sessions provide a hand context, never a tournament learning horizon."""
from dataclasses import asdict
import random
import unittest

from actions import ACTION_IDS
from cfr_solver import Traversal, hand_root, regret_matching
from ml.deep_cfr import DeepCFRSolver, ModelSnapshot, TraversalTask, generate_samples
from ml.memory import ReservoirMemory
from outcome_sampling import OutcomeSamplingTraversal
from tournament import TournamentState


def uniform(observation):
    return regret_matching([0.0] * len(ACTION_IDS), observation.legal_mask)


class HandSolverBoundaryTests(unittest.TestCase):
    def test_terminal_hand_chip_utilities_sum_zero_and_never_start_next_hand(self):
        tournament = TournamentState(rng=random.Random(3))
        tournament.start_hand()
        tournament.hand.act("FOLD")
        tournament.hand.act("FOLD")
        self.assertTrue(tournament.hand.terminal)
        self.assertFalse(tournament.terminal)
        before = asdict(tournament.hand)
        rng_before = tournament.rng.getstate()
        def forbidden(_):
            self.fail("A terminal hand must not query a future strategy")
        external_values, outcome_values = [], []
        for player in tournament.hand.players:
            external = Traversal(forbidden, random.Random(17))
            external_values.append(external.regrets(tournament, player, lambda *_: self.fail("Unexpected regret")))
            external.average(tournament, player, lambda *_: self.fail("Unexpected policy sample"))
            outcome = OutcomeSamplingTraversal(forbidden, random.Random(17))
            outcome_values.append(outcome.run(tournament, player, lambda *_: self.fail("Unexpected regret"),
                                              lambda *_: self.fail("Unexpected policy sample")))
            self.assertEqual(outcome.max_depth_seen, 0)
        self.assertAlmostEqual(sum(external_values), 0)
        self.assertAlmostEqual(sum(outcome_values), 0)
        for value, expected in zip(external_values, [0.0, -0.5, 0.5]):
            self.assertAlmostEqual(value, expected)
        for value, expected in zip(outcome_values, external_values):
            self.assertAlmostEqual(value, expected)
        self.assertEqual(asdict(tournament.hand), before)
        self.assertEqual(tournament.rng.getstate(), rng_before)
        self.assertEqual(tournament.hand_number, 1)
        self.assertFalse(tournament.completed)

    def test_live_tournament_and_extracted_hand_generate_identical_samples(self):
        tournament = TournamentState({0: 2.0, 1: 2.0}, rng=random.Random(3))
        tournament.start_hand()
        while tournament.hand.street != "RIVER":
            tournament.hand.act("CALL" if tournament.hand.to_call() else "CHECK")
        before = asdict(tournament.hand)
        rng_before = tournament.rng.getstate()
        for mode in ("external_sampling", "outcome_sampling"):
            task = TraversalTask(0, 0, tournament, 17, 1000, 100, traversal_mode=mode)
            snapshot = ModelSnapshot(0, "hand_chip_delta", {}, True)
            actual = generate_samples(snapshot, task)
            expected = generate_samples(snapshot, TraversalTask(0, 0, hand_root(tournament), 17, 1000, 100,
                                                                traversal_mode=mode))
            self.assertEqual(actual, expected)
            self.assertGreater(actual.max_depth_seen, 0)
            self.assertTrue(all(sample.state.objective == "hand_chip_delta"
                                for sample in actual.advantages + actual.strategies))
        self.assertEqual(asdict(tournament.hand), before)
        self.assertEqual(tournament.rng.getstate(), rng_before)
        self.assertEqual(tournament.hand_number, 1)
        self.assertFalse(tournament.completed)

    def test_folded_hand_chip_delta_is_exact_without_other_players_continuing(self):
        tournament = TournamentState(rng=random.Random(3))
        tournament.start_hand()
        tournament.hand.act("CALL")
        tournament.hand.act("FOLD")
        self.assertFalse(tournament.hand.terminal)
        def forbidden(_):
            self.fail("Folded-player chip utility is already settled")
        expected = -0.5
        self.assertEqual(Traversal(forbidden, random.Random(1)).regrets(tournament, 1, lambda *_: None), expected)
        self.assertAlmostEqual(OutcomeSamplingTraversal(forbidden, random.Random(1)).run(tournament, 1,
                              lambda *_: None, lambda *_: None), expected)

    def test_missing_hand_and_tournament_objectives_are_explicit_failures(self):
        tournament = TournamentState(rng=random.Random(3))
        with self.assertRaisesRegex(ValueError, "existing tournament hand"):
            hand_root(tournament)
        self.assertIsNone(tournament.hand)
        self.assertEqual(tournament.hand_number, 0)
        self.assertEqual(DeepCFRSolver((0, 1)).objective, "hand_chip_delta")
        for objective in ("tournament_winner", "ICM"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                DeepCFRSolver((0, 1), objective)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                ReservoirMemory(10, 1, "advantage", objective)


if __name__ == "__main__":
    unittest.main()
