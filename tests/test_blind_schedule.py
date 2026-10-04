"""Blind progression is hand-boundary context, never a chip denomination change."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import random
import unittest

from blind_schedule import BlindSchedule, BlindStage, DEFAULT_SIMULATION_SCHEDULE
from poker_game_expresso import BlindLevel, HandState
from tournament import TournamentState
from actions import legal_actions


def end_by_folds(tournament: TournamentState) -> None:
    tournament.act("FOLD")
    tournament.act("FOLD")


class BlindScheduleTests(unittest.TestCase):
    def test_short_heads_up_big_blind_below_small_blind_has_no_phantom_call(self):
        hand = HandState.start({0: 25, 1: 0.3}, 0, random.Random(3))
        # No opponent can act, and the SB has already covered the BB's all-in.
        # The engine runs out the board without manufacturing a call decision.
        self.assertTrue(hand.terminal)
        self.assertTrue(hand.showdown)
        self.assertEqual(hand.to_call(hand.players[0]), 0)
        self.assertEqual(legal_actions(hand), ())
        self.assertEqual([event.action for event in hand.history], ["BLIND", "BLIND"])
        self.assertAlmostEqual(sum(p.stack for p in hand.players.values()), 25.3)
        self.assertGreaterEqual(hand.awards[0], 0.2)

    def test_lone_small_blind_calls_actual_short_big_blind_amount(self):
        hand = HandState.start({0: 55, 1: 20}, 0, random.Random(3), BlindLevel(16, 32))
        self.assertEqual(hand.current_player, 0)
        self.assertEqual(hand.highest, 32)
        self.assertEqual(hand.to_call(), 4)
        self.assertFalse(hand.can_raise())
        hand.act("CALL")
        self.assertTrue(hand.terminal)
        self.assertEqual(hand.history[-1].amount_to, 20)
        self.assertEqual(hand.history[-1].amount_added, 4)
        self.assertAlmostEqual(sum(p.stack for p in hand.players.values()), 75)

    def test_two_actionable_players_still_face_full_nominal_big_blind(self):
        hand = HandState.start({0: 25, 1: 25, 2: 0.3}, 0, random.Random(3))
        self.assertEqual(hand.current_player, 0)
        self.assertEqual(hand.to_call(), 1)
        hand.act("CALL")
        self.assertEqual(hand.current_player, 1)
        self.assertEqual(hand.to_call(), 0.5)
        self.assertTrue(hand.can_raise())
        hand.act("CALL")
        self.assertEqual(hand.street, "FLOP")
        self.assertEqual(hand.players[0].contribution, 1)
        self.assertEqual(hand.players[1].contribution, 1)
        self.assertEqual(hand.players[2].contribution, 0.3)
        hand.assert_invariants()

    def test_simulation_preset_boundaries_and_last_level_held(self):
        for hand, index, expected in ((1, 0, (0.5, 1)), (10, 0, (0.5, 1)), (11, 1, (1, 2)),
                                      (20, 1, (1, 2)), (21, 2, (2, 4)), (31, 3, (4, 8)),
                                      (41, 4, (8, 16)), (51, 5, (16, 32)), (1000, 5, (16, 32))):
            with self.subTest(hand=hand):
                actual_index, blinds = DEFAULT_SIMULATION_SCHEDULE.for_hand(hand)
                self.assertEqual(actual_index, index)
                self.assertEqual((blinds.small, blinds.big), expected)

    def test_configuration_rejects_missing_unsorted_and_decreasing_levels(self):
        first = BlindStage(1, BlindLevel())
        invalid = ((), [first], (BlindStage(2, BlindLevel()),),
                   (first, BlindStage(1, BlindLevel(1, 2))),
                   (first, BlindStage(3, BlindLevel())),
                   (first, BlindStage(3, BlindLevel(0.25, 2))))
        for stages in invalid:
            with self.subTest(stages=stages):
                with self.assertRaises(ValueError):
                    BlindSchedule(stages)
        for first_hand in (0, -1, 1.5, True):
            with self.subTest(first_hand=first_hand):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    BlindStage(first_hand, BlindLevel())
        for hand_number in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                DEFAULT_SIMULATION_SCHEDULE.for_hand(hand_number)
        with self.assertRaisesRegex(ValueError, "BlindLevel"):
            BlindStage(1, (0.5, 1))

    def test_schedule_and_levels_are_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_SIMULATION_SCHEDULE.stages = ()
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_SIMULATION_SCHEDULE.stages[0].first_hand = 2
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_SIMULATION_SCHEDULE.stages[0].blinds.big = 2

    def test_new_level_applies_after_settlement_without_rescaling_stacks(self):
        schedule = BlindSchedule((BlindStage(1, BlindLevel()), BlindStage(2, BlindLevel(1, 2))))
        tournament = TournamentState(rng=random.Random(3), blind_schedule=schedule)
        first = tournament.start_hand()
        self.assertEqual(first.pot, 1.5)
        self.assertEqual(first.hand_number, 1)
        self.assertEqual(first.blind_level_index, 0)
        with self.assertRaisesRegex(ValueError, "before settlement"):
            tournament.start_hand()
        self.assertEqual(tournament.blinds, BlindLevel())
        end_by_folds(tournament)
        settled = {i: p.stack for i, p in first.players.items()}
        second = tournament.start_hand()
        self.assertEqual(second.initial_stacks, settled)
        self.assertEqual(second.blinds, BlindLevel(1, 2))
        self.assertEqual(second.pot, 3)
        self.assertEqual(second.hand_number, 2)
        self.assertEqual(second.blind_level_index, 1)
        self.assertEqual(first.blinds, BlindLevel())
        self.assertEqual(first.blind_level_index, 0)
        self.assertEqual(tournament.completed[0].blinds, BlindLevel())
        self.assertEqual(sum(p.stack for p in second.players.values()) + second.pot, 75)
        self.assertEqual(tournament.total_chips, 75)
        tournament.assert_invariants()

    def test_fixed_schedule_is_an_explicit_nonprogressing_fixture(self):
        tournament = TournamentState(rng=random.Random(9), blind_schedule=BlindSchedule.fixed())
        for number in range(1, 13):
            hand = tournament.start_hand()
            self.assertEqual(hand.hand_number, number)
            self.assertEqual(hand.blind_level_index, 0)
            self.assertEqual(hand.blinds, BlindLevel())
            end_by_folds(tournament)
        self.assertFalse(tournament.terminal)
        self.assertEqual([p.stack for p in tournament.hand.players.values()], [25, 25, 25])

    def test_default_progresses_at_hand_eleven(self):
        tournament = TournamentState(rng=random.Random(9))
        for _ in range(10):
            tournament.start_hand()
            end_by_folds(tournament)
        previous = {i: p.stack for i, p in tournament.hand.players.items()}
        hand = tournament.start_hand()
        self.assertEqual(hand.blinds, BlindLevel(1, 2))
        self.assertEqual(hand.blind_level_index, 1)
        self.assertEqual(hand.initial_stacks, previous)
        self.assertEqual(tournament.total_chips, 75)

    def test_settled_chip_ev_is_zero_sum_and_detects_corrupt_initial_stacks(self):
        tournament = TournamentState(rng=random.Random(7))
        hand = tournament.start_hand()
        end_by_folds(tournament)
        self.assertAlmostEqual(sum(hand.utility(i) for i in hand.players), 0)
        hand.initial_stacks[0] += 1
        with self.assertRaisesRegex(ValueError, "zero-sum"):
            hand.assert_invariants()

    def test_only_explicit_winner_take_all_payout_is_supported(self):
        self.assertEqual(TournamentState().payout, "winner_take_all")
        for payout in ("icm", "top_two", None):
            with self.assertRaisesRegex(ValueError, "Unsupported tournament payout"):
                TournamentState(payout=payout)

    def test_hand_context_validated_and_survives_cloning(self):
        hand = HandState.start({0: 25, 1: 25}, 0, random.Random(3),
                               hand_number=11, blind_level_index=1)
        self.assertEqual((hand.clone().hand_number, hand.clone().blind_level_index), (11, 1))
        for kwargs in ({"hand_number": 0}, {"hand_number": True}, {"blind_level_index": -1}):
            with self.assertRaisesRegex(ValueError, "hand/blind context"):
                HandState.start({0: 25, 1: 25}, 0, random.Random(3), **kwargs)
        tournament = TournamentState()
        tournament.start_hand()
        tournament.hand.blind_level_index += 1
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            tournament.assert_invariants()
