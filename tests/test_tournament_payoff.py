"""Exact payoff pruning never treats live all-in chips as elimination."""
import random
import unittest

from actions import legal_actions
from tournament import TournamentState


def short_player_tournament() -> TournamentState:
    private = [40, 41, 48, 49, 44, 45]
    board = [0, 5, 22, 31, 36]
    deck = ([card for card in range(52) if card not in private + board]
            + list(reversed(board)) + list(reversed(private)))
    tournament = TournamentState({0: 1.0, 1: 25.0, 2: 25.0}, rng=random.Random(3))
    tournament.start_hand(deck=deck)
    return tournament


def finish_passively(tournament: TournamentState) -> None:
    while not tournament.hand.terminal:
        tournament.hand.act("CALL" if tournament.hand.to_call() else "CHECK")


class SettledTournamentPayoffTests(unittest.TestCase):
    def test_live_all_in_is_not_a_settled_loss(self):
        tournament = short_player_tournament()
        tournament.act("CALL")
        self.assertEqual(tournament.hand.players[0].stack, 0)
        self.assertFalse(tournament.hand.players[0].folded)
        self.assertFalse(tournament.hand.terminal)
        self.assertIsNone(tournament.settled_utility(0))

    def test_folded_player_with_chips_still_has_unresolved_tournament_payoff(self):
        tournament = short_player_tournament()
        tournament.act("FOLD")
        self.assertTrue(tournament.hand.players[0].folded)
        self.assertGreater(tournament.hand.players[0].stack, 0)
        self.assertIsNone(tournament.settled_utility(0))

    def test_eliminated_payoff_is_known_before_heads_up_finishes(self):
        tournament = short_player_tournament()
        finish_passively(tournament)
        self.assertFalse(tournament.terminal)
        self.assertEqual(tournament.hand.players[0].stack, 0)
        self.assertEqual(tournament.settled_utility(0), -1 / 3)
        self.assertIsNone(tournament.settled_utility(1))
        self.assertIsNone(tournament.settled_utility(2))
        with self.assertRaisesRegex(ValueError, "terminal tournament"):
            tournament.utility(0)
        tournament.start_hand()
        self.assertNotIn(0, tournament.hand.players)
        self.assertEqual(tournament.settled_utility(0), -1 / 3)
        self.assertEqual(len(tournament.active), 2)

    def test_pruned_loss_equals_eventual_terminal_payoff(self):
        tournament = short_player_tournament()
        finish_passively(tournament)
        predicted = tournament.settled_utility(0)
        for _ in range(100):
            if tournament.terminal:
                break
            if tournament.hand.terminal:
                tournament.start_hand()
            else:
                actions = legal_actions(tournament.hand)
                action = next((a for a in actions if a.action_id == "ALL_IN"), None)
                if action is None:
                    action = next((a for a in actions if a.action_id == "CALL"), actions[0])
                tournament.act(action.action_id)
        self.assertTrue(tournament.terminal)
        self.assertEqual(predicted, tournament.utility(0))
        for player in tournament.original_players:
            self.assertEqual(tournament.settled_utility(player), tournament.utility(player))
        self.assertAlmostEqual(sum(tournament.utility(p) for p in tournament.original_players), 0)
        tournament.assert_invariants()

    def test_initial_elimination_retains_original_three_player_utility_baseline(self):
        tournament = TournamentState({0: 0.0, 1: 37.5, 2: 37.5}, button=1)
        self.assertEqual(tournament.settled_utility(0), -1 / 3)
        self.assertIsNone(tournament.settled_utility(1))
        self.assertIsNone(tournament.settled_utility(2))
        tournament.start_hand()
        self.assertEqual(tournament.settled_utility(0), -1 / 3)

    def test_terminal_winner_and_loser_utilities(self):
        tournament = TournamentState({0: 0.0, 1: 75.0, 2: 0.0}, button=1)
        self.assertTrue(tournament.terminal)
        self.assertAlmostEqual(tournament.settled_utility(1), 2 / 3)
        self.assertEqual(tournament.settled_utility(0), -1 / 3)
        self.assertEqual(tournament.settled_utility(2), -1 / 3)

    def test_payoff_query_rejects_unknown_player_and_does_not_mutate_state(self):
        tournament = short_player_tournament()
        finish_passively(tournament)
        hand = tournament.hand
        stacks, board, history = dict(tournament.stacks), list(hand.board), list(hand.history)
        rng_state = tournament.rng.getstate()
        self.assertEqual(tournament.settled_utility(0), -1 / 3)
        with self.assertRaisesRegex(ValueError, "original player: 99"):
            tournament.settled_utility(99)
        self.assertIs(tournament.hand, hand)
        self.assertEqual(tournament.stacks, stacks)
        self.assertEqual(hand.board, board)
        self.assertEqual(hand.history, history)
        self.assertEqual(tournament.rng.getstate(), rng_state)
        self.assertEqual(tournament.hand_number, 1)
        self.assertEqual(tournament.completed, [])
