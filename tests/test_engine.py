import math
import random
import unittest

from actions import apply_action, legal_actions
from blind_schedule import BlindSchedule
from poker_game_expresso import HandState
from tournament import TournamentState


def hand(stacks=(25, 25, 25), button=0):
    return HandState.start(dict(enumerate(stacks)), button, random.Random(7))


class BettingTests(unittest.TestCase):
    def test_clone_isolates_all_mutable_branch_state(self):
        from dataclasses import asdict
        original = hand()
        before = asdict(original)
        branch = original.clone()
        branch.actor.stack -= 1
        branch.actor.cards = (0, 1)
        branch.deck.pop()
        branch.initial_stacks[0] = 99
        branch.board.append(2)
        branch.pending.clear()
        branch.history.clear()
        branch.awards[0] = 5
        self.assertEqual(asdict(original), before)

    def test_blinds_order_and_streets(self):
        for stacks, first, postflop in (((25, 25, 25), 0, 1), ((25, 25), 0, 1)):
            h = hand(stacks)
            self.assertEqual(h.pot, 1.5)
            self.assertEqual(h.current_player, first)
            while h.street == "PREFLOP":
                h.act("CALL" if h.to_call() else "CHECK")
            self.assertEqual(h.street, "FLOP")
            self.assertEqual(h.current_player, postflop)
            while not h.terminal:
                h.act("CHECK")
            self.assertEqual(len(h.board), 5)
            self.assertAlmostEqual(sum(p.stack for p in h.players.values()), sum(stacks))

    def test_arbitrary_minimum_raises_and_invalid_actions(self):
        h = hand()
        with self.assertRaisesRegex(ValueError, "Cannot check"):
            h.act("CHECK")
        with self.assertRaisesRegex(ValueError, "below minimum"):
            h.act("RAISE", 1.5)
        h.act("RAISE", 2)
        self.assertEqual(h.min_raise_to, 3)
        h.act("RAISE", 3.7)
        self.assertAlmostEqual(h.min_raise_to, 5.4)
        h.act("RAISE", 5.4)
        self.assertEqual(h.current_player, 0)
        h.act("FOLD")
        h.act("FOLD")
        self.assertTrue(h.terminal)
        self.assertEqual(h.board, [])

    def test_short_all_in_does_not_reopen(self):
        h = hand((25, 3, 25))
        h.act("RAISE", 2.5)
        h.act("RAISE", 3)
        self.assertEqual(h.last_full_raise, 1.5)
        h.act("CALL")
        self.assertEqual(h.current_player, 0)
        self.assertFalse(h.can_raise())
        self.assertNotIn("ALL_IN", [a.action_id for a in legal_actions(h)])
        with self.assertRaisesRegex(ValueError, "raising rights"):
            h.act("RAISE", 25)
        h.act("CALL")
        self.assertEqual(h.street, "FLOP")

    def test_full_all_in_reopens(self):
        h = hand((25, 5, 25))
        h.act("RAISE", 2.5)
        h.act("RAISE", 5)
        h.act("CALL")
        self.assertTrue(h.can_raise())
        self.assertEqual(h.min_raise_to, 7.5)

    def test_short_all_in_preserves_unacted_raise_rights(self):
        h = hand((25, 1.4, 25))
        h.act("CALL")
        h.act("RAISE", 1.4)
        self.assertEqual(h.current_player, 2)
        self.assertTrue(h.can_raise())
        self.assertAlmostEqual(h.min_raise_to, 2.4)
        h.act("RAISE", 2.4)
        self.assertEqual(h.current_player, 0)
        self.assertTrue(h.can_raise())
        self.assertAlmostEqual(h.min_raise_to, 3.4)

    def test_short_call_and_uncalled_refund(self):
        h = hand((1, 25))
        h.act("CALL")
        self.assertTrue(h.terminal)
        self.assertAlmostEqual(sum(p.stack for p in h.players.values()), 26)
        h = hand((25, 2, 3))
        h.act("RAISE", 25)
        h.act("CALL")
        h.act("CALL")
        self.assertTrue(h.terminal)
        self.assertGreaterEqual(h.players[0].stack, 22)
        self.assertAlmostEqual(sum(h.awards.values()), 30)

    def test_sizing_masks_and_deduplication(self):
        h = hand()
        a = legal_actions(h)
        self.assertEqual([x.amount_to for x in a if x.category == "RAISE"], [2, 2.5, 3, 4, 25])
        while h.street == "PREFLOP":
            h.act("CALL" if h.to_call() else "CHECK")
        a = legal_actions(h)
        self.assertAlmostEqual(next(x.amount_to for x in a if x.action_id == "BET_RAISE_50P"), 1.5)
        amounts = [x.amount_to for x in a if x.category == "RAISE"]
        self.assertEqual(len(amounts), len(set(amounts)))
        with self.assertRaisesRegex(ValueError, "Illegal canonical"):
            apply_action(h, "RAISE")

    def test_multiple_side_pots_folded_chips_and_ties(self):
        h = hand((5, 10, 20))
        h.act("RAISE", 5)
        h.act("RAISE", 10)
        h.act("CALL")
        self.assertTrue(h.terminal)
        self.assertAlmostEqual(sum(h.awards.values()), 25)
        h = hand()
        h.act("RAISE", 3)
        h.act("CALL")
        h.act("RAISE", 8)
        h.act("FOLD")
        h.act("CALL")
        while not h.terminal:
            h.act("CHECK")
        self.assertEqual(h.awards[0], 0)
        self.assertAlmostEqual(sum(p.stack for p in h.players.values()), 75)
        # Royal flush on board splits all matched contributions.
        board = [32, 36, 40, 44, 48]
        private = [0, 1, 4, 5, 8, 9]
        deck = [c for c in range(52) if c not in board + private] + list(reversed(board)) + private
        h = HandState.start({0: 1, 1: 1, 2: 1}, 0, random.Random(1), deck=deck)
        h.act("CALL")
        h.act("CALL")
        self.assertTrue(h.terminal)
        self.assertEqual([p.stack for p in h.players.values()], [1, 1, 1])

    def test_exact_side_pot_awards(self):
        private = [40, 41, 44, 45, 48, 49]  # Short AA wins main; middle KK wins side; QQ receives uncalled chips.
        board = [0, 5, 22, 31, 36]
        deck = [c for c in range(52) if c not in private + board] + list(reversed(board)) + list(reversed(private))
        h = HandState.start({0: 20, 1: 10, 2: 5}, 0, random.Random(0), deck=deck)
        h.act("RAISE", 20)
        h.act("CALL")
        h.act("CALL")
        self.assertEqual(h.awards, {0: 10, 1: 10, 2: 15})
        self.assertEqual([p.stack for p in h.players.values()], [10, 10, 15])

    def test_postflop_minimum_and_checked_short_all_in(self):
        h = hand()
        while h.street == "PREFLOP":
            h.act("CALL" if h.to_call() else "CHECK")
        with self.assertRaisesRegex(ValueError, "below minimum"):
            h.act("RAISE", 0.5)
        h.act("RAISE", 1)
        self.assertEqual(h.min_raise_to, 2)
        h = hand((25, 25, 1.4))
        while h.street == "PREFLOP":
            h.act("CALL" if h.to_call() else "CHECK")
        h.act("CHECK")
        h.act("RAISE", h.actor.stack)
        h.act("CALL")
        self.assertFalse(h.can_raise())
        h.act("CALL")
        self.assertEqual(h.street, "TURN")

    def test_unlimited_full_raises_and_all_button_positions(self):
        h = hand()
        for _ in range(9):
            h.act("RAISE", h.min_raise_to)
        self.assertEqual(len([e for e in h.history if e.action == "RAISE"]), 9)
        for count in (2, 3):
            for button in range(count):
                h = hand((25,) * count, button)
                self.assertEqual(h.current_player, button)
                self.assertEqual(h.players[button].position, "SB" if count == 2 else "BTN")
                while h.street == "PREFLOP":
                    h.act("CALL" if h.to_call() else "CHECK")
                self.assertEqual(h.current_player, (button + 1) % count)

    def test_invalid_deck_and_chips(self):
        with self.assertRaisesRegex(ValueError, "Deck"):
            HandState.start({0: 25, 1: 25}, 0, random.Random(1), deck=[0] * 52)
        with self.assertRaisesRegex(ValueError, "positive"):
            hand((25, float("nan")))
        h = hand()
        h.pot += 1
        with self.assertRaisesRegex(ValueError, "conservation"):
            h.assert_invariants()

    def test_small_seeded_games_conserve_chips(self):
        rng = random.Random(42)
        for _ in range(30):
            h = HandState.start({i: rng.uniform(0.1, 25) for i in range(rng.choice((2, 3)))}, 0, rng)
            for _ in range(150):
                if h.terminal:
                    break
                rng.choice(legal_actions(h)).apply(h)
            self.assertTrue(h.terminal)
            h.assert_invariants()


class TournamentTests(unittest.TestCase):
    def test_heads_up_transition_advances_big_blind_for_every_eliminated_position(self):
        board = [8, 17, 26, 35, 40]
        for button in range(3):
            for eliminated in range(3):
                with self.subTest(button=button, eliminated=eliminated):
                    survivors = [i for i in range(3) if i != eliminated]
                    holes = {eliminated: [0, 1], survivors[0]: [48, 49], survivors[1]: [44, 45]}
                    private = [c for i in range(3) for c in holes[i]]
                    deck = ([c for c in range(52) if c not in board + private]
                            + list(reversed(board)) + list(reversed(private)))
                    t = TournamentState({i: 1.0 if i == eliminated else 25.0 for i in range(3)},
                                        button=button, rng=random.Random(0))
                    h = t.start_hand(deck=deck)
                    previous_bb = next(i for i, p in h.players.items() if p.position == "BB")
                    while not h.terminal:
                        h.act("CALL" if h.to_call() else "CHECK")
                    self.assertEqual(h.players[eliminated].stack, 0)
                    self.assertEqual(t.active, tuple(survivors))
                    next_bb = next((previous_bb + step) % 3 for step in (1, 2, 3)
                                   if (previous_bb + step) % 3 in survivors)
                    hu = t.start_hand()
                    self.assertEqual(set(hu.players), set(survivors))
                    self.assertEqual(hu.players[next_bb].position, "BB")
                    self.assertEqual(hu.players[t.button].position, "SB")
                    self.assertEqual(hu.current_player, t.button)
                    if previous_bb in survivors:
                        self.assertNotEqual(next_bb, previous_bb)
                    while hu.street == "PREFLOP":
                        hu.act("CALL" if hu.to_call() else "CHECK")
                    self.assertEqual(hu.current_player, next_bb)
                    while not hu.terminal:
                        hu.act("CHECK")
                    following = t.start_hand()
                    self.assertEqual(following.button, next_bb)
                    self.assertEqual(following.players[t.button].position, "SB")
                    t.assert_invariants()

    def test_rotation_persistence(self):
        t = TournamentState(rng=random.Random(1))
        t.start_hand()
        t.act("FOLD")
        t.act("FOLD")
        settled = {i: p.stack for i, p in t.hand.players.items()}
        t.start_hand()
        self.assertEqual(t.button, 1)
        self.assertEqual(t.hand_number, 2)
        self.assertEqual(t.hand.initial_stacks, settled)
        self.assertEqual(t.hand.players[1].position, "BTN")

    def test_elimination_and_heads_up(self):
        t = TournamentState({0: 1, 1: 25, 2: 25}, rng=random.Random(3))
        t.start_hand()
        while not t.hand.terminal:
            t.act("ALL_IN" if "ALL_IN" in [a.action_id for a in legal_actions(t.hand)] else "CALL")
        self.assertLess(len(t.active), 3)
        if not t.terminal:
            t.start_hand()
            self.assertEqual(len(t.hand.players), 2)
            self.assertEqual(t.hand.actor.position, "SB")
        t.assert_invariants()

    def test_fixed_blinds_allow_nonterminating_fold_cycle(self):
        t = TournamentState(rng=random.Random(9), blind_schedule=BlindSchedule.fixed())
        for _ in range(3):
            t.start_hand()
            t.act("FOLD")
            t.act("FOLD")
        self.assertFalse(t.terminal)
        self.assertEqual([p.stack for p in t.hand.players.values()], [25, 25, 25])

    def test_tournament_completion(self):
        t = TournamentState(rng=random.Random(9))
        t.start_hand()
        rng = random.Random(2)
        for _ in range(300):
            if t.terminal:
                break
            if t.hand.terminal:
                t.start_hand()
            else:
                actions = legal_actions(t.hand)
                chosen = next((a for a in actions if a.action_id == "ALL_IN"), None)
                (chosen or rng.choice(actions)).apply(t.hand)
            t.assert_invariants()
        self.assertTrue(t.terminal)
        self.assertAlmostEqual(t.hand.players[t.winner].stack, 75)
        self.assertAlmostEqual(sum(t.utility(i) for i in t.original_players), 0)
        with self.assertRaisesRegex(ValueError, "already won"):
            t.start_hand()


if __name__ == "__main__":
    unittest.main()
