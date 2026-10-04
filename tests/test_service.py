import random
import unittest
from server import PokerService
from actions import legal_actions
from tournament import TournamentState


class ServiceTests(unittest.TestCase):
    def test_public_contract_and_private_cards(self):
        service = PokerService()
        v = service.new(7)
        self.assertEqual(v["version"], 2)
        self.assertEqual(v["policy"]["status"], "unavailable")
        self.assertIsNone(v["policy"]["probabilities"])
        if not v["hand_terminal"]:
            self.assertEqual(v["hero_result_bb"], 0)
            self.assertEqual(v["hand_results_bb"], {})
        for p in v["players"]:
            if p["player_id"] != v["hero"] and not v["hand_terminal"]:
                self.assertEqual(p["cards"], [])
        self.assertAlmostEqual(sum(p["stack_bb"] for p in v["players"]) + v["pot_bb"], 75)
        session = service.sessions[v["session_id"]]
        expected = [a.action_id for a in legal_actions(session.tournament.hand)]
        self.assertEqual([a["action_id"] for a in v["legal_actions"]], expected)

    def test_actions_revision_persistence_and_error_atomicity(self):
        service = PokerService()
        v = service.new(7, opponents="calling_station")
        session = service.sessions[v["session_id"]]
        with self.assertRaisesRegex(ValueError, "Illegal canonical"):
            service.command(v["session_id"], 0, "action", "BOGUS")
        self.assertEqual(service.view(v["session_id"], session), v)
        with self.assertRaisesRegex(ValueError, "Stale revision"):
            service.command(v["session_id"], 9, "next")
        while not v["hand_terminal"]:
            action = next((a for a in v["legal_actions"] if a["action_id"] == "CHECK"), v["legal_actions"][0])
            v = service.command(v["session_id"], v["revision"], "action", action["action_id"])
        settled = {p["player_id"]: p["stack_bb"] for p in v["players"]}
        before = v["revision"]
        v = service.command(v["session_id"], before, "next")
        self.assertEqual(v["hand_number"], 2)
        self.assertEqual(session.tournament.hand.initial_stacks, settled)
        service.close(v["session_id"])
        with self.assertRaisesRegex(KeyError, "Unknown session"):
            service.command(v["session_id"], v["revision"], "action", "CALL")

    def test_whole_tournament_via_public_actions(self):
        service = PokerService()
        v = service.new(7, opponents="shove_fold")
        for _ in range(100):
            if v["tournament_terminal"]:
                break
            if v["hand_terminal"]:
                v = service.command(v["session_id"], v["revision"], "next")
            else:
                names = [a["action_id"] for a in v["legal_actions"]]
                action = "ALL_IN" if "ALL_IN" in names else "CALL" if "CALL" in names else names[0]
                v = service.command(v["session_id"], v["revision"], "action", action)
        self.assertTrue(v["tournament_terminal"])
        self.assertIsNotNone(v["winner"])
        self.assertEqual(len(v["active_players"]), 1)
        self.assertAlmostEqual(max(p["stack_bb"] for p in v["players"]), 75)

    def test_fold_winner_does_not_reveal_private_cards(self):
        service = PokerService()
        v = service.new(7, hero=0, opponents="nit")
        v = service.command(v["session_id"], v["revision"], "action", "FOLD")
        self.assertTrue(v["hand_terminal"])
        session = service.sessions[v["session_id"]]
        self.assertFalse(session.tournament.hand.showdown)
        for p in v["players"]:
            if p["player_id"] != v["hero"]:
                self.assertEqual(p["cards"], [])

    def test_configuration_failures(self):
        service = PokerService(max_sessions=1)
        with self.assertRaisesRegex(ValueError, "Invalid table"):
            service.new("7")
        service.new(7)
        with self.assertRaisesRegex(ValueError, "limit reached"):
            service.new(8)
