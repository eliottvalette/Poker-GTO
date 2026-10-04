import json
import random
import tempfile
import unittest
from pathlib import Path
from actions import ACTION_IDS
from cfr_solver import ExternalSamplingMCCFR, Traversal, TraversalBudgetExceeded, regret_matching
from evaluation import subgame_best_response
from infoset import observe, NUMERIC_NAMES
from poker_game_expresso import HandState
from policy import TabularAveragePolicy


def river():
    h = HandState.start({0: 2.0, 1: 2.0}, 0, random.Random(3))
    while h.street != "RIVER":
        h.act("CALL" if h.to_call() else "CHECK")
    return h


def uniform(obs):
    return regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)


class SolverTests(unittest.TestCase):
    def test_masked_regret_matching(self):
        obs = observe(river())
        r = [0.0] * len(ACTION_IDS)
        r[ACTION_IDS.index("CHECK")] = 2
        r[ACTION_IDS.index("ALL_IN")] = 6
        p = regret_matching(r, obs.legal_mask)
        self.assertEqual(p[ACTION_IDS.index("ALL_IN")], 0.75)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            regret_matching([float("nan")] * len(r), obs.legal_mask)

    def test_future_traverser_nodes_generate_regrets(self):
        h = HandState.start({0: 2, 1: 2}, 0, random.Random(3))
        samples = []
        walk = Traversal(uniform, random.Random(4), 10000)
        walk.regrets(h, 0, lambda o, t, w: samples.append((o, t)))
        self.assertGreater(len(samples), 1)
        self.assertGreater(len({o.street for o, _ in samples}), 1)
        for o, target in samples:
            self.assertAlmostEqual(sum(p * r for p, r in zip(uniform(o), target)), 0)
            self.assertTrue(all(r == 0 for r, m in zip(target, o.legal_mask) if not m))

    def test_sample_value_against_exact_tree(self):
        h = river()
        exact = subgame_best_response([(1.0, h)], uniform, 0)
        values = [Traversal(uniform, random.Random(seed)).regrets(h.clone(), 0, lambda *_: None)
                  for seed in range(100)]
        self.assertAlmostEqual(sum(values) / len(values), exact["policy_value_bb"], delta=0.3)
        self.assertGreaterEqual(exact["best_response_gain_bb"], 0)

    def test_average_full_tree_reference_and_importance(self):
        h = river()
        expected = {}
        def full(state, reach):
            if state.terminal:
                return
            o = observe(state)
            p = uniform(o)
            if state.current_player == 0:
                expected[o.key()] = tuple(reach * v for v in p)
            from actions import legal_actions
            for a in legal_actions(state):
                full(Traversal.child(state, a), reach * p[ACTION_IDS.index(a.action_id)]
                     if state.current_player == 0 else reach)
        full(h, 1.0)
        sums = {}
        for seed in range(120):
            def add(o, target, weight):
                row = sums.setdefault(o.key(), [0.0] * len(ACTION_IDS))
                for i, v in enumerate(target):
                    row[i] += weight * v / 120
            Traversal(uniform, random.Random(seed)).average(h.clone(), 0, add)
        self.assertEqual(set(sums), set(expected))
        for key in sums:
            for x, y in zip(sums[key], expected[key]):
                self.assertAlmostEqual(x, y, delta=0.15)

    def test_atomic_budget_failure_and_policy_roundtrip(self):
        solver = ExternalSamplingMCCFR(max_nodes=1)
        with self.assertRaises(TraversalBudgetExceeded):
            solver.run_iteration(lambda _: river(), (0, 1))
        self.assertFalse(solver.regret_sum)
        self.assertEqual(solver.iteration, 0)
        solver = ExternalSamplingMCCFR(seed=1)
        for _ in range(2):
            solver.run_iteration(lambda _: river(), (0, 1))
        policy = TabularAveragePolicy.from_solver(solver, "hand_chip_delta")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            policy.save(path)
            loaded = TabularAveragePolicy.load(path)
            self.assertEqual(loaded.query(observe(river())), policy.query(observe(river())))
            path.write_text('{"100": {"policy": [1, 255]}}')
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                TabularAveragePolicy.load(path)

    def test_tournament_traversal_stays_in_current_hand(self):
        from dataclasses import asdict
        from tournament import TournamentState
        t = TournamentState({0: 2.0, 1: 2.0}, rng=random.Random(3))
        t.start_hand()
        baseline = asdict(t.hand)
        samples = []
        value = Traversal(uniform, random.Random(1), 1000, 100).regrets(t, 0, lambda o,r,w: samples.append(o))
        self.assertGreater(len(samples), 1)
        self.assertTrue(all(o.objective == "hand_chip_delta" for o in samples))
        self.assertTrue(all(o.numeric[NUMERIC_NAMES.index("hand_number")] == 1 / 25 for o in samples))
        self.assertGreaterEqual(value, -2)
        self.assertLessEqual(value, 2)
        self.assertTrue(all(len(json.loads(o.recall)) == 1 for o in samples))
        self.assertEqual(asdict(t.hand), baseline)
        self.assertEqual(t.hand_number, 1)
        self.assertFalse(t.completed)

    def test_previous_hand_cards_do_not_enter_current_hand_learning_state(self):
        from tournament import TournamentState
        from utils import rank7
        t = TournamentState(rng=random.Random(7))
        t.start_hand()
        while not t.hand.terminal:
            t.hand.act("CALL" if t.hand.to_call() else "CHECK")
        self.assertTrue(t.hand.showdown)
        t.start_hand()
        o = observe(t)
        previous = t.completed[0]
        records = json.loads(o.recall)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["shown_cards"], {})
        self.assertIsNone(records[0]["final_stacks"])
        stack_events = [e for e in o.history if e[0] == 1 / 25 and e[4] == 1 and e[-1] == 0]
        self.assertEqual(len(stack_events), 3)
        changed = t.clone()
        p = changed.completed[0].players[0]
        before = rank7((*p.cards, *previous.board))
        replacement = next(c for c in changed.completed[0].deck if c // 4 == p.cards[0] // 4
                           and rank7((c, p.cards[1], *previous.board)) == before)
        idx = changed.completed[0].deck.index(replacement)
        changed.completed[0].deck[idx] = p.cards[0]
        p.cards = (replacement, p.cards[1])
        changed.completed[0].assert_invariants()
        other = observe(changed)
        self.assertEqual(o.key(), other.key())
        self.assertEqual(o.history, other.history)
        self.assertEqual(o.numeric, other.numeric)
        self.assertEqual(o.cards, other.cards)

    def test_exact_cards_history_and_no_hidden_leak(self):
        h = river()
        obs = observe(h)
        self.assertEqual(len(obs.numeric), len(NUMERIC_NAMES))
        self.assertEqual(obs.cards[:2], h.actor.cards)
        self.assertGreater(len(obs.history), 5)
        changed = h.clone()
        opponent = changed.players[changed.next(changed.current_player)]
        opponent.cards = tuple(changed.deck[:2])
        self.assertEqual(observe(changed).key(), obs.key())
        changed = h.clone()
        changed.history.pop()
        self.assertNotEqual(observe(changed).key(), obs.key())
        changed = h.clone()
        changed.actor.stack -= 0.001
        self.assertNotEqual(observe(changed).key(), obs.key())


if __name__ == "__main__":
    unittest.main()
