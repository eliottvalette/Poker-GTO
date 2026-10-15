"""Root ranges and continuation contracts prevent contradictory neural inputs."""
from itertools import permutations
from pathlib import Path
import tempfile
import unittest

import torch

from actions import ACTION_IDS, legal_actions
from hybrid.knowledge import calculate, game_contract, root_supervision
from hybrid.ranges import HandRange, JointRanges
from hybrid.solver_inputs import encode_roots, range_vector, RIVER_CONTINUATION
from hybrid.state import ComputeBudget, instantiate
from infoset import observe
from test_hybrid_river import fixture


class SolverInputTests(unittest.TestCase):
    def test_joint_suit_and_card_order_invariance(self):
        from hybrid.solver_experiments import bounded_roots
        roots = [fixture(count) for count in (2, 3)]
        roots += [next(bounded_roots(count, "train")) for count in (2, 3)]
        for root, ranges in roots:
            deal = ranges.enumerate()[0]
            state = instantiate(root, deal.hands)
            contract = game_contract(root, ranges, ComputeBudget(), "train")
            obs = observe(state)
            expected = range_vector(obs, contract)
            for perm in permutations(range(4)):
                card = lambda c: c // 4 * 4 + perm[c % 4]
                changed = state.clone()
                changed.board = [card(c) for c in state.board]
                for p in changed.players:
                    changed.players[p].cards = tuple(reversed([card(c) for c in state.players[p].cards]))
                transformed = JointRanges({p: HandRange({tuple(sorted(card(c) for c in h)): w for h, w in r.weights.items()})
                                           for p, r in ranges.ranges.items()}, tuple(changed.board))
                c = game_contract(changed, transformed, ComputeBudget(), "train")
                actual = range_vector(observe(changed), c)
                self.assertTrue(torch.equal(expected, actual), perm)

    def test_seat_renaming_preserves_context_encoding(self):
        from dataclasses import replace
        for count in (2, 3):
            root, ranges = fixture(count)
            state = instantiate(root, ranges.enumerate()[0].hands)
            contract = game_contract(state, ranges, ComputeBudget(), "test")
            expected = encode_roots([observe(state)], [contract], (RIVER_CONTINUATION,))
            mapping = {p: 17 + 3 * p for p in state.players}
            state.players = {mapping[p]: replace(v, player_id=mapping[p]) for p, v in state.players.items()}
            state.initial_stacks = {mapping[p]: v for p, v in state.initial_stacks.items()}
            state.current_player = mapping[state.current_player]
            state.button = mapping[state.button]
            state.pending = {mapping[p] for p in state.pending}
            state.history = [replace(event, player_id=mapping[event.player_id]) for event in state.history]
            renamed = JointRanges({mapping[p]: r for p, r in ranges.ranges.items()}, tuple(state.board))
            actual = encode_roots([observe(state)], [game_contract(state, renamed, ComputeBudget(), "test")], (RIVER_CONTINUATION,))
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_range_inputs_differ_without_changing_private_observation(self):
        root, ranges = fixture(2)
        obs = observe(instantiate(root, ranges.enumerate()[0].hands))
        other = next(p for p in root.players if p != obs.hero)
        altered = JointRanges({p: HandRange({h: (1 / w if p == other else w) for h, w in r.weights.items()})
                               for p, r in ranges.ranges.items()}, tuple(root.board))
        a = encode_roots([obs], [game_contract(root, ranges, ComputeBudget(), "train")], (RIVER_CONTINUATION,))
        b = encode_roots([obs], [game_contract(root, altered, ComputeBudget(), "train")], (RIVER_CONTINUATION,))
        for key in a:
            self.assertEqual(torch.equal(a[key], b[key]), key != "solving_context")
        factors = a["solving_context"][0, :3978].reshape(3, 1326)
        self.assertAlmostEqual(float(factors[0].sum()), 1.)
        self.assertEqual(int((factors[0] > 0).sum()), 2)

    def test_root_advantage_matches_independent_call_payoff(self):
        root, ranges = fixture(2)
        # A facing-jam decision has exact deterministic fold and showdown payoffs.
        next(a for a in legal_actions(root) if a.action_id == "ALL_IN").apply(root)
        ranges = JointRanges(dict(ranges.ranges), tuple(root.board))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game.json"
            calculate(path, root, ranges, ComputeBudget(iterations=100), split="train")
            _, arows = root_supervision(path, "advantage")
            _, brows = root_supervision(path, "strategy")
            for a, b in zip(arows, brows):
                conditional = [d for d in ranges.enumerate() if d.hands[root.current_player] == tuple(sorted(a.state.cards[:2]))]
                mass = sum(d.probability for d in conditional)
                ev = {}
                for action in legal_actions(root):
                    value = 0.
                    for deal in conditional:
                        state = instantiate(root, deal.hands)
                        action.apply(state)
                        self.assertTrue(state.terminal)
                        value += deal.probability / mass * state.utility(root.current_player)
                    ev[action.action_id] = value
                baseline = sum(b.target[ACTION_IDS.index(k)] * v for k, v in ev.items())
                for action, value in ev.items():
                    self.assertAlmostEqual(a.target[ACTION_IDS.index(action)], (value - baseline) / root.blinds.big, places=12)
                self.assertAlmostEqual(sum(x * y for x, y in zip(a.target, b.target)), 0., places=12)

    def test_same_observation_opposite_ranges_change_exact_call_decision(self):
        from itertools import combinations
        from utils import rank7
        root, _ = fixture(2)
        next(a for a in legal_actions(root) if a.action_id == "ALL_IN").apply(root)
        actor = root.current_player
        other = next(p for p in root.players if p != actor)
        hands = list(combinations([c for c in range(52) if c not in root.board], 2))
        selected = None
        for hero in hands:
            score = rank7((*hero, *root.board))
            better = [h for h in hands if not set(h).intersection(hero) and rank7((*h, *root.board)) > score]
            worse = [h for h in hands if not set(h).intersection(hero) and rank7((*h, *root.board)) < score]
            if better and worse:
                selected = hero, better[0], worse[0]
                break
        self.assertIsNotNone(selected)
        hero, strong, weak = selected
        observations, strategies, advantages = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            for index, weights in enumerate(((.99, .01), (.01, .99))):
                ranges = JointRanges({actor: HandRange({hero: 1}), other: HandRange(dict(zip((strong, weak), weights)))}, tuple(root.board))
                path = Path(directory) / f"range-{index}.json"
                calculate(path, root, ranges, ComputeBudget(iterations=200), split="train")
                _, a = root_supervision(path, "advantage")
                _, b = root_supervision(path, "strategy")
                observations.append(a[0].state)
                strategies.append(b[0].target)
                advantages.append(a[0].target)
        self.assertEqual(observations[0], observations[1])
        call = ACTION_IDS.index("CALL")
        self.assertLess(strategies[0][call], .01)
        self.assertGreater(strategies[1][call], .99)
        self.assertLess(advantages[0][call], 0)
        self.assertGreater(advantages[1][call], 0)

    def test_reject_unknown_profile_and_downstream_query(self):
        root, ranges = fixture(2)
        state = instantiate(root, ranges.enumerate()[0].hands)
        contract = game_contract(root, ranges, ComputeBudget(), "train")
        with self.assertRaisesRegex(ValueError, "Unsupported continuation"):
            encode_roots([observe(state)], [contract], ("unknown",))
        state.act("CHECK")
        with self.assertRaisesRegex(ValueError, "public decision root"):
            range_vector(observe(state), contract)

    def test_continuation_and_depth_are_explicit_inputs(self):
        from hybrid.solver_experiments import bounded_roots
        from copy import deepcopy
        root, ranges = next(bounded_roots(3, "train"))
        obs = observe(instantiate(root, ranges.enumerate()[0].hands))
        contract = game_contract(root, ranges, ComputeBudget(max_depth=1), "train")
        changed = deepcopy(contract)
        changed["continuation"] = "another-declared-profile"
        changed["forest"]["depth"] = 2
        inputs = encode_roots([obs, obs], [contract, changed], (contract["continuation"], changed["continuation"]))
        self.assertTrue(torch.equal(inputs["solving_context"][0, :3978], inputs["solving_context"][1, :3978]))
        self.assertFalse(torch.equal(inputs["solving_context"][0, 3978:], inputs["solving_context"][1, 3978:]))

    def test_cold_or_wrong_track_is_not_silently_used(self):
        from hybrid.distillation import SolverStudents
        root, ranges = fixture(3)
        obs = observe(instantiate(root, ranges.enumerate()[0].hands))
        contract = game_contract(root, ranges, ComputeBudget(), "test")
        bank = SolverStudents(2).models["strategy"]
        with self.assertRaisesRegex(ValueError, "track mismatch"):
            bank.probabilities(obs, contract)
        with self.assertRaisesRegex(ValueError, "Untrained"):
            SolverStudents(3).models["strategy"].probabilities(obs, contract)

    def test_hidden_deal_and_future_deck_not_in_context(self):
        root, ranges = fixture(2)
        state = instantiate(root, ranges.enumerate()[0].hands)
        obs = observe(state)
        before = game_contract(state, ranges, ComputeBudget(), "test")
        other = next(p for p in state.players if p != obs.hero)
        state.players[other].cards = (50, 51)
        state.deck.reverse()
        after = game_contract(state, ranges, ComputeBudget(), "test")
        self.assertEqual(before, after)
        self.assertEqual(obs, observe(state))


if __name__ == "__main__":
    unittest.main()
