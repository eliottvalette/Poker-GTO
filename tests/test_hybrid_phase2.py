from pathlib import Path
import random
import tempfile
import unittest

from hybrid.experiments import small_game
from hybrid.learning import Learner
from hybrid.policy_source import UniformLegalPolicy
from hybrid.ranges import HandRange, JointRanges
from hybrid.replay import ProtectedReplay
from hybrid.search import search
from hybrid.state import ComputeBudget
from hybrid.value_features import suit_aware_features
from infoset import observe
from ml.memory import TrainingSample
from poker_game_expresso import HandState
from training.metrics import Coverage


class Phase2Tests(unittest.TestCase):
    def test_cached_proposals_match_original_joint_rejection_stream(self):
        joint = JointRanges({p: HandRange.uniform() for p in range(3)})
        original_rng, cached_rng = random.Random(718), random.Random(718)
        for _ in range(512):
            while True:
                expected = {p: original_rng.choices(tuple(r.weights), weights=tuple(r.weights.values()))[0]
                            for p, r in joint.ranges.items()}
                cards = [c for h in expected.values() for c in h]
                if len(set(cards)) == len(cards):
                    break
            self.assertEqual(expected, joint.sample(cached_rng))

    def test_suit_aware_vectors_invariant_but_preserve_blockers(self):
        root, ranges = small_game(3)
        holding = tuple(sorted(root.actor.cards))
        first = suit_aware_features(root, ranges, holding)
        mapping = [2, 0, 3, 1]

        def card(c):
            return c // 4 * 4 + mapping[c % 4]

        other = root.clone()
        other.board = list(map(card, root.board))
        for p in other.players.values():
            p.cards = tuple(map(card, p.cards))
        changed = JointRanges(
            {
                p: HandRange({tuple(map(card, h)): w for h, w in r.weights.items()})
                for p, r in ranges.ranges.items()
            },
            tuple(other.board),
        )
        second = suit_aware_features(other, changed, tuple(sorted(map(card, holding))))
        self.assertEqual(first, second)
        self.assertEqual(first, suit_aware_features(root, ranges, holding[::-1]))
        self.assertGreater(len(first), 1326 * 3)

    def test_adaptive_estimates_use_fresh_worlds_and_every_action(self):
        root, ranges = small_game(2, street="TURN")
        budget = ComputeBudget(samples=32, max_depth=1, seed=19)
        private = ranges.conditioned(root.current_player, root.actor.cards)
        result = search(
            root,
            private,
            root.current_player,
            dict.fromkeys(root.players, UniformLegalPolicy()),
            budget,
            adaptive=True,
        )
        self.assertEqual(result.work.samples, 32)
        self.assertTrue(any("Pilot excluded" in warning for warning in result.warnings))
        self.assertTrue(all(se is not None for se in result.standard_errors.values()))
        repeat = search(
            root,
            private,
            root.current_player,
            dict.fromkeys(root.players, UniformLegalPolicy()),
            budget,
            adaptive=True,
        )
        self.assertEqual(result.action_ev, repeat.action_ev)

    def test_protected_replay_exact_total_and_resume(self):
        opening = HandState.start({0: 3.0, 1: 3.0}, 0, random.Random(2))
        river, _ = small_game(2)

        def sample(index):
            obs = observe(opening if index % 10 == 0 else river)
            target = tuple(float(m) * (index % 7) for m in obs.legal_mask)
            return TrainingSample(1, obs.hero, obs, target, 1.0, "advantage", 0)

        memory = ProtectedReplay(5, 15, kind="advantage", seed=7)
        for i in range(100):
            memory.add(sample(i))
        self.assertEqual(memory.diagnostics()["opening"]["retained"], 5)
        self.assertAlmostEqual(sum(s.weight for s in memory.samples), 100.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.pt"
            memory.save(path)
            resumed = ProtectedReplay.load(path)
            for i in range(100, 130):
                memory.add(sample(i))
                resumed.add(sample(i))
            self.assertEqual(memory.samples, resumed.samples)
        coverage = Coverage()
        for s in memory.samples:
            coverage.record(s.state, s.kind, iteration=s.iteration, weight=s.weight)
        merged = Coverage()
        merged.merge(coverage.as_dict())
        merged.merge(coverage.as_dict())
        self.assertAlmostEqual(
            merged.as_dict()["effective_sample_size"]["advantage"],
            2 * coverage.as_dict()["effective_sample_size"]["advantage"],
        )

    def test_fitting_invalidates_previous_quality_gate(self):
        from test_hybrid_learning import behavior_labels

        learner = Learner(behavior_labels(), "behavior")
        learner.validate(100.0, 100.0)
        self.assertTrue(learner.accepted)
        learner.fit_to(1)
        self.assertFalse(learner.accepted)
        self.assertEqual(learner.validation, {})
