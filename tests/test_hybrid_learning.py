"""The bounded learning lifecycle includes deterministic optimizer resume."""
import random
from pathlib import Path
import tempfile
import unittest
import torch

from hybrid.learning import Learner, LearnedBehavior, Supervision, observation_features
from hybrid.policy_source import category_policy, distribution
from infoset import observe
from poker_game_expresso import HandState


def behavior_labels(count=2):
    teacher = category_policy({"FOLD": .1, "CALL": 3, "CHECK": 3, "RAISE": .2, "ALL_IN": .1}, "synthetic-passive-v1")
    result = []
    for seed in range(16):
        state = HandState.start({p: 3.0 for p in range(count)}, 0, random.Random(seed))
        obs = observe(state)
        result.append(Supervision(observation_features(obs), distribution(teacher, state), obs.legal_mask,
                                   teacher.version, count, {"policy_queries": 1}, 0., str(seed)))
    return result


class LearningTests(unittest.TestCase):
    def test_save_resume_export_and_policy_load(self):
        records = behavior_labels()
        learner = Learner(records, "behavior", seed=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behavior.pt"
            learner.fit_to(3)
            learner.save(path)
            resumed = Learner.load(path)
            learner.fit_to(8)
            resumed.fit_to(8)
            for a, b in zip(learner.model.parameters(), resumed.model.parameters()):
                self.assertTrue(torch.equal(a, b))
            payload = resumed.export(Path(directory) / "behavior.json")
            self.assertEqual(payload["epochs"], 8)
            state = HandState.start({0: 3, 1: 3}, 0, random.Random(31))
            policy = LearnedBehavior(resumed, experimental=True)
            self.assertAlmostEqual(sum(policy.probabilities(observe(state))), 1)
            with self.assertRaisesRegex(ValueError, "player-count"):
                policy.probabilities(observe(HandState.start({0: 3, 1: 3, 2: 3}, 0, random.Random(31))))
            resumed.validate(baseline_loss=0, maximum_loss=0)
            with self.assertRaises(ValueError):
                LearnedBehavior(resumed)
