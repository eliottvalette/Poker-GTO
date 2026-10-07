"""Importance weights retain their objective even above tensor/sum ranges."""
from dataclasses import replace
import math
import unittest

import torch

from actions import ACTION_IDS
from cfr_solver import regret_matching
from infoset import observe
from ml.memory import TrainingSample
from ml.model import AdvantageNetwork, AveragePolicyNetwork
from ml.train import evaluate_loss, fit, loss_for, weight_quality, _loss_with_weights
from test_solver import river


class TrainingWeightTests(unittest.TestCase):
    def test_weighted_sampling_has_the_same_expected_objective(self):
        model = AveragePolicyNetwork().eval()
        samples = self.samples()
        actual = loss_for(model, samples)
        expected = .25 * _loss_with_weights(model, [samples[0]], [1.0]) + .75 * _loss_with_weights(model, [samples[1]], [1.0])
        self.assertAlmostEqual(float(actual.detach()), float(expected.detach()), places=6)

    def test_weight_concentration_is_reported_without_clipping(self):
        samples = self.samples()
        result = weight_quality(samples)
        self.assertAlmostEqual(result['weight_effective_samples'], 1.6)
        self.assertAlmostEqual(result['largest_weight_share'], .75)
        self.assertAlmostEqual(result['top10_weight_share'], 1.)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def samples(self):
        obs = observe(river())
        first = next(i for i, legal in enumerate(obs.legal_mask) if legal)
        deterministic = tuple(float(i == first) for i in range(len(ACTION_IDS)))
        uniform = regret_matching([0.0] * len(ACTION_IDS), obs.legal_mask)
        return [TrainingSample(1, obs.hero, obs, deterministic, 1.0, "strategy", 0),
                TrainingSample(1, obs.hero, obs, uniform, 3.0, "strategy", 0)]

    def test_finite_weights_above_float32_range_preserve_loss(self):
        model = AveragePolicyNetwork().eval()
        samples = self.samples()
        huge = [replace(sample, weight=sample.weight * 1e200) for sample in samples]
        self.assertAlmostEqual(float(loss_for(model, samples).detach()), float(loss_for(model, huge).detach()), places=6)
        self.assertAlmostEqual(evaluate_loss(model, samples, 1), evaluate_loss(model, huge, 1), places=6)
        loss_for(model, huge).backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_sum_overflow_is_avoided_without_changing_weighted_objective(self):
        model = AveragePolicyNetwork().eval()
        samples = self.samples()
        huge = [replace(samples[0], weight=9e307), replace(samples[1], weight=9e307)]
        equal = [replace(sample, weight=1.0) for sample in samples]
        self.assertTrue(all(math.isfinite(sample.weight) for sample in huge))
        self.assertAlmostEqual(float(loss_for(model, equal).detach()), float(loss_for(model, huge).detach()), places=6)
        self.assertAlmostEqual(evaluate_loss(model, equal, 1), evaluate_loss(model, huge, 1), places=6)

    def test_tiny_training_with_large_importance_weights_has_finite_metrics(self):
        samples = [replace(sample, weight=sample.weight * 1e200) for sample in self.samples()] * 2
        metrics = fit(AveragePolicyNetwork(), samples, epochs=1, batch_size=1, seed=17)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_positive_weight_float32_underflow_is_explicit(self):
        samples = self.samples()
        samples[0] = replace(samples[0], weight=1e-100)
        with self.assertRaisesRegex(ValueError, "underflowed in float32"):
            loss_for(AveragePolicyNetwork(), samples)

    def test_float64_positive_ratio_underflow_is_explicit(self):
        samples = self.samples()
        samples[0] = replace(samples[0], weight=1e-300)
        samples[1] = replace(samples[1], weight=1e300)
        with self.assertRaisesRegex(ValueError, "underflowed in float64"):
            loss_for(AveragePolicyNetwork(), samples)

    def test_unrepresentable_float32_regret_target_is_explicit(self):
        sample = self.samples()[0]
        target = tuple(1e100 if legal else 0 for legal in sample.state.legal_mask)
        sample = replace(sample, kind="advantage", target=target)
        with self.assertRaisesRegex(ValueError, "target cannot be represented as finite float32"):
            loss_for(AdvantageNetwork(), [sample])

    def test_nonzero_regret_float32_underflow_is_explicit(self):
        sample = self.samples()[0]
        target = tuple(1e-100 if legal else 0 for legal in sample.state.legal_mask)
        sample = replace(sample, kind="advantage", target=target)
        with self.assertRaisesRegex(ValueError, "target underflowed in float32"):
            loss_for(AdvantageNetwork(), [sample])


if __name__ == "__main__":
    unittest.main()
