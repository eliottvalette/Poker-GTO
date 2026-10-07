"""Temporary random-weight fixtures validate inference transport, not policy quality."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
import torch

from actions import ACTION_IDS
from infoset import STATE_VERSION, observe
from features import FEATURE_SCHEMA_VERSION
from features.neural import NEURAL_NUMERIC_NAMES as NUMERIC_NAMES, numeric_names
from ml.deep_cfr import NeuralAveragePolicy
from ml.export_onnx import INPUT_NAMES, export_average_policy
from ml.model import MODEL_ARCHITECTURE, AveragePolicyNetwork, encode_batch, model_architecture
from poker_game_expresso import HandState


class ONNXExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dynamic_complete_history_matches_pytorch(self):
        # Explicitly untrained weights exist only in this temporary directory.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            weights = AveragePolicyNetwork().state_dict()
        observations = [observe(HandState.start(stacks, 0, random.Random(19)))
                        for stacks in ({0: 25.0, 1: 25.0}, {0: 25.0, 1: 25.0, 2: 25.0})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, model, manifest_path = root / "untrained_fixture.pt", root / "policy.onnx", root / "policy.json"
            torch.save({"version": 5, "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "training_metadata": {"traversal_mode": "external_sampling"}, "state_version": STATE_VERSION, "architecture": MODEL_ARCHITECTURE,
                        "actions": list(ACTION_IDS), "numeric_names": list(NUMERIC_NAMES),
                        "objective": "hand_chip_delta", "iteration": 1, "supported_player_counts": [2, 3],
                        "weights": weights, "metrics": [{"fixture": "untrained inference validation"}]}, checkpoint)
            manifest = export_average_policy(checkpoint, model, manifest_path, observations[0])
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
            self.assertEqual(manifest["model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest())
            self.assertTrue(manifest["full_history"])
            self.assertLess(manifest["validation_max_absolute_error"], 1e-6)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
            policy = NeuralAveragePolicy(checkpoint)
            for observation in observations:
                inputs = encode_batch([observation])
                output = session.run(["probabilities"], {name: inputs[name].numpy() for name in INPUT_NAMES})[0][0]
                np.testing.assert_allclose(output, policy.query(observation), rtol=1e-5, atol=1e-6)
                self.assertEqual(float(output[~np.asarray(observation.legal_mask)].sum()), 0)

    def test_invalid_checkpoint_does_not_publish_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, model, manifest = root / "legacy.pt", root / "policy.onnx", root / "policy.json"
            torch.save({"version": 1}, checkpoint)
            observation = observe(HandState.start({0: 25.0, 1: 25.0}, 0, random.Random(1)))
            with self.assertRaisesRegex(ValueError, "Incompatible average-policy"):
                export_average_policy(checkpoint, model, manifest, observation)
            self.assertFalse(model.exists())
            self.assertFalse(manifest.exists())

    def test_old_feature_inference_and_export_keep_the_declared_encoding(self):
        observation = observe(HandState.start({0: 25.0, 1: 25.0}, 0, random.Random(19)))
        for version in (2, 3):
            with self.subTest(feature_version=version), tempfile.TemporaryDirectory() as directory:
                model = AveragePolicyNetwork(feature_version=version).eval()
                root = Path(directory)
                checkpoint = root / f"feature_v{version}.pt"
                torch.save({"version": 5, "feature_schema_version": version, "state_version": STATE_VERSION,
                            "architecture": model_architecture(version), "actions": list(ACTION_IDS),
                            "numeric_names": list(numeric_names(version)), "iteration": 1, "objective": "hand_chip_delta",
                            "supported_player_counts": [2], "training_metadata": {"traversal_mode": "external_sampling"},
                            "weights": model.state_dict()}, checkpoint)
                policy = NeuralAveragePolicy(checkpoint)
                self.assertEqual(policy.query(observation), model.probabilities(observation))
                manifest = export_average_policy(checkpoint, root / "policy.onnx", root / "manifest.json", observation)
                self.assertEqual(manifest["feature_schema_version"], version)
                self.assertEqual(manifest["numeric_names"], list(numeric_names(version)))
                self.assertEqual(manifest["suit_normalization"], "first_observable_occurrence" if version == 2 else "private_order_minimum")
                self.assertEqual(manifest["architecture"], model_architecture(version))


if __name__ == "__main__":
    unittest.main()
