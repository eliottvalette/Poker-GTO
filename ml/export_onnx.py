"""Export a validated average policy for singleton, full-history browser inference.

The training encoder packs padded batches. A singleton has no padding, so using
the same GRU weights directly preserves its full sequence and exports cleanly.
No training, GPU inference, or legacy checkpoint conversion occurs here.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from actions import ACTION_IDS
from infoset import EVENTS, HISTORY_WIDTH, NUMERIC_NAMES, POSITIONS, STATE_VERSION, Observation
from ml.deep_cfr import NeuralAveragePolicy
from ml.model import MODEL_ARCHITECTURE, AveragePolicyNetwork, encode_batch

INPUT_NAMES = ("cards", "street", "position", "numeric", "history", "mask")
EXPORT_VERSION = 1


class SingleObservationAveragePolicy(nn.Module):
    """The exact average-policy inference graph, with batch size fixed to one."""

    def __init__(self, model: AveragePolicyNetwork) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.head = model.head

    def forward(self, cards: torch.Tensor, street: torch.Tensor, position: torch.Tensor,
                numeric: torch.Tensor, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder.history(history)
        features = torch.cat((self.encoder.cards(cards).flatten(1), self.encoder.street(street),
                              self.encoder.position(position), self.encoder.numeric(numeric), hidden[-1]), dim=1)
        return self.head(features).masked_fill(~mask, -torch.inf).softmax(dim=1)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_average_policy(checkpoint: str | Path, model_path: str | Path, manifest_path: str | Path,
                          example_observation: Observation) -> dict[str, object]:
    """Validate and export to explicit destinations; return the browser manifest.

    Requires onnx and onnxruntime. Both CPU inference checks finish before files
    are published. Each file is replaced atomically; the SHA256 binding rejects
    mismatched pairs if publication is interrupted between the replacements.
    """
    import onnx
    import onnxruntime as ort

    model_path, manifest_path = Path(model_path), Path(manifest_path)
    if model_path.resolve() == manifest_path.resolve():
        raise ValueError(f"Model and manifest must have distinct paths: {model_path}")
    if model_path.suffix != ".onnx" or manifest_path.suffix != ".json":
        raise ValueError(f"Expected .onnx model and .json manifest paths: {model_path}, {manifest_path}")
    policy = NeuralAveragePolicy(checkpoint)
    policy.query(example_observation)
    graph = SingleObservationAveragePolicy(policy.model).eval()
    batch = encode_batch([example_observation])
    payload = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(graph, tuple(batch[name] for name in INPUT_NAMES), payload,
                          input_names=list(INPUT_NAMES), output_names=["probabilities"],
                          dynamic_axes={"history": {1: "history_length"}}, opset_version=17,
                          dynamo=False, external_data=False)
    model_bytes = payload.getvalue()
    onnx.checker.check_model(onnx.load_model_from_string(model_bytes))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
    errors = []
    # A second complete length verifies that export did not freeze the example's
    # time dimension. The extra token is a structural inference test only.
    longer = replace(example_observation, history=(*example_observation.history, example_observation.history[-1]))
    for obs in (example_observation, longer):
        inputs = encode_batch([obs])
        expected = np.asarray(policy.query(obs), dtype=np.float32)
        actual = runtime.run(["probabilities"], {name: inputs[name].numpy() for name in INPUT_NAMES})[0][0]
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"ONNX average-policy parity failed: expected={expected.tolist()}, actual={actual.tolist()}")
        errors.append(float(np.abs(actual - expected).max()))
    manifest = {"version": EXPORT_VERSION, "state_version": STATE_VERSION, "architecture": MODEL_ARCHITECTURE,
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(), "objective": policy.objective,
                "iteration": policy.iteration, "supported_player_counts": list(policy.supported_player_counts),
                "actions": list(ACTION_IDS), "numeric_names": list(NUMERIC_NAMES), "positions": list(POSITIONS),
                "events": list(EVENTS), "normalization_bb": 25.0, "card_encoding": "rank_index_times_4_plus_suit_index",
                "card_slots": 7, "unknown_card": 52, "card_vocabulary": 53, "history_width": HISTORY_WIDTH,
                "full_history": True, "batch_size": 1, "inputs": list(INPUT_NAMES), "output": "probabilities",
                "validation_max_absolute_error": max(errors)}
    _atomic_write(model_path, model_bytes)
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return manifest
