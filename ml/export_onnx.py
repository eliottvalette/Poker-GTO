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

from actions import ACTION_IDS, ACTION_SCHEMA_VERSION
from infoset import EVENTS, HISTORY_WIDTH, POSITIONS, STATE_VERSION, Observation
from features.neural import numeric_names
from ml.deep_cfr import NeuralAveragePolicy
from ml.model import AveragePolicyNetwork, StreetNetworks, STREETS, encode_batch, model_architecture

INPUT_NAMES = ("cards", "street", "position", "numeric", "history", "mask")
EXPORT_VERSION = 4


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
    if isinstance(policy.model, StreetNetworks):
        return export_street_policy(policy, checkpoint, model_path, manifest_path)
    policy.query(example_observation)
    graph = SingleObservationAveragePolicy(policy.model).eval()
    batch = encode_batch([example_observation], policy.model.feature_version)
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
        inputs = encode_batch([obs], policy.model.feature_version)
        expected = np.asarray(policy.query(obs), dtype=np.float32)
        actual = runtime.run(["probabilities"], {name: inputs[name].numpy() for name in INPUT_NAMES})[0][0]
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"ONNX average-policy parity failed: expected={expected.tolist()}, actual={actual.tolist()}")
        errors.append(float(np.abs(actual - expected).max()))
    manifest = {"version": EXPORT_VERSION, "action_schema_version": ACTION_SCHEMA_VERSION, "feature_schema_version": policy.model.feature_version,
                "suit_normalization": "private_order_minimum" if policy.model.feature_version >= 3 else "first_observable_occurrence", "seat_normalization": "hero_then_clockwise_positions", "traversal_mode": policy.traversal_mode, "state_version": STATE_VERSION, "architecture": model_architecture(policy.model.feature_version),
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(), "objective": policy.objective,
                "iteration": policy.iteration, "supported_player_counts": list(policy.supported_player_counts),
                "actions": list(ACTION_IDS), "numeric_names": list(numeric_names(policy.model.feature_version)), "positions": list(POSITIONS),
                "events": list(EVENTS), "normalization_bb": 25.0, "card_encoding": "rank_index_times_4_plus_suit_index",
                "card_slots": 7, "unknown_card": 52, "card_vocabulary": 53, "history_width": HISTORY_WIDTH,
                "full_history": True, "batch_size": 1, "inputs": list(INPUT_NAMES), "output": "probabilities",
                "validation_max_absolute_error": max(errors), "amount_units": "current_big_blinds",
                "utility_units": "initial_big_blind_chips", "history_scope": "current_hand",
                "payout_scope": "winner_take_all"}
    _atomic_write(model_path, model_bytes)
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return manifest


def export_street_policy(policy, checkpoint, model_path: Path, manifest_path: Path) -> dict:
    """Export all four validated routes; the manifest is the bundle commit point."""
    from training.evaluation import fixed_roots
    from infoset import observe
    from training.checkpoint import atomic_bytes
    count = policy.supported_player_counts[0]
    roots = {hand.street: observe(hand) for _,hand in fixed_roots(count)}
    if set(roots) != set(STREETS):
        raise ValueError("Export needs legal example observations on every street")
    source_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    routes = {}
    with tempfile.TemporaryDirectory(dir=manifest_path.parent, prefix=".street-export-") as staging:
        staging = Path(staging)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for index, street in enumerate(STREETS):
            name = f"{model_path.stem}_{street.lower()}"
            single = {k:v for k,v in raw.items() if k not in ("layout", "specialists")}
            single.update(version=5, weights=policy.model.specialists[street].state_dict())
            small = staging / f"{name}.pt"
            torch.save(single, small)
            manifest = export_average_policy(small, staging/f"{name}.onnx", staging/f"{name}.json", roots[street])
            routes[street] = {"player_count": count, "street": street, "model_kind": "AVERAGE",
                             "model_version": policy.specialist_metadata[street]["model_version"],
                             "checkpoint_source": source_hash, "feature_schema": manifest["feature_schema_version"],
                             "action_schema": manifest["action_schema_version"], "model_hash": manifest["model_sha256"],
                             "model_file": f"{name}.onnx", "manifest": manifest,
                             "coverage": policy.specialist_metadata[street]}
        catalog = {"version": 5, "layout": "independent_streets_v1", "iteration": policy.iteration,
                   "supported_player_counts": [count], "routes": routes}
        for route in routes.values():
            atomic_bytes(manifest_path.parent/route["model_file"], (staging/route["model_file"]).read_bytes())
        atomic_bytes(manifest_path, (json.dumps(catalog, indent=2, allow_nan=False)+"\n").encode())
    return catalog
