"""Bounded CPU learning with provenance, held-out gates and resumable optimizer state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from actions import ACTION_IDS
from features.neural import neural_observation
from infoset import Observation
from training.checkpoint import atomic_bytes

MODEL_SCHEMA = 1


def observation_features(obs: Observation) -> tuple[float, ...]:
    """Visible cards/state and complete history with an explicit 128-event limit."""
    encoded = neural_observation(obs)
    cards = tuple(float(c == index) for c in encoded.cards for index in range(53))
    history = encoded.history
    if len(history) > 128:
        raise ValueError("Experimental hybrid encoder supports at most 128 complete history events")
    padded = tuple(v for row in history for v in row) + (0.0,) * (12 * (128 - len(history)))
    return (*cards, *encoded.numeric, len(history) / 128, *padded)


@dataclass(frozen=True)
class Supervision:
    features: tuple[float, ...]
    target: tuple[float, ...]
    mask: tuple[bool, ...]
    provenance: str
    player_count: int
    compute_budget: dict
    standard_error: float | None
    split_group: str


class Approximation(nn.Module):
    def __init__(self, width: int, kind: Literal["behavior", "continuation"], hidden_width: int = 32) -> None:
        super().__init__()
        if kind not in ("behavior", "continuation"):
            raise ValueError(f"Unknown approximation role: {kind}")
        if type(hidden_width) is not int or hidden_width < 1:
            raise ValueError("Positive hidden width required")
        self.width, self.kind, self.hidden_width = width, kind, hidden_width
        self.layers = nn.Sequential(nn.Linear(width, hidden_width), nn.Tanh(),
                                    nn.Linear(hidden_width, len(ACTION_IDS) if kind == "behavior" else 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class Learner:
    """Full-batch deterministic diagnostics; checkpoint contains the entire label set."""

    def __init__(self, records: list[Supervision], kind: Literal["behavior", "continuation"],
                 seed: int = 0, learning_rate: float = .003, hidden_width: int = 32) -> None:
        if len(records) < 4 or len({r.split_group for r in records}) < 2:
            raise ValueError("Learning needs at least four labels and two independent split groups")
        if len({r.player_count for r in records}) != 1:
            raise ValueError("HU and three-player models require separate learning tracks")
        if any(not r.provenance or len(r.features) != len(records[0].features) for r in records):
            raise ValueError("Consistent feature width and label provenance are mandatory")
        for record in records:
            if record.player_count not in (2, 3) or any(not math.isfinite(v) for v in (*record.features, *record.target)):
                raise ValueError("Learning labels require finite features/targets and supported player counts")
            if kind == "behavior" and (len(record.target) != len(ACTION_IDS) or len(record.mask) != len(ACTION_IDS)
                                       or any(p < 0 or (p > 0 and not m) for p, m in zip(record.target, record.mask))
                                       or not math.isclose(sum(record.target), 1., abs_tol=1e-7)):
                raise ValueError("Behavior labels must be normalized legal action probabilities")
            if kind == "continuation" and len(record.target) != 1:
                raise ValueError("Continuation labels contain one private-indexed physical-chip value")
        self.records, self.kind, self.seed = records, kind, seed
        self.learning_rate = learning_rate
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = Approximation(len(records[0].features), kind, hidden_width)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = 0
        self.accepted = False
        self.validation: dict = {}
        groups = sorted({r.split_group for r in records}, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
        held = set(groups[:max(1, len(groups) // 5)])
        self.train_indices = [i for i, r in enumerate(records) if r.split_group not in held]
        self.held_indices = [i for i, r in enumerate(records) if r.split_group in held]

    def _loss(self, indices: list[int]) -> torch.Tensor:
        rows = [self.records[i] for i in indices]
        outputs = self.model(torch.tensor([r.features for r in rows], dtype=torch.float32))
        targets = torch.tensor([r.target for r in rows], dtype=torch.float32)
        if self.kind == "continuation":
            return (outputs - targets).square().mean()
        masks = torch.tensor([r.mask for r in rows], dtype=torch.bool)
        logp = outputs.masked_fill(~masks, -1e9).log_softmax(1)
        return -(targets * logp).sum(1).mean()

    def fit_to(self, epochs: int) -> dict:
        if type(epochs) is not int or epochs < self.epochs:
            raise ValueError("Requested fit budget must be an integer at least the completed epoch count")
        if epochs > self.epochs:
            self.accepted = False
            self.validation = {}
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for _ in range(self.epochs, epochs):
                self.optimizer.zero_grad()
                loss = self._loss(self.train_indices)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite learning loss")
                loss.backward()
                self.optimizer.step()
                self.epochs += 1
            with torch.no_grad():
                return {"epochs": self.epochs, "train_loss": float(self._loss(self.train_indices)),
                        "held_out_loss": float(self._loss(self.held_indices)),
                        "training_labels": len(self.train_indices), "held_out_labels": len(self.held_indices)}
        finally:
            torch.set_num_threads(threads)

    def validate(self, baseline_loss: float, maximum_loss: float) -> dict:
        with torch.no_grad():
            loss = float(self._loss(self.held_indices))
        self.accepted = loss < baseline_loss and loss <= maximum_loss
        self.validation = {"held_out_loss": loss, "baseline_loss": baseline_loss,
                           "maximum_loss": maximum_loss, "accepted": self.accepted,
                           "scope": "held-out split groups of this label distribution only"}
        return self.validation

    def save(self, path: Path) -> None:
        payload = {"schema": MODEL_SCHEMA, "records": [asdict(r) for r in self.records], "kind": self.kind,
                   "seed": self.seed, "learning_rate": self.learning_rate, "epochs": self.epochs,
                   "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "accepted": self.accepted, "validation": self.validation, "hidden_width": self.model.hidden_width}
        stream = io.BytesIO()
        torch.save(payload, stream)
        atomic_bytes(path, stream.getvalue())

    @classmethod
    def load(cls, path: Path) -> Learner:
        raw = torch.load(path, weights_only=True, map_location="cpu")
        if raw["schema"] != MODEL_SCHEMA:
            raise ValueError(f"Incompatible hybrid model schema: {raw['schema']}")
        result = cls([Supervision(**r) for r in raw["records"]], raw["kind"], raw["seed"], raw["learning_rate"], raw.get("hidden_width", 32))
        result.model.load_state_dict(raw["model"], strict=True)
        result.optimizer.load_state_dict(raw["optimizer"])
        result.epochs, result.accepted, result.validation = raw["epochs"], raw["accepted"], raw["validation"]
        return result

    def export(self, path: Path) -> dict:
        """Portable exact MLP weights; no browser runtime or acceptance is implied."""
        payload = {"schema": MODEL_SCHEMA, "kind": self.kind, "width": self.model.width,
                   "player_count": self.records[0].player_count, "epochs": self.epochs, "hidden_width": self.model.hidden_width,
                   "accepted": self.accepted, "validation": self.validation,
                   "layers": {k: v.tolist() for k, v in self.model.state_dict().items()}}
        atomic_bytes(path, json.dumps(payload, allow_nan=False).encode())
        return payload

    def export_onnx(self, path: Path) -> dict:
        """Validate CPU parity before publishing an optional approximation graph."""
        import numpy as np
        import onnx
        import onnxruntime as ort
        stream = io.BytesIO()
        example = torch.tensor([self.records[i].features for i in self.held_indices], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            torch.onnx.export(self.model, example[:1], stream, input_names=["features"], output_names=["outputs"],
                              dynamic_axes={"features": {0: "batch"}, "outputs": {0: "batch"}},
                              opset_version=17, dynamo=False, external_data=False)
            expected = self.model(example).numpy()
        payload = stream.getvalue()
        onnx.checker.check_model(onnx.load_model_from_string(payload))
        session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
        actual = session.run(["outputs"], {"features": example.numpy()})[0]
        gap = float(np.max(np.abs(actual - expected)))
        if gap > 1e-5:
            raise ValueError(f"Hybrid ONNX numerical parity failed: maximum error={gap}")
        manifest = {"schema": MODEL_SCHEMA, "kind": self.kind, "input_width": self.model.width,
                    "output_semantics": "legal-mask-before-softmax logits" if self.kind == "behavior" else "physical_chips/hand_delta",
                    "player_count": self.records[0].player_count, "accepted": self.accepted,
                    "validation": self.validation, "maximum_absolute_error": gap,
                    "sha256": hashlib.sha256(payload).hexdigest()}
        atomic_bytes(path, payload)
        atomic_bytes(path.with_suffix(".manifest.json"), json.dumps(manifest, indent=2).encode())
        return manifest


class LearnedBehavior:
    def __init__(self, learner: Learner, *, experimental: bool = False) -> None:
        if learner.kind != "behavior" or (not learner.accepted and not experimental):
            raise ValueError("Behavior model requires acceptance or explicit experimental=True")
        self.learner = learner
        digest = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in learner.model.parameters())).hexdigest()[:16]
        self.version = f"behavior-v{MODEL_SCHEMA}/players={learner.records[0].player_count}/width={learner.model.hidden_width}/epoch={learner.epochs}/{digest}"

    def probabilities(self, observation: Observation) -> tuple[float, ...]:
        from infoset import NUMERIC_NAMES
        from cfr_solver import validate_strategy
        if round(observation.numeric[NUMERIC_NAMES.index("player_count")] * 3) != self.learner.records[0].player_count:
            raise ValueError("Behavior model player-count mismatch")
        with torch.no_grad():
            logits = self.learner.model(torch.tensor([observation_features(observation)], dtype=torch.float32))[0]
            probabilities = logits.masked_fill(~torch.tensor(observation.legal_mask), -torch.inf).softmax(0).tolist()
        total = sum(probabilities)
        result = tuple(p / total for p in probabilities)
        validate_strategy(result, observation.legal_mask)
        return result
