"""Optional street specialists learn from computed games, never steer the teacher.

This is a separate, explicitly context-balanced distillation objective. It does
not change Deep CFR's importance-weighted objective or any published schema.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import io
from pathlib import Path
import random

import torch
from torch import nn

from actions import ACTION_IDS
from hybrid.knowledge import supervision
from ml.health import numerical_health
from ml.model import AdvantageNetwork, STREETS, StreetNetworks
from features import FEATURE_SCHEMA_VERSION
from ml.train import evaluate_loss, loss_for
from training.checkpoint import atomic_bytes
from training.metrics import decision_context


class ResidualAdvantage(AdvantageNetwork):
    """Fresh experimental head with a normalized direct path; never load old A."""

    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, len(ACTION_IDS)))
        self.direct = nn.Linear(128, len(ACTION_IDS))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.normalization(self.encoder(batch))
        values = self.head(features) + self.direct(features)
        if not torch.isfinite(values).all():
            raise ValueError("Residual advantage produced nonfinite values")
        return values


def conditional_weights(samples: list) -> list:
    """Equal context mass, preserving relative target weights inside each context."""
    totals = defaultdict(float)
    for sample in samples:
        totals[decision_context(sample.state)] += sample.weight * sample.iteration
    return [replace(s, weight=s.weight / totals[decision_context(s.state)]) for s in samples]


class SolverStudents:
    """Eight independent optional approximators per track, with exact resume."""

    def __init__(self, player_count: int, *, residual: bool = True, seed: int = 0) -> None:
        if player_count not in (2, 3):
            raise ValueError("Solver students require a HU or 3-max track")
        self.player_count, self.residual, self.seed = player_count, residual, seed
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.models = {kind: StreetNetworks(kind) for kind in ("advantage", "strategy")}
            if residual:
                for street in STREETS:
                    candidate = ResidualAdvantage()
                    candidate.encoder.load_state_dict(self.models["advantage"].specialists[street].encoder.state_dict())
                    self.models["advantage"].specialists[street] = candidate
        self.optimizers = {(kind, street): torch.optim.Adam(model.parameters(), lr=3e-4)
                           for kind, bank in self.models.items() for street, model in bank.specialists.items()}
        self.updates = {key: 0 for key in self.optimizers}
        self.dataset_contract: list | None = None

    def fit(self, paths: list[Path], *, updates: int = 64, batch_size: int = 64) -> dict:
        if type(updates) is not int or updates < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer update and minibatch budgets required")
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("Duplicate solver supervision artifacts")
        populations = defaultdict(list)
        contracts = []
        groups = {}
        for path in sorted(paths):
            for kind in self.models:
                contract, samples = supervision(path, kind)
                if len(contract["root"]["players"]) != self.player_count:
                    raise ValueError("Solver supervision has the wrong player count")
                if kind == "advantage":
                    import hashlib
                    contracts.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
                for sample in samples:
                    split = contract["split"]
                    if sample.root_group in groups and groups[sample.root_group] != split:
                        raise ValueError("Public root leakage across solver supervision splits")
                    groups[sample.root_group] = split
                    populations[kind, STREETS[sample.state.street], split].append(sample)
        if self.dataset_contract is not None and self.dataset_contract != contracts:
            raise ValueError("Frozen distillation dataset changed; create a separate study")
        # Validate every population before mutating any specialist.
        for kind, street in self.optimizers:
            for split in ("train", "validation", "test"):
                if not populations[kind, street, split]:
                    raise ValueError(f"Missing solver supervision: {kind}/{street}/{split}")
        self.dataset_contract = contracts
        reports = {}
        for (kind, street), optimizer in self.optimizers.items():
            model = self.models[kind].specialists[street]
            training = conditional_weights(populations[kind, street, "train"])
            before = numerical_health(model, training[:min(len(training), 128)])
            for _ in range(updates):
                rng = random.Random(f"{self.seed}/{kind}/{street}/{self.updates[kind, street]}")
                batch = rng.choices(training, k=batch_size)
                optimizer.zero_grad(set_to_none=True)
                # Fixed population normalization keeps minibatch context mass exact
                # in expectation; never renormalize by a random minibatch's mass.
                mean_weight = sum(s.weight * s.iteration for s in training) / len(training)
                loss_for(model, batch, mean_weight).backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("Nonfinite solver student gradients")
                optimizer.step()
                self.updates[kind, street] += 1
            self.models[kind].trained[STREETS.index(street)] = True
            splits = {}
            for split in ("train", "validation", "test"):
                rows = populations[kind, street, split]
                contexts = sorted({decision_context(s.state) for s in rows})
                splits[split] = {context: {"records": len(selected),
                    "loss": evaluate_loss(model, selected, 128),
                    "root_groups": len({s.root_group for s in selected})}
                    for context in contexts if (selected := [s for s in rows if decision_context(s.state) == context])}
            reports[f"{kind}/{street}"] = {"updates": self.updates[kind, street], "before": before,
                "after": numerical_health(model, training[:min(len(training), 128)]), "splits": splits,
                "accepted": False, "scope": "distillation only; range priors absent from legacy neural encoder; not accepted as a range-aware solver"}
        return reports

    def save(self, path: Path) -> None:
        raw = {"schema": "solver-students-v1", "player_count": self.player_count, "residual": self.residual,
               "feature_schema": FEATURE_SCHEMA_VERSION, "actions": ACTION_IDS,
               "seed": self.seed, "models": {k: m.state_dict() for k, m in self.models.items()},
               "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
               "updates": self.updates, "dataset_contract": self.dataset_contract, "accepted": False}
        stream = io.BytesIO()
        torch.save(raw, stream)
        atomic_bytes(path, stream.getvalue())

    @classmethod
    def load(cls, path: Path) -> SolverStudents:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if raw["schema"] != "solver-students-v1" or raw["feature_schema"] != FEATURE_SCHEMA_VERSION or tuple(raw["actions"]) != tuple(ACTION_IDS):
            raise ValueError("Incompatible solver student checkpoint")
        result = cls(raw["player_count"], residual=raw["residual"], seed=raw["seed"])
        for key, model in result.models.items():
            model.load_state_dict(raw["models"][key], strict=True)
        for key, optimizer in result.optimizers.items():
            optimizer.load_state_dict(raw["optimizers"][key])
        result.updates, result.dataset_contract = raw["updates"], raw["dataset_contract"]
        return result
