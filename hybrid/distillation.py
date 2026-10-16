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
from ml.model import AdvantageNetwork, STREETS
from features import FEATURE_SCHEMA_VERSION
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
    """Eight root specialists per track; ranges are mandatory, even in ablations."""

    def __init__(self, player_count: int, *, residual: bool = True, seed: int = 0,
                 continuations: tuple[str, ...] = ("terminal-river-v1", "uniform-legal-v1"),
                 conditioned: bool = True) -> None:
        from hybrid.solver_inputs import RootStudentBank
        if player_count not in (2, 3) or not continuations or len(set(continuations)) != len(continuations):
            raise ValueError("Explicit HU/3-max track and unique continuation vocabulary required")
        self.player_count, self.residual, self.seed = player_count, residual, seed
        self.continuations, self.conditioned = tuple(continuations), conditioned
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.models = {kind: RootStudentBank(kind, self.continuations, player_count=player_count, conditioned=conditioned, residual=residual)
                           for kind in ("advantage", "strategy")}
        self.optimizers = {(kind, street): torch.optim.Adam(model.parameters(), lr=3e-4)
                           for kind, bank in self.models.items() for street, model in bank.specialists.items()}
        self.updates = {key: 0 for key in self.optimizers}
        self.dataset_contract: list | None = None

    def fit(self, paths: list[Path], *, updates: int = 64, batch_size: int = 64,
            max_encoding_bytes: int = 256 * 1024 * 1024, allow_dataset_update: bool = False) -> dict:
        from hybrid.knowledge import root_supervision
        from hybrid.solver_inputs import encode_roots
        import hashlib
        if type(updates) is not int or updates < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer update and minibatch budgets required")
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("Duplicate solver supervision artifacts")
        populations, contexts = defaultdict(list), {}
        contracts, groups = [], {}
        for path in sorted(paths):
            contracts.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
            for kind in self.models:
                contract, samples = root_supervision(path, kind)
                if len(contract["root"]["players"]) != self.player_count:
                    raise ValueError("Solver supervision has the wrong player count")
                for sample in samples:
                    split = contract["split"]
                    if sample.root_group in groups and groups[sample.root_group] != split:
                        raise ValueError("Public root leakage across solver supervision splits")
                    groups[sample.root_group] = split
                    populations[kind, STREETS[sample.state.street], split].append(sample)
                    contexts[id(sample)] = contract
        if type(allow_dataset_update) is not bool:
            raise ValueError("Dataset update permission must be explicit boolean")
        if self.dataset_contract is not None and self.dataset_contract != contracts and not allow_dataset_update:
            raise ValueError("Frozen distillation dataset changed; create a separate study")
        from features.neural import numeric_names
        from infoset import HISTORY_WIDTH
        if type(max_encoding_bytes) is not int or max_encoding_bytes < 1:
            raise ValueError("Positive encoded-dataset memory budget required")
        estimated = sum(len(rows) * (4 * (3978 + len(self.continuations) + 1 + len(numeric_names()))
                        + 7 * 8 + 3 * 8 + len(ACTION_IDS)
                        + max(len(s.state.history) for s in rows) * HISTORY_WIDTH * 4)
                        for rows in populations.values() if rows)
        if estimated > max_encoding_bytes:
            raise MemoryError(f"Root dataset encoding requires {estimated} bytes; budget={max_encoding_bytes}")
        encoded = {}
        for kind, street in self.optimizers:
            for split in ("train", "validation", "test"):
                rows = populations[kind, street, split]
                if not rows:
                    raise ValueError(f"Missing solver supervision: {kind}/{street}/{split}")
                encoded[kind, street, split] = encode_roots([s.state for s in rows],
                    [contexts[id(s)] for s in rows], self.continuations)
        self.dataset_contract = contracts
        reports = {}
        for (kind, street), optimizer in self.optimizers.items():
            model = self.models[kind].specialists[street]
            rows = populations[kind, street, "train"]
            weights = torch.tensor([s.weight for s in conditional_weights(rows)])
            weights /= weights.mean()
            targets = torch.tensor([s.target for s in rows])
            inputs = encoded[kind, street, "train"]

            def losses(output, target, mask):
                if kind == "advantage":
                    return ((output - target).square() * mask).sum(1) / mask.sum(1)
                return (target * (target.clamp_min(1e-12).log() - output.clamp_min(1e-12).log())).sum(1)

            for _ in range(updates):
                rng = random.Random(f"{self.seed}/{kind}/{street}/{self.updates[kind, street]}")
                indices = torch.tensor(rng.choices(range(len(rows)), k=batch_size))
                batch = {k: v.index_select(0, indices) for k, v in inputs.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = (losses(model(batch), targets[indices], batch["mask"]) * weights[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite solver student loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("Nonfinite solver student gradients")
                optimizer.step()
                self.updates[kind, street] += 1
            self.models[kind].trained[STREETS.index(street)] = True
            splits = {}
            for split in ("train", "validation", "test"):
                rows = populations[kind, street, split]
                inputs = encoded[kind, street, split]
                with torch.no_grad():
                    output = model(inputs)
                    errors = losses(output, torch.tensor([s.target for s in rows]), inputs["mask"])
                splits[split] = {"records": len(rows), "loss": float(errors.mean()),
                    "root_groups": len({s.root_group for s in rows}),
                    "all_nonpositive_fraction": float((output.masked_fill(~inputs["mask"], -torch.inf).max(1).values <= 0).float().mean()) if kind == "advantage" else None,
                    "contexts": {c: {"records": len(indices), "loss": float(errors[indices].mean())}
                        for c in sorted({decision_context(s.state) for s in rows})
                        if (indices := [i for i, s in enumerate(rows) if decision_context(s.state) == c])}}
            reports[f"{kind}/{street}"] = {"updates": self.updates[kind, street], "splits": splits,
                "accepted": False, "scope": "root-only exact public factors and declared continuation; finite-game labels"}
        return reports

    def save(self, path: Path) -> None:
        from hybrid.solver_inputs import INPUT_SCHEMA
        raw = {"schema": "solver-root-students-v2", "input_schema": INPUT_SCHEMA,
               "target_schema": "average-profile-root-advantage-bb-v1",
               "player_count": self.player_count, "residual": self.residual,
               "continuations": self.continuations, "conditioned": self.conditioned,
               "feature_schema": FEATURE_SCHEMA_VERSION, "actions": ACTION_IDS,
               "seed": self.seed, "models": {k: m.state_dict() for k, m in self.models.items()},
               "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
               "updates": self.updates, "dataset_contract": self.dataset_contract, "accepted": False}
        stream = io.BytesIO()
        torch.save(raw, stream)
        atomic_bytes(path, stream.getvalue())

    @classmethod
    def load(cls, path: Path) -> SolverStudents:
        from hybrid.solver_inputs import INPUT_SCHEMA
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if (raw["schema"] != "solver-root-students-v2" or raw["input_schema"] != INPUT_SCHEMA
                or raw["target_schema"] != "average-profile-root-advantage-bb-v1"
                or raw["feature_schema"] != FEATURE_SCHEMA_VERSION or tuple(raw["actions"]) != tuple(ACTION_IDS)):
            raise ValueError("Incompatible solver student checkpoint; explicit new run required")
        result = cls(raw["player_count"], residual=raw["residual"], seed=raw["seed"],
                     continuations=tuple(raw["continuations"]), conditioned=raw["conditioned"])
        for key, model in result.models.items():
            model.load_state_dict(raw["models"][key], strict=True)
        for key, optimizer in result.optimizers.items():
            optimizer.load_state_dict(raw["optimizers"][key])
        result.updates, result.dataset_contract = raw["updates"], raw["dataset_contract"]
        return result
