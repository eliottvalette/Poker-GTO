"""Bounded fresh-head/context-objective study on an immutable audited replay."""
from __future__ import annotations

import io
import json
from pathlib import Path
import random
import time

import torch

from hybrid.distillation import ResidualAdvantage, conditional_weights
from ml.health import numerical_health
from ml.model import StreetNetworks
from ml.street_training import grouped_partition
from ml.train import EncodedReplay, _loss_with_weights
from training.chain_audit import replay_metrics
from training.checkpoint import atomic_bytes, unpack_memory
from training.metrics import decision_context


def run(source: Path, output: Path, *, updates: int = 256) -> dict:
    if output.exists():
        raise ValueError(f"Repair study output already exists: {output}")
    if type(updates) is not int or updates < 1:
        raise ValueError("Positive update budget required")
    output.mkdir(parents=True)
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        raw = torch.load(source, map_location="cpu", weights_only=True)
        samples = unpack_memory(raw["preflop"]["advantage"]).samples
        training = [s for s in samples if grouped_partition(s, raw["seed"]) == "train"]
        probe = [s for s in training if decision_context(s.state) == "first_open"][:128]
        if not probe:
            raise ValueError("Frozen replay lacks training openings")
        current = StreetNetworks("advantage")
        current.load_state_dict(raw["advantage_weights"], strict=True)
        report = {"source_iteration": raw["iteration"], "source": str(source), "updates": updates,
                  "current_opening_health": numerical_health(current.specialists["PREFLOP"], probe),
                  "scope": "fixed historical replay, preserved grouped splits; no new CFR or production mutation",
                  "variants": {}}
        for residual in (False, True):
            for balanced in (False, True):
                name = f"{'residual' if residual else 'relu'}-{'conditional' if balanced else 'original'}"
                with torch.random.fork_rng():
                    torch.manual_seed(81819)
                    model = StreetNetworks("advantage")
                    if residual:
                        candidate = ResidualAdvantage()
                        candidate.encoder.load_state_dict(model.specialists["PREFLOP"].encoder.state_dict())
                        model.specialists["PREFLOP"] = candidate
                specialist = model.specialists["PREFLOP"]
                rows = conditional_weights(training) if balanced else training
                encoder = EncodedReplay(rows, specialist.feature_version)
                mean_weight = sum(s.weight * s.iteration for s in rows) / len(rows)
                optimizer = torch.optim.Adam(specialist.parameters(), lr=3e-4)
                trace = [{"update": 0, **numerical_health(specialist, probe)}]
                started = time.perf_counter()
                for step in range(updates):
                    batch = random.Random(81819 + step).choices(rows, k=128)
                    weights = [s.weight * s.iteration / mean_weight for s in batch]
                    optimizer.zero_grad(set_to_none=True)
                    loss = _loss_with_weights(specialist, batch, weights, encoder.select(batch))
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in specialist.parameters()):
                        raise ValueError(f"Nonfinite gradients: {name}/{step}")
                    optimizer.step()
                    if (step + 1) % 64 == 0 or step + 1 == updates:
                        trace.append({"update": step + 1, **numerical_health(specialist, probe)})
                seconds = time.perf_counter() - started
                report["variants"][name] = {"seconds": seconds, "opening_trace": trace,
                    "metrics": replay_metrics(model, samples, raw["seed"]), "accepted": False}
                stream = io.BytesIO()
                torch.save({"schema": "fixed-replay-repair-study-v1", "variant": name, "source": str(source),
                    "model": specialist.state_dict(), "optimizer": optimizer.state_dict(), "updates": updates,
                    "accepted": False}, stream)
                atomic_bytes(output / f"{name}.pt", stream.getvalue())
                print(name, f"{seconds:.2f}s", "opening test loss",
                      report["variants"][name]["metrics"]["partitions"]["test/first_open"]["loss"], flush=True)
        atomic_bytes(output / "report.json", json.dumps(report, indent=2, allow_nan=False).encode())
        return report
    finally:
        torch.set_num_threads(before_threads)
