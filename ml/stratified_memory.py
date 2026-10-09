"""Optional context-stratified replay for the existing Deep CFR learner."""
from dataclasses import replace
import io
from pathlib import Path
import torch
from ml.memory import ReservoirMemory, TrainingSample
from training.metrics import opening_hand_class


class ProtectedReplay:
    """Independent Algorithm-R strata with explicit inverse-inclusion correction.

    Corrected weights estimate the original stream's weighted loss total using
    Horvitz–Thompson inclusion probabilities. A normalized fitted loss remains a
    finite-sample ratio estimator. This opt-in memory does not alter collectors.
    """

    def __init__(
        self,
        opening_capacity: int,
        other_capacity: int,
        *,
        kind: str,
        seed: int,
        byte_budget: int = 64 * 1024 * 1024,
    ) -> None:
        if type(byte_budget) is not int or byte_budget < 2:
            raise ValueError("Protected replay requires at least two budget bytes")
        self.memories = {
            "opening": ReservoirMemory(
                opening_capacity, seed, kind, "hand_chip_delta", byte_budget // 2
            ),
            "other": ReservoirMemory(
                other_capacity, seed + 1, kind, "hand_chip_delta", byte_budget - byte_budget // 2
            ),
        }

    def add(self, sample: TrainingSample) -> None:
        if sample.traversal_mode != "external_sampling":
            raise ValueError("Protected replay supports external_sampling only")
        group = "opening" if opening_hand_class(sample.state) is not None else "other"
        self.memories[group].add(sample)

    @property
    def capacity(self) -> int:
        return sum(m.capacity for m in self.memories.values())

    @property
    def byte_budget(self) -> int:
        return sum(m.byte_budget for m in self.memories.values())

    @property
    def used_bytes(self) -> int:
        return sum(m.used_bytes for m in self.memories.values())

    @property
    def seen(self) -> int:
        return sum(m.seen for m in self.memories.values())

    @property
    def kind(self) -> str:
        return self.memories["opening"].kind

    @property
    def objective(self) -> str:
        return self.memories["opening"].objective

    @property
    def traversal_mode(self) -> str | None:
        return "external_sampling" if self.seen else None

    @property
    def samples(self) -> list[TrainingSample]:
        result = []
        for memory in self.memories.values():
            if memory.samples:
                inclusion = min(1.0, memory.capacity / memory.seen)
                result.extend(
                    replace(s, weight=s.weight / inclusion) for s in memory.samples
                )
        return result

    def diagnostics(self) -> dict:
        return {
            name: {
                "generated": m.seen,
                "retained": len(m.samples),
                "capacity": m.capacity,
                "inclusion_probability": min(1.0, m.capacity / m.seen)
                if m.seen
                else None,
            }
            for name, m in self.memories.items()
        }

    def save(self, path: Path) -> None:
        from training.checkpoint import atomic_bytes, pack_memory

        buffer = io.BytesIO()
        torch.save(
            {
                "schema": 1,
                "strata": {name: pack_memory(m) for name, m in self.memories.items()},
            },
            buffer,
        )
        atomic_bytes(path, buffer.getvalue())

    @classmethod
    def load(cls, path: Path):
        from training.checkpoint import unpack_memory

        raw = torch.load(path, weights_only=True, map_location="cpu")
        if raw["schema"] != 1 or set(raw["strata"]) != {"opening", "other"}:
            raise ValueError("Unsupported protected replay schema/strata")
        result = cls.__new__(cls)
        result.memories = {name: unpack_memory(m) for name, m in raw["strata"].items()}
        return result

