"""Independent street populations with bounded Algorithm-R memories."""
from __future__ import annotations
from dataclasses import replace
from ml.memory import ReservoirMemory, TrainingSample
from ml.model import STREETS
from ml.train import weight_quality


class StreetReplay:
    def __init__(self, capacities: dict[str, int], byte_budgets: dict[str, int],
                 kind: str, seed: int):
        if set(capacities) != set(STREETS) or set(byte_budgets) != set(STREETS):
            raise ValueError("Explicit capacities and byte budgets required for all four streets")
        self.memories = {name: ReservoirMemory(capacities[name], seed + index, kind,
                                              "hand_chip_delta", byte_budgets[name])
                         for index, name in enumerate(STREETS)}

    def add(self, sample: TrainingSample) -> None:
        if sample.state.street not in range(4) or not sample.trajectory_id or not sample.root_group:
            raise ValueError("Street replay requires public street and trajectory/root provenance")
        self.memories[STREETS[sample.state.street]].add(sample)

    @property
    def kind(self):
        return self.memories[STREETS[0]].kind

    @property
    def objective(self):
        return "hand_chip_delta"

    @property
    def capacity(self):
        return sum(m.capacity for m in self.memories.values())

    @property
    def byte_budget(self):
        return sum(m.byte_budget for m in self.memories.values())

    @property
    def used_bytes(self):
        return sum(m.used_bytes for m in self.memories.values())

    @property
    def seen(self):
        return sum(m.seen for m in self.memories.values())

    @property
    def traversal_mode(self):
        modes = {m.traversal_mode for m in self.memories.values() if m.seen}
        if len(modes) > 1:
            raise ValueError("Mixed traversal modes in street replay")
        return next(iter(modes)) if modes else None

    @property
    def samples(self):
        # This aggregate is for diagnostics. Each street fit uses its own raw
        # reservoir: constant within-street inclusion cancels in its mean loss.
        return [replace(s, weight=s.weight / min(1., m.capacity / m.seen))
                for m in self.memories.values() for s in m.samples]

    def diagnostics(self):
        return {name: {"generated": m.seen, "retained": len(m.samples),
                       "capacity": m.capacity, "bytes": m.used_bytes,
                       "inclusion_probability": min(1., m.capacity / m.seen) if m.seen else None,
                       "independent_trajectories": len({s.trajectory_id for s in m.samples}),
                       "public_root_groups": len({s.root_group for s in m.samples}),
                       **(weight_quality(m.samples) if m.samples else {})}
                for name, m in self.memories.items()}
