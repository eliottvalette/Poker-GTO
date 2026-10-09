"""Bounded Algorithm R reservoir sampling, with lossless versioned persistence."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields, is_dataclass
import gzip
import json
import math
from pathlib import Path
import random
import sys
import os
import tempfile
from actions import ACTION_IDS
from cfr_solver import validate_strategy
from infoset import STATE_VERSION
from features import FEATURE_SCHEMA_VERSION
from features.neural import NeuralObservation, neural_observation

MEMORY_VERSION = 6
DEFAULT_BYTE_BUDGET = 64 * 1024 * 1024
TRAVERSAL_MODES = ("external_sampling", "outcome_sampling")


def sample_bytes(sample: object) -> int:
    """Conservative Python-owned accounting, stable across neural serialization.

    Count scalar occurrences independently so pickle interning cannot change
    budgets. Deduplicate containers within one entry; entries remain independent.
    Excludes allocator overhead, tensors and process RSS.
    """
    if type(sample) is TrainingSample and type(sample.state) is NeuralObservation:
        state = sample.state
        containers = (state.cards, state.legal_mask, sample.target)
        if all(type(v) in (tuple, list) and all(type(x) in (int, float, bool) for x in v) for v in containers):
            # Same occurrence-based accounting as the generic walk, including aliased tuples.
            total = sum(map(sys.getsizeof, (sample, state, sample.iteration, sample.player,
                sample.weight, sample.kind, sample.model_version, sample.traversal_mode,
                state.version, state.hero, state.objective, state.street, state.numeric_data,
                state.history_data, state.feature_version)))
            visited = set()
            for value in containers:
                if id(value) not in visited:
                    visited.add(id(value))
                    total += sys.getsizeof(value) + sum(map(sys.getsizeof, value))
            return total
    visited: set[int] = set()

    def size(value: object) -> int:
        if is_dataclass(value) or isinstance(value, (tuple, list, dict)):
            if id(value) in visited:
                return 0
            visited.add(id(value))
        total = sys.getsizeof(value)
        if is_dataclass(value) and not isinstance(value, type):
            total += sys.getsizeof(value.__dict__) if hasattr(value, "__dict__") else 0
            total += sum(size(getattr(value, f.name)) for f in fields(value))
        elif isinstance(value, (tuple, list)):
            total += sum(size(item) for item in value)
        elif isinstance(value, dict):
            total += sum(size(k) + size(v) for k, v in value.items())
        return total

    return size(sample)


@dataclass(frozen=True, slots=True)
class TrainingSample:
    iteration: int
    player: int
    state: NeuralObservation
    target: tuple[float, ...]
    weight: float
    kind: str
    model_version: int
    traversal_mode: str = "external_sampling"

    def __post_init__(self) -> None:
        if not isinstance(self.state, NeuralObservation):
            object.__setattr__(self, "state", neural_observation(self.state))

    def __deepcopy__(self, memo: dict) -> TrainingSample:
        import copy
        state = copy.deepcopy(self.state, memo)
        if (state is self.state and type(self.target) is tuple
                and all(type(v) in (int, float, bool) for v in self.target)
                and all(type(v) in (int, float, str) for v in
                        (self.iteration, self.player, self.weight, self.kind, self.model_version, self.traversal_mode))):
            memo[id(self)] = self
            return self
        result = type(self)(self.iteration, self.player, state, copy.deepcopy(self.target, memo),
                            self.weight, self.kind, self.model_version, self.traversal_mode)
        memo[id(self)] = result
        return result

    def validate(self) -> None:
        if self.state.objective != "hand_chip_delta":
            raise ValueError(f"Only per-hand chip-delta samples are supported, received objective={self.state.objective}")
        if self.traversal_mode not in TRAVERSAL_MODES:
            raise ValueError(f"Invalid sample traversal mode: {self.traversal_mode}; expected {TRAVERSAL_MODES}")
        if self.kind not in ("advantage", "strategy") or self.iteration < 1 or self.model_version != self.iteration - 1:
            raise ValueError(f"Invalid sample metadata: {self.kind}, iteration={self.iteration}, version={self.model_version}")
        if self.player != self.state.hero or self.state.version != STATE_VERSION:
            raise ValueError(f"Sample perspective/schema mismatch: player={self.player}, state={self.state}")
        if len(self.target) != len(ACTION_IDS) or any(not math.isfinite(v) for v in self.target):
            raise ValueError(f"Invalid target vector: {self.target}")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError(f"Invalid sample weight {self.weight}")
        if self.kind == "advantage" and self.traversal_mode == "outcome_sampling" and self.weight != 1.0:
            raise ValueError(f"Outcome advantage targets already contain importance correction; expected weight=1, got {self.weight}")
        if any(v != 0 and not m for v, m in zip(self.target, self.state.legal_mask)):
            raise ValueError(f"Nonzero illegal target: {self.target}")
        if self.kind == "strategy":
            validate_strategy(self.target, self.state.legal_mask)


class ReservoirMemory:
    def __init__(self, capacity: int, seed: int, kind: str, objective: str,
                 byte_budget: int = DEFAULT_BYTE_BUDGET, traversal_mode: str | None = None):
        if objective != "hand_chip_delta":
            raise ValueError(f"Only hand_chip_delta replay is supported; tournament-winner/ICM objective={objective} is unsupported")
        if (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1
                or kind not in ("advantage", "strategy")):
            raise ValueError(f"Invalid memory contract: {capacity}, {kind}, {objective}")
        if not isinstance(byte_budget, int) or isinstance(byte_budget, bool) or byte_budget < 1:
            raise ValueError(f"Memory byte_budget must be a positive integer: {byte_budget}")
        if traversal_mode is not None and traversal_mode not in TRAVERSAL_MODES:
            raise ValueError(f"Invalid memory traversal mode: {traversal_mode}; expected {TRAVERSAL_MODES}")
        self.capacity, self.kind, self.objective = capacity, kind, objective
        self.byte_budget = byte_budget
        self.traversal_mode = traversal_mode
        self.used_bytes = 0
        self._sample_sizes: list[int] = []
        self.rng = random.Random(seed)
        self.seen = 0
        self.samples: list[TrainingSample] = []

    def add(self, sample: TrainingSample) -> None:
        sample.validate()
        if sample.state.feature_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Replay requires feature={FEATURE_SCHEMA_VERSION}, received={sample.state.feature_version}; cannot resume old training")
        if sample.kind != self.kind or sample.state.objective != self.objective:
            raise ValueError(f"Memory expects {self.kind}/{self.objective}, received {sample.kind}/{sample.state.objective}")
        if self.traversal_mode is not None and sample.traversal_mode != self.traversal_mode:
            raise ValueError(f"Memory traversal mode={self.traversal_mode}, received {sample.traversal_mode}; do not mix estimators")
        size = sample_bytes(sample)
        if size > self.byte_budget:
            raise MemoryError(f"Sample requires {size} accounted bytes, memory budget={self.byte_budget}")
        rng_state = self.rng.getstate()
        index = len(self.samples) if len(self.samples) < self.capacity else self.rng.randrange(self.seen + 1)
        replaced_bytes = self._sample_sizes[index] if index < len(self.samples) else 0
        next_bytes = self.used_bytes + size - replaced_bytes
        if index < self.capacity and next_bytes > self.byte_budget:
            self.rng.setstate(rng_state)
            raise MemoryError(f"Reservoir update requires {next_bytes} accounted bytes, budget={self.byte_budget}; "
                              "increase the explicit budget or reduce capacity")
        self.seen += 1
        self.traversal_mode = sample.traversal_mode
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
            self._sample_sizes.append(size)
            self.used_bytes = next_bytes
        else:
            if index < self.capacity:
                self.samples[index] = sample
                self._sample_sizes[index] = size
                self.used_bytes = next_bytes

    def save(self, path: str | Path) -> None:
        path = Path(path)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(descriptor)
        try:
            with gzip.open(temporary, "wt") as file:
                # Stream entries instead of constructing a second full replay.
                metadata = {"version": MEMORY_VERSION, "state_version": STATE_VERSION,
                            "actions": ACTION_IDS, "capacity": self.capacity, "kind": self.kind,
                            "objective": self.objective, "seen": self.seen, "byte_budget": self.byte_budget,
                            "rng": self.rng.getstate(), "traversal_mode": self.traversal_mode}
                file.write(json.dumps(metadata, allow_nan=False)[:-1] + ', "samples":[')
                for index, sample in enumerate(self.samples):
                    if index:
                        file.write(",")
                    item = asdict(sample)
                    item["state"]["numeric_data"] = sample.state.numeric_data.hex()
                    item["state"]["history_data"] = sample.state.history_data.hex()
                    json.dump(item, file, allow_nan=False)
                file.write("]}")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> ReservoirMemory:
        with gzip.open(path, "rt") as file:
            raw = json.load(file)
        if raw.get("version") != MEMORY_VERSION or raw.get("state_version") != STATE_VERSION or raw.get("actions") != list(ACTION_IDS):
            raise ValueError(f"Incompatible memory schema at {path}")
        expected = {"version", "state_version", "actions", "capacity", "kind", "objective", "seen", "rng", "samples", "byte_budget", "traversal_mode"}
        if set(raw) != expected or not isinstance(raw["samples"], list):
            raise ValueError(f"Invalid memory fields at {path}: {sorted(raw)}")
        memory = cls(raw["capacity"], 0, raw["kind"], raw["objective"], raw["byte_budget"], raw["traversal_mode"])
        if (not isinstance(raw["seen"], int) or isinstance(raw["seen"], bool)
                or raw["seen"] < 0 or len(raw["samples"]) != min(memory.capacity, raw["seen"])):
            raise ValueError(f"Invalid reservoir counts at {path}")
        if raw["seen"] > 0 and memory.traversal_mode is None:
            raise ValueError(f"Populated reservoir must declare its traversal mode at {path}")
        for item in raw["samples"]:
            expected_sample = {f.name for f in fields(TrainingSample)}
            if not isinstance(item, dict) or set(item) != expected_sample:
                raise ValueError(f"Invalid sample fields at {path}: {item}")
            o = item["state"]
            if not isinstance(o, dict) or set(o) != {f.name for f in fields(NeuralObservation)}:
                raise ValueError(f"Invalid observation fields at {path}: {o}")
            obs = NeuralObservation(o["version"], o["hero"], o["objective"], tuple(o["cards"]), o["street"],
                                    bytes.fromhex(o["numeric_data"]), tuple(o["legal_mask"]),
                                    bytes.fromhex(o["history_data"]), o["feature_version"])
            memory.add(TrainingSample(item["iteration"], item["player"], obs, tuple(item["target"]),
                                      item["weight"], item["kind"], item["model_version"], item["traversal_mode"]))
        memory.seen = raw["seen"]
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value, list) else value
        memory.rng.setstate(tuples(raw["rng"]))
        return memory
