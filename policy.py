"""Versioned, lossless tabular average policies. Unknown states are explicit errors."""
from __future__ import annotations
import json
from pathlib import Path
from actions import ACTION_IDS
from cfr_solver import validate_strategy
from infoset import STATE_VERSION, Observation

POLICY_VERSION = 2


class TabularAveragePolicy:
    def __init__(self, entries: dict[str, tuple[float, ...]], objective: str):
        if objective not in ("hand_chip_delta", "tournament_winner"):
            raise ValueError(f"Invalid policy objective: {objective}")
        self.entries = entries
        self.objective = objective

    def query(self, obs: Observation) -> tuple[float, ...]:
        if obs.objective != self.objective:
            raise ValueError(f"Policy objective {self.objective} != observation {obs.objective}")
        if obs.key() not in self.entries:
            raise KeyError("Average policy unavailable: observation has no tabular coverage")
        strategy = self.entries[obs.key()]
        validate_strategy(strategy, obs.legal_mask)
        return strategy

    @classmethod
    def from_solver(cls, solver, objective: str) -> TabularAveragePolicy:
        entries = {k: tuple(v / sum(row) for v in row) for k, row in solver.strategy_sum.items() if sum(row) > 0}
        return cls(entries, objective)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"version": POLICY_VERSION, "state_version": STATE_VERSION,
                                         "actions": ACTION_IDS, "objective": self.objective,
                                         "entries": self.entries}, allow_nan=False))

    @classmethod
    def load(cls, path: str | Path) -> TabularAveragePolicy:
        raw = json.loads(Path(path).read_text())
        if (raw.get("version") != POLICY_VERSION or raw.get("state_version") != STATE_VERSION
                or raw.get("actions") != list(ACTION_IDS)):
            raise ValueError(f"Incompatible policy schema at {path}; expected policy/state v2 and {ACTION_IDS}")
        entries = {k: tuple(v) for k, v in raw["entries"].items()}
        for key, strategy in entries.items():
            obs = json.loads(key)
            validate_strategy(strategy, tuple(obs["legal_mask"]))
        return cls(entries, raw["objective"])
