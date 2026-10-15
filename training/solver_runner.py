"""Bounded resumable offline refinement; explicit CFR tables own every update."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import fcntl
from pathlib import Path
import re
import time

from hybrid.knowledge import calculate, game_contract
from hybrid.policy_source import PolicySource
from hybrid.ranges import JointRanges
from hybrid.state import ComputeBudget
from poker_game_expresso import HandState


@dataclass(frozen=True)
class SolverTask:
    name: str
    state: HandState
    ranges: JointRanges
    split: str
    continuation: PolicySource | None = None


class SolverRunner:
    """A finite declared curriculum, with atomic per-game rather than NN state.

    Each call targets an absolute iteration count. An interrupted call resumes
    without replaying completed updates. Disk use and per-game nodes are bounded;
    no neural fitting, publication, silent eviction, or training is implicit.
    """

    def __init__(self, output: Path, tasks: list[SolverTask], budget: ComputeBudget,
                 *, max_storage_bytes: int = 256 * 1024 * 1024) -> None:
        if not tasks or len({t.name for t in tasks}) != len(tasks):
            raise ValueError("A nonempty uniquely named solver curriculum is required")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", t.name) for t in tasks):
            raise ValueError("Solver task names must be safe explicit file identifiers")
        if type(max_storage_bytes) is not int or max_storage_bytes < 1:
            raise ValueError("Positive storage byte budget required")
        if any(t.split not in ("train", "validation", "test") for t in tasks):
            raise ValueError("Every computed game needs an explicit independent split")
        output.mkdir(parents=True, exist_ok=True)
        expected = {f"{t.name}.json" for t in tasks} | {".writer.lock"}
        unexpected = {p.name for p in output.iterdir()} - expected
        if unexpected:
            raise ValueError(f"Unexpected files in solver curriculum: {sorted(unexpected)}")
        for task in tasks:
            path = output / f"{task.name}.json"
            if path.exists() and json.loads(path.read_text())["contract"] != game_contract(
                    task.state, task.ranges, budget, task.split, task.continuation):
                raise ValueError(f"Solver curriculum contract changed: {task.name}")
        self.output, self.tasks, self.budget = output, tuple(tasks), budget
        self.max_storage_bytes = max_storage_bytes

    def refine_to(self, total_iterations: int, *, max_seconds: float | None = None) -> dict:
        with (self.output / ".writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Solver curriculum already has an active writer: {self.output}") from error
            return self._refine_to(total_iterations, max_seconds=max_seconds)

    def _refine_to(self, total_iterations: int, *, max_seconds: float | None = None) -> dict:
        if type(total_iterations) is not int or total_iterations < 1:
            raise ValueError("Positive absolute CFR iteration target required")
        if max_seconds is not None and (not 0 < max_seconds < float("inf")):
            raise ValueError("Optional wall budget must be finite and positive")
        started = time.perf_counter()
        report = []
        progress = {}
        for task in self.tasks:
            path = self.output / f"{task.name}.json"
            progress[task.name] = json.loads(path.read_text())["checkpoint"]["iterations"] if path.exists() else 0

        def outcome(reason: str) -> dict:
            return {"status": reason, "complete": all(n >= total_iterations for n in progress.values()),
                    "target_iterations": total_iterations, "completed_iterations": progress,
                    "seconds": time.perf_counter() - started, "work": report}
        # Round-robin chunks keep the curriculum useful when a wall budget stops
        # between tasks. A task itself always respects the configured node cap.
        while True:
            changed = False
            for task in self.tasks:
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    return outcome("wall_budget_reached")
                path = self.output / f"{task.name}.json"
                saved = json.loads(path.read_text()) if path.exists() else None
                completed = saved["checkpoint"]["iterations"] if saved else 0
                if completed >= total_iterations:
                    continue
                used = sum(p.stat().st_size for p in self.output.iterdir() if p != path)
                remaining = self.max_storage_bytes - used
                if remaining <= 0:
                    raise ValueError("Solver knowledge storage budget exhausted")
                budget = replace(self.budget, iterations=min(self.budget.iterations, total_iterations - completed))
                begin = time.perf_counter()
                solution = calculate(path, task.state, task.ranges, budget, split=task.split,
                                     continuation=task.continuation, max_artifact_bytes=remaining)
                changed = True
                progress[task.name] = solution.checkpoint["iterations"]
                row = {"task": task.name, "iterations": solution.checkpoint["iterations"],
                       "nodes": solution.work.nodes, "seconds": time.perf_counter() - begin,
                       "artifact_bytes": path.stat().st_size}
                report.append(row)
                print(f"{task.name}: CFR {row['iterations']}, {row['nodes']} nodes, {row['seconds']:.3f}s", flush=True)
            if not changed:
                return outcome("target_reached")
