"""Scoped finite-game calculations: explicit regrets are the strategic state.

Artifacts are separate from Deep CFR checkpoints. Neural predictions never
generate these targets. Chance forests remain finite approximations.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from hybrid.policy_source import LocallySolvedPolicy, PolicySource, UniformLegalPolicy
from hybrid.ranges import JointRanges
from hybrid.river_solver import RiverSolver, RiverSolution
from hybrid.state import ComputeBudget
from infoset import Observation
from ml.memory import TrainingSample
from poker_game_expresso import HandState
from training.checkpoint import atomic_bytes


def public_root(state: HandState) -> dict:
    raw = asdict(state)
    raw.pop("deck")
    for player in raw["players"].values():
        player.pop("cards")
    raw["pending"] = sorted(raw["pending"])
    return raw


def game_contract(state: HandState, ranges: JointRanges, budget: ComputeBudget, split: str,
                  continuation: PolicySource | None = None) -> dict:
    if split not in ("train", "validation", "test"):
        raise ValueError("An independent root split is required")
    continuation = continuation if continuation is not None else UniformLegalPolicy()
    return json.loads(json.dumps({
        "schema": 1, "root": public_root(state),
        "ranges": {p: [[list(h), w] for h, w in r.weights.items()] for p, r in ranges.ranges.items()},
        "forest": {"samples": budget.samples, "depth": budget.max_depth, "seed": budget.seed},
        "continuation": continuation.version, "split": split,
    }, sort_keys=True, allow_nan=False))


def calculate(path: Path, state: HandState, ranges: JointRanges, budget: ComputeBudget,
              *, split: str, continuation: PolicySource | None = None,
              max_artifact_bytes: int | None = None) -> RiverSolution:
    """Add bounded iterations to this exact declared game and atomically save.

    The range prior and root stay explicit. Every saved strategy belongs to this
    context, not a universal policy table keyed only by an observation.
    """
    continuation = continuation if continuation is not None else UniformLegalPolicy()
    contract = game_contract(state, ranges, budget, split, continuation)
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is not None and previous["contract"] != contract:
        raise ValueError(f"Solver artifact contract changed at {path}; use a distinct explicit experiment")
    solver = RiverSolver(resume=previous["checkpoint"] if previous else None)
    result = (solver.solve(state, ranges, budget) if state.street == "RIVER" else
              solver.solve_chance(state, ranges, budget, continuation=continuation))
    group = hashlib.sha256(json.dumps(contract["root"], sort_keys=True).encode()).hexdigest()
    iterations = result.checkpoint["iterations"]
    records = []
    for key, strategy in result.policy.table.items():
        records.append({"observation": json.loads(key), "strategy": strategy,
                        "regret": [v / iterations / state.blinds.big for v in result.regrets[key]],
                        "strategy_mass": result.strategy_mass[key]})
    payload = {"contract": contract, "root_group": group, "checkpoint": result.checkpoint,
               "records": records, "action_ev_by_hand": {str(h): ev for h, ev in result.action_ev_by_hand.items()},
               "work": {"nodes": result.work.nodes, "samples": result.work.samples,
                        "additional_iterations": result.work.iterations},
               "warnings": result.warnings,
               "target_semantics": "A: cumulative counterfactual regret / iterations / current BB; B: own-reach average strategy",
               "quality": "finite-game approximation; no automatic strategic acceptance"}
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    if max_artifact_bytes is not None and len(encoded) > max_artifact_bytes:
        raise ValueError(f"Computed game exceeds storage budget: {len(encoded)} > {max_artifact_bytes} bytes")
    atomic_bytes(path, encoded)
    return result


def supervision(path: Path, kind: str) -> tuple[dict, list[TrainingSample]]:
    if kind not in ("advantage", "strategy"):
        raise ValueError(f"Unsupported solver supervision kind: {kind}")
    payload = json.loads(path.read_text())
    if payload["contract"]["schema"] != 1:
        raise ValueError("Unsupported solver knowledge schema")
    result = []
    for row in payload["records"]:
        # This is a distinct distillation population, not an importance-weighted
        # external-sampling replay. CFR average targets have already accumulated
        # own reach; zero own-reach averages have no supervision.
        if kind == "strategy" and sum(row["strategy_mass"]) == 0:
            continue
        raw = row["observation"]
        observation = Observation(**{**raw, "cards": tuple(raw["cards"]), "numeric": tuple(raw["numeric"]),
                                     "legal_mask": tuple(raw["legal_mask"]),
                                     "history": tuple(tuple(event) for event in raw["history"])})
        sample = TrainingSample(1, observation.hero, observation,
                                tuple(row["regret" if kind == "advantage" else "strategy"]),
                                1., kind, 0, "finite_game_cfr", payload["root_group"], payload["root_group"])
        sample.validate()
        result.append(sample)
    return payload["contract"], result


def cached_policy(path: Path, state: HandState, ranges: JointRanges) -> LocallySolvedPolicy:
    """Query an explicit table only after checking its public root and prior.

    Off-tree decisions remain explicit KeyErrors. This function makes no
    acceptance or equilibrium claim about the saved iteration budget.
    """
    from cfr_solver import validate_strategy
    payload = json.loads(path.read_text())
    contract = payload["contract"]
    expected = json.loads(json.dumps({"root": public_root(state), "ranges": {
        p: [[list(h), w] for h, w in r.weights.items()] for p, r in ranges.ranges.items()}}, sort_keys=True))
    if contract["schema"] != 1 or any(contract[k] != expected[k] for k in expected):
        raise ValueError("Cached strategy requires its exact public root and private-range prior")
    table = {}
    for row in payload["records"]:
        probabilities = tuple(row["strategy"])
        validate_strategy(probabilities, tuple(row["observation"]["legal_mask"]))
        key = json.dumps(row["observation"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        table[key] = probabilities
    return LocallySolvedPolicy(table, f"computed/{payload['checkpoint']['game_hash']}/{payload['checkpoint']['iterations']}")
