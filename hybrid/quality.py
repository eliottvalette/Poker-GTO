"""Versioned, grouped and matched strategic evaluation on independent worlds."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

from hybrid.beliefs import BeliefState
from hybrid.experiments import opponent_pool
from hybrid.ranges import HandRange, JointRanges, combo
from hybrid.search import search
from hybrid.state import BudgetExceeded, ComputeBudget
from poker_game_expresso import HandState

SUITE_VERSION = "phase2-public-roots-v1"


@dataclass
class Scenario:
    name: str
    split: str
    state: HandState
    beliefs: BeliefState


def scenarios(split: str = "test", per_cell: int = 1, range_shape: str = "narrow") -> list[Scenario]:
    """Independent public roots; changing the split changes cards and histories."""
    if split not in ("train", "validation", "test") or per_cell < 1:
        raise ValueError(
            "Explicit train/validation/test split and positive count required"
        )
    if range_shape not in ("narrow", "wide-polarized"):
        raise ValueError("Unknown range-shape suite")
    offset = {"train": 1000, "validation": 2000, "test": 3000}[split]
    result = []
    for count in (2, 3):
        for street in ("PREFLOP", "FLOP", "TURN", "RIVER"):
            for region, stacks in (
                ("shallow", [3.0, 4.0, 5.0]),
                ("medium", [12.0, 12.0, 12.0]),
                ("deep-asymmetric", [45.0, 25.0, 8.0]),
            ):
                for repeat in range(per_cell):
                    seed = offset + len(result)
                    rng = random.Random(seed)
                    state = HandState.start(
                        dict(enumerate(stacks[:count])), repeat % count, rng
                    )
                    while state.street != street and not state.terminal:
                        state.act("CALL" if state.to_call() else "CHECK")
                    if region == "deep-asymmetric" and state.can_raise():
                        state.act("RAISE", state.min_raise_to + 0.25)
                    available = [c for c in range(52) if c not in state.board]
                    factors = {}
                    for seat, player in state.players.items():
                        rows = {combo(player.cards): 2.0}
                        for _ in range(3 if region == "medium" else 1):
                            rows[combo(tuple(rng.sample(available, 2)))] = rng.choice(
                                (0.25, 1.0, 3.0)
                            )
                        if range_shape == "wide-polarized":
                            rows = dict(HandRange.uniform(tuple(state.board)).weights)
                            if seat % 2:
                                rows = {h: w * (9. if sum(c//4 for c in h) < 7 or sum(c//4 for c in h) > 19 else .1)
                                        for h,w in rows.items()}
                        factors[seat] = HandRange(rows)
                    result.append(
                        Scenario(
                            f"{SUITE_VERSION}/{split}/{count}/{street}/{region}/{repeat}",
                            split,
                            state,
                            BeliefState(JointRanges(factors, tuple(state.board))),
                        )
                    )
    return result


def mean_interval(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "standard_error": None,
            "normal_95_interval": None,
        }
    mean = math.fsum(values) / len(values)
    se = (
        math.sqrt(
            math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1) / len(values)
        )
        if len(values) > 1
        else None
    )
    return {
        "count": len(values),
        "mean": mean,
        "standard_error": se,
        "normal_95_interval": None
        if se is None
        else [mean - 1.96 * se, mean + 1.96 * se],
    }


def evaluate(
    path: Path,
    *,
    samples: tuple[int, ...] = (8, 32),
    reference_samples: int = 128,
    per_cell: int = 1,
    candidates: dict | None = None,
    teacher_family: str = "phase1",
    adaptive: bool = False,
    range_shape: str = "narrow",
) -> dict:
    """Finite-world EV references use independent seeds and disclose their MC error.

    Each row is an independent public-root cluster, not an independent action.
    References describe each explicit teacher, not a claimed equilibrium.
    """
    pool = opponent_pool()
    if teacher_family not in ("phase1", "hand-conditioned"):
        raise ValueError("Unknown evaluation teacher family")
    candidates = candidates or {"uniform": pool["uniform"]}
    rows = []
    for scenario in scenarios(per_cell=per_cell, range_shape=range_shape):
        state, belief = scenario.state, scenario.beliefs
        hero = state.current_player
        private = belief.private_view(hero, combo(state.actor.cards))
        teacher = pool["passive" if "medium" in scenario.name else "aggressive"]
        if teacher_family == "hand-conditioned":
            from hybrid.reference import ProfilePolicy

            teacher = ProfilePolicy(
                "tight_passive" if "medium" in scenario.name else "loose_aggressive"
            )
        ref_budget = ComputeBudget(
            samples=reference_samples, max_nodes=500000, max_depth=1, seed=7819
        )
        reference = search(
            state, private, hero, dict.fromkeys(state.players, teacher), ref_budget
        )
        for name, policy in candidates.items():
            selected_policy = policy(scenario) if callable(policy) else policy
            for n in samples:
                started = time.perf_counter()
                row = {
                    "scenario": scenario.name,
                    "candidate": name,
                    "teacher": teacher.version,
                    "samples_requested": n,
                    "reference_samples": reference_samples,
                    "reference_ev": reference.action_ev,
                    "reference_se": reference.standard_errors,
                }
                try:
                    predicted = search(
                        state,
                        private,
                        hero,
                        dict.fromkeys(state.players, selected_policy),
                        ComputeBudget(
                            samples=n, max_nodes=100000, max_depth=1, seed=9181
                        ),
                        adaptive=adaptive,
                    )
                    errors = [
                        predicted.action_ev[a] - reference.action_ev[a]
                        for a in reference.action_ev
                    ]
                    chosen = sum(
                        p * reference.action_ev[a]
                        for a, p in predicted.probabilities.items()
                    )
                    row.update(
                        ev=predicted.action_ev,
                        probabilities=predicted.probabilities,
                        mean_squared_ev_error=sum(v * v for v in errors) / len(errors),
                        ev_bias=sum(errors) / len(errors),
                        reference_policy_value=chosen,
                        reference_action_loss=max(reference.action_ev.values())
                        - chosen,
                        nodes=predicted.work.nodes,
                        samples=predicted.work.samples,
                        failure=None,
                        policy_version=selected_policy.version,
                        warnings=predicted.warnings,
                    )
                except BudgetExceeded as error:
                    row["failure"] = str(error)
                row["seconds"] = time.perf_counter() - started
                rows.append(row)
    report = {
        "suite": SUITE_VERSION,
        "teacher_family": teacher_family,
        "range_shape": range_shape,
        "adaptive": adaptive,
        "split": "test",
        "budgets": {
            "samples": samples,
            "reference_samples": reference_samples,
            "per_cell": per_cell,
        },
        "rows": rows,
        "scope": "independent-root clustered diagnostics; MC references have error; no general poker-strength claim",
        "summary": {
            name: {
                "by_budget": {
                    n: {
                        "mse": mean_interval(
                            [
                                r["mean_squared_ev_error"]
                                for r in rows
                                if r["candidate"] == name
                                and r["samples_requested"] == n
                                and not r["failure"]
                            ]
                        ),
                        "action_loss": mean_interval(
                            [
                                r["reference_action_loss"]
                                for r in rows
                                if r["candidate"] == name
                                and r["samples_requested"] == n
                                and not r["failure"]
                            ]
                        ),
                    }
                    for n in samples
                },
                "failure_rate": sum(
                    bool(r["failure"]) for r in rows if r["candidate"] == name
                )
                / sum(r["candidate"] == name for r in rows),
            }
            for name in candidates
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def save_scenarios(path: Path, split: str) -> None:
    records = []
    for item in scenarios(split):
        state = asdict(item.state)
        state["pending"] = sorted(item.state.pending)
        records.append(
            {
                "name": item.name,
                "split": split,
                "state": state,
                "ranges": {
                    p: [{"cards": h, "probability": w} for h, w in r.weights.items()]
                    for p, r in item.beliefs.public.ranges.items()
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")
