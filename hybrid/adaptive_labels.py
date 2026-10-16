"""Paired response-value labels with finite-look simultaneous uncertainty.

These labels describe fixed continuation policies, never CFR equilibrium or
historical average strategies. Raw paired worlds are retained for audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Callable, Mapping

from actions import legal_actions
from hybrid.knowledge import game_contract
from hybrid.leaf_evaluation import rollout
from hybrid.policy_source import PolicySource
from hybrid.ranges import JointRanges
from hybrid.state import BudgetExceeded, ComputeBudget, WorkCounter, instantiate
from training.checkpoint import atomic_bytes
from poker_game_expresso import HandState


@dataclass(frozen=True)
class LabelBudget:
    looks: tuple[int, ...] = (64, 256, 1024, 4096)
    alpha: float = .05
    tolerance_bb: float = .05
    max_nodes: int = 1000000
    seed: int = 0

    def __post_init__(self) -> None:
        if (not self.looks or tuple(sorted(set(self.looks))) != self.looks
                or any(type(n) is not int or n < 2 for n in self.looks)
                or not 0 < self.alpha < 1 or not math.isfinite(self.tolerance_bb)
                or self.tolerance_bb < 0 or type(self.max_nodes) is not int or self.max_nodes < 1
                or type(self.seed) is not int):
            raise ValueError("Invalid adaptive-label budget")


def paired_summary(rows: list[dict[str, float]], *, utility_span_bb: float,
                   alpha: float, looks: int, tolerance_bb: float) -> dict:
    """Empirical Bernstein bounds on each difference in [-span, span]."""
    if len(rows) < 2 or not math.isfinite(utility_span_bb) or utility_span_bb <= 0:
        raise ValueError("At least two paired rows and a positive physical payoff span required")
    actions = tuple(rows[0])
    if len(actions) < 2 or any(tuple(row) != actions or any(not math.isfinite(v) for v in row.values()) for row in rows):
        raise ValueError("Every paired row must contain all legal actions")
    if not 0 < alpha < 1 or looks < 1 or tolerance_bb < 0:
        raise ValueError("Invalid simultaneous confidence settings")
    pairs = list(combinations(actions, 2))
    # Each side of each pair at each declared look gets alpha/(2*P*L).
    log_term = math.log(4 * len(pairs) * looks / alpha)
    means = {a: statistics.fmean(row[a] for row in rows) for a in actions}
    intervals = {}
    for a, b in pairs:
        differences = [row[a] - row[b] for row in rows]
        if any(abs(d) > utility_span_bb + 1e-9 for d in differences):
            raise ValueError("Paired payoff difference exceeds conserved-chip bound")
        mean, variance = statistics.fmean(differences), statistics.variance(differences)
        radius = math.sqrt(2 * variance * log_term / len(rows)) + 14 * utility_span_bb * log_term / (3 * (len(rows) - 1))
        intervals[a, b] = (max(-utility_span_bb, mean - radius), min(utility_span_bb, mean + radius))
    best = max(means, key=means.get)
    upper_loss, unique = 0., True
    for action in actions:
        if action == best:
            continue
        if (action, best) in intervals:
            low, high = intervals[action, best]
        else:
            lo, hi = intervals[best, action]
            low, high = -hi, -lo
        upper_loss = max(upper_loss, high)
        unique = unique and high < 0
    status = "resolved" if unique else "epsilon_optimal" if upper_loss <= tolerance_bb else "indeterminate"
    return {"action_ev_bb": means, "selected_action": best, "status": status,
            "selection_loss_upper_bb": upper_loss, "samples": len(rows),
            "pairs": [{"actions": [a, b], "difference_bb": means[a] - means[b],
                       "interval_bb": list(intervals[a, b]),
                       "paired_se_bb": math.sqrt(statistics.variance([r[a]-r[b] for r in rows]) / len(rows))}
                      for a, b in pairs]}


def generate(public: HandState, ranges: JointRanges, holding: tuple[int, int],
             policies: Mapping[int, PolicySource], budget: LabelBudget,
             *, cancelled: Callable[[], bool] = lambda: False) -> dict:
    actor = public.current_player
    if public.terminal or set(policies) != set(public.players) or set(ranges.ranges) != set(public.players) or ranges.board != tuple(public.board):
        raise ValueError("Adaptive labels require a live matching public state, ranges and all continuation policies")
    private = ranges.conditioned(actor, holding)
    actions = legal_actions(public)
    work = WorkCounter(ComputeBudget(max_nodes=budget.max_nodes, samples=budget.looks[-1], seed=budget.seed), cancelled)
    rng = random.Random(budget.seed)
    rows, history = [], []
    started = time.process_time()
    summary = None
    reason = "sample_budget"
    for sample in range(1, budget.looks[-1] + 1):
        if cancelled():
            reason = "cancelled"
            break
        world = instantiate(public, private.sample(rng), rng)
        continuation_seed = rng.randrange(2**63)
        row = {}
        try:
            for action in actions:
                work.visit()
                child = world.clone()
                action.apply(child)
                row[action.action_id] = rollout(child, actor, policies, random.Random(continuation_seed), work) / public.blinds.big
        except BudgetExceeded:
            reason = "cancelled" if cancelled() else "node_budget"
            break
        rows.append(row)
        work.samples += 1
        if sample in budget.looks:
            summary = paired_summary(rows, utility_span_bb=public.total_chips / public.blinds.big,
                                     alpha=budget.alpha, looks=len(budget.looks), tolerance_bb=budget.tolerance_bb)
            history.append(summary)
            if summary["status"] != "indeterminate":
                reason = "precision_reached"
                break
    # Only scheduled looks have coverage. Do not issue a new interval at a
    # path-cost-dependent node cutoff; retain the last scheduled estimate.
    contract = game_contract(public, ranges, work.budget, "train")
    return {"schema": "paired-response-label-v1", "budget": asdict(budget),
            "scope": "fixed-policy action response; not a CFR average-strategy target",
            "public_root": contract["root"], "public_ranges": contract["ranges"],
            "continuations": {str(p): policy.version for p, policy in policies.items()},
            "holding": list(sorted(holding)), "actor": actor,
            "stop_reason": reason, "estimate": summary, "looks": history, "paired_worlds": rows,
            "work": {"nodes": work.nodes, "completed_worlds": work.samples,
                     "reported_worlds": summary["samples"] if summary else 0,
                     "cpu_seconds": time.process_time() - started},
            "uncertainty_scope": "sampling only; fixed ranges, fixed policies, independent seeded worlds",
            "accepted_for_strategy_distillation": False}


def save(path: Path, result: dict) -> None:
    if result.get("schema") != "paired-response-label-v1":
        raise ValueError("Unsupported adaptive-label schema")
    atomic_bytes(path, json.dumps(result, allow_nan=False, separators=(",", ":")).encode())


def response_target(result: dict) -> dict:
    """Explicit consumer gate; unresolved observations remain auditable data."""
    estimate = result.get("estimate")
    if result.get("schema") != "paired-response-label-v1" or estimate is None or estimate["status"] == "indeterminate":
        raise ValueError("Response target is unresolved; retain uncertainty or allocate more computation")
    reference = next(iter(estimate["action_ev_bb"]))
    return {"reference_action": reference,
            "relative_values_bb": {a: v - estimate["action_ev_bb"][reference] for a, v in estimate["action_ev_bb"].items()},
            "pair_intervals": estimate["pairs"], "loss_upper_bb": estimate["selection_loss_upper_bb"],
            "semantics": "fixed-policy root action values", "continuations": result["continuations"]}
