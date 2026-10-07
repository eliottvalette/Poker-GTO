"""Bounded preflop work and weight diagnostics for three-player collectors."""
from __future__ import annotations

import math
import random
import time

from cfr_solver import Traversal
from poker_game_expresso import HandState
from scripts.strategy_collector_audit import (COLLECTORS, audit_collectors, varying_policy,
                                              exact_reference, enumerated_expectation, small_root)
from actions import ACTION_IDS


def collect_root(root: HandState, collector: str, iteration: int, trial: int) -> dict:
    records = []
    node_counts = []
    started = time.perf_counter()
    for player in root.players:
        rng = random.Random(190000 + iteration * 1000 + trial * 3 + player)
        policy = varying_policy(iteration)
        orientations = [p for p in root.players if p != player] if collector == "partial_enumeration" else [None]
        for opponent in orientations:
            walk = Traversal(policy, rng, max_nodes=10000, max_depth=100)

            def sink(obs, target, weight):
                records.append({"player": obs.hero, "infoset": obs.key(), "target": target,
                                "weight": weight * iteration / len(orientations)})

            if collector == "partial_enumeration":
                walk.average_partial(root, player, opponent, sink)
            elif collector == "uniform_importance":
                walk.average(root, player, sink)
            elif collector == "opponent_nodes":
                walk.regrets(root, player, lambda *_: None, strategy_sink=sink)
            else:
                raise ValueError(f"Unknown collector: {collector}")
            node_counts.append(walk.nodes)
    return {"iteration": iteration, "trial": trial, "nodes_by_traversal": node_counts,
            "seconds": time.perf_counter() - started, "records": records}


def audit_preflop(trials: int = 12) -> dict:
    if type(trials) is not int or trials < 1:
        raise ValueError(f"Positive trial count required: {trials}")
    result = {}
    for stack in (1.5, 5.0, 15.0):
        root = HandState.start(dict.fromkeys(range(3), stack), 0, random.Random(3))
        result[str(stack)] = {}
        for collector in COLLECTORS:
            runs = [collect_root(root, collector, iteration, trial)
                    for iteration in (1, 2, 3) for trial in range(trials)]
            weights = [record["weight"] for run in runs for record in run["records"]]
            total = math.fsum(weights)
            nodes = sum(sum(run["nodes_by_traversal"]) for run in runs)
            ess = total * total / math.fsum(w * w for w in weights)
            summary = {"records": len(weights), "nodes": nodes, "weight_ess": ess,
                       "ess_fraction": ess / len(weights), "ess_per_1000_nodes": ess * 1000 / nodes,
                       "largest_weight_share": max(weights) / total,
                       "top10_weight_share": math.fsum(sorted(weights, reverse=True)[:10]) / total,
                       "seconds": sum(run["seconds"] for run in runs),
                       "per_iteration_total_mass_variance": {}}
            for iteration in (1, 2, 3):
                totals = [math.fsum(r["weight"] for r in run["records"])
                          for run in runs if run["iteration"] == iteration]
                mean = math.fsum(totals) / len(totals)
                summary["per_iteration_total_mass_variance"][str(iteration)] = (
                    math.fsum((v - mean) ** 2 for v in totals) / max(1, len(totals) - 1))
            result[str(stack)][collector] = {"summary": summary, "runs": runs}
            print(f"preflop stack={stack} collector={collector}: records={len(weights)}, "
                  f"ESS={ess:.1f}, nodes={nodes}, seconds={summary['seconds']:.2f}", flush=True)
    return result


def exact_orientation_audit() -> dict:
    def sparse_policy(iteration):
        def policy(obs):
            legal = [i for i, valid in enumerate(obs.legal_mask) if valid]
            chosen = legal[(iteration + obs.hero + len(obs.history)) % len(legal)]
            return tuple(float(i == chosen) for i in range(len(ACTION_IDS)))
        return policy

    root = small_root(3)
    result = {}
    for name, factory in (("varying", varying_policy), ("deterministic_zero_support", sparse_policy)):
        reference = exact_reference(root, factory)["reference"]
        result[name] = {"reference": reference, "orientations": {}}
        for orientation in (0, 1):
            actual = enumerated_expectation(root, "partial_enumeration", factory, orientation)
            error = max(abs(a - e) for key, expected in reference.items()
                        for a, e in zip(actual.get(key, {"action_mass": [0.] * len(ACTION_IDS)})["action_mass"],
                                        expected["action_mass"]))
            if error > 1e-11:
                raise AssertionError(f"Partial collector expectation mismatch: {name}/{orientation}: {error}")
            result[name]["orientations"][str(orientation)] = {"max_action_mass_error": error, "actual": actual}
    return result


def run_audit(trials: int = 12) -> dict:
    report = {"scope": "Exact conditional river reference plus fixed-deal preflop weight/work diagnostics",
            "limitations": ["ESS describes weights, not independence or policy accuracy.",
                            "Finite-sample normalized averages are ratio estimators; exact unbiasedness applies to action mass.",
                            "Both partial orientations are merged with one-half weights.",
                            "No variance guarantee follows for arbitrary deeper trees or learned policies.",
                            "Preflop deals and synthetic varying policies are fixed; no full-game training claim."],
            "exact_rng_enumeration": exact_orientation_audit(),
            "exact_river": audit_collectors(256), "preflop": audit_preflop(trials)}
    intern_preflop_infosets(report)
    return report


def intern_preflop_infosets(report: dict) -> None:
    """Store lossless observation keys once while retaining every emitted record."""
    infosets = {}
    for entries in report["preflop"].values():
        for entry in entries.values():
            for run in entry["runs"]:
                for record in run["records"]:
                    key = record.pop("infoset")
                    record["infoset_id"] = infosets.setdefault(key, len(infosets))
    report["preflop_infosets"] = list(infosets)
